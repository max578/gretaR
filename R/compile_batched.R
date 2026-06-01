# compile_batched.R -- M0b: batched-chains log-density (build-beside).
#
# A parallel forward to compile_log_prob() that evaluates the model for ALL
# chains at once: theta is [C, P] and the result is [C] (one log-density per
# chain). The single-chain path (compile_log_prob) is UNTOUCHED and remains the
# correctness reference; this is verified to match it to ~1e-6 (lp + gradient).
#
# Convention: every node value carries a leading chain dim -> shape
# [C, d1, d2] where [d1, d2] is the node's dim_. Data nodes broadcast as
# [1, d1, d2]. Reductions (densities, jacobians) sum the trailing dims, keeping
# the chain dim, so a scalar per-chain log-density [C] falls out. Operations are
# re-expressed as batched torch ops keyed on op_type. Batching makes the chain
# state a single tensor, so there is no shared-mutable-node-state conflict (GS2).
#
# Coverage (M0b first cut): the workhorse model classes -- Gaussian / GLM /
# hierarchical -- i.e. ops {matmul, index_select, binary_+ - * /, math_exp/log,
# sigmoid, transpose} x transforms {identity, log, lower-bound, (scaled-)logit}
# x univariate distributions. Multivariate / gated families (MVN, Dirichlet,
# LKJ, Wishart) abort with a clear message and are the documented follow-on.

# --- batched elementwise log-densities: value [C, m, 1], params broadcast -----
# Each returns the ELEMENTWISE log-density (same shape as x); the caller sums
# the trailing dims. Keeping density and reduction separate is what lets one
# implementation serve both single-chain (sum all) and batched (sum trailing).
.bd <- new.env(parent = emptyenv())

.bd$normal <- function(x, p) {
  mu <- p$mean; sigma <- torch_clamp(p$sd, min = 1e-30)
  z <- (x - mu) / sigma
  -0.9189385332046727 - torch_log(sigma) - 0.5 * z * z
}
.bd$lognormal <- function(x, p) {
  xc <- torch_clamp(x, min = 1e-30); sigma <- torch_clamp(p$sd, min = 1e-30)
  z <- (torch_log(xc) - p$meanlog) / sigma
  -torch_log(xc) - torch_log(sigma) - 0.9189385332046727 - 0.5 * z * z
}
.bd$exponential <- function(x, p) torch_log(p$rate) - p$rate * x
# Uniform density -log(b-a) per element (constrained variables are always in
# support; out-of-support -Inf is handled by the single-chain path / data range).
.bd$uniform <- function(x, p) -torch_log(p$upper - p$lower) * torch_ones_like(x)
.bd$poisson <- function(x, p) x * torch_log(p$rate) - p$rate - torch_lgamma(x + 1)
.bd$bernoulli <- function(x, p) {
  pr <- torch_clamp(p$prob, min = 1e-7, max = 1 - 1e-7)
  x * torch_log(pr) + (1 - x) * torch_log(1 - pr)
}
.bd$cauchy <- function(x, p) {
  sc <- torch_clamp(p$scale, min = 1e-30); z <- (x - p$location) / sc
  -1.1447298858494002 - torch_log(sc) - torch_log(1 + z * z)  # -log(pi) - ...
}
.bd$gamma <- function(x, p) {
  xc <- torch_clamp(x, min = 1e-30)
  p$shape * torch_log(p$rate) - torch_lgamma(p$shape) +
    (p$shape - 1) * torch_log(xc) - p$rate * xc
}
.bd$beta <- function(x, p) {
  xc <- torch_clamp(x, min = 1e-7, max = 1 - 1e-7)
  torch_lgamma(p$alpha + p$beta) - torch_lgamma(p$alpha) - torch_lgamma(p$beta) +
    (p$alpha - 1) * torch_log(xc) + (p$beta - 1) * torch_log(1 - xc)
}

# half-normal / half-cauchy on the positive half-line (constants match the
# single-chain log_prob in distributions.R exactly).
.bd$half_normal <- function(x, p) {
  sigma <- torch_clamp(p$sd, min = 1e-30); z <- x / sigma
  -0.2257914 - torch_log(sigma) - 0.5 * z * z           # log(sqrt(2/pi)) - ...
}
.bd$half_cauchy <- function(x, p) {
  sc <- torch_clamp(p$scale, min = 1e-30)
  log(2) - log(pi) - torch_log(sc) - torch_log(1 + (x / sc)^2)
}

# --- batched transform jacobian (trailing-dim sum), inverse reused as-is ------
.batched_ldj <- function(transform, y) {
  cls <- class(transform)[1]
  s_trailing <- function(t) torch_sum(t, dim = c(2L, 3L))
  if (cls == "IdentityTransform") {
    torch_zeros(y$shape[1], dtype = y$dtype, device = y$device)
  } else if (cls %in% c("LogTransform", "LowerBoundTransform")) {
    s_trailing(y)                                   # log|det J| = sum y
  } else if (cls == "LogitTransform") {
    sg <- torch_sigmoid(y); s_trailing(torch_log(sg) + torch_log(1 - sg))
  } else if (cls == "ScaledLogitTransform") {
    sg <- torch_sigmoid(y); rng <- transform$upper - transform$lower
    s_trailing(log(rng) + torch_log(sg) + torch_log(1 - sg))
  } else if (cls == "SoftplusTransform") {
    s_trailing(torch_log(torch_sigmoid(y)))
  } else {
    cli_abort("Batched jacobian not implemented for {.cls {cls}}.")
  }
}

# --- resolve a distribution's parameters to batched tensors --------------------
.batched_resolve <- function(param, bcompute) {
  if (inherits(param, "gretaR_array") || inherits(param, "GretaRArray")) {
    node <- if (inherits(param, "gretaR_array")) get_node(param) else param
    bcompute(node$id)
  } else if (inherits(param, "torch_tensor")) {
    param$reshape(c(1L, -1L, 1L))
  } else if (is.numeric(param)) {
    torch_tensor(param, dtype = torch_float32())$reshape(c(1L, -1L, 1L))
  } else {
    cli_abort("Cannot batch-resolve a parameter of class {.cls {class(param)}}.")
  }
}

.batched_density <- function(dist, x, bcompute) {
  nm <- dist$name
  fn <- .bd[[nm]]
  if (is.null(fn)) {
    cli_abort(c("Batched density not implemented for distribution {.val {nm}}.",
                "i" = "Multivariate / gated families are the M0b follow-on."))
  }
  # Resolve each named parameter to a batched tensor.
  p <- lapply(dist$parameters, .batched_resolve, bcompute = bcompute)
  elementwise <- fn(x, p)
  torch_sum(elementwise, dim = c(2L, 3L))          # [C]
}

#' Compile a batched (all-chains-at-once) log-density
#'
#' @param model A `gretaR_model`.
#' @return `function(theta_mat)` where `theta_mat` is a `[C, P]` tensor (C
#'   chains, P unconstrained params) returning a length-C tensor of per-chain
#'   log-densities. Pure over the chain batch (no shared mutable node state).
#' @noRd
compile_log_prob_batched <- function(model) {
  var_order <- model$var_order
  param_info <- model$param_info
  dag_nodes <- model$dag_nodes
  dtype <- model$dtype

  function(theta_mat) {
    C <- theta_mat$shape[1]
    cache <- new.env(parent = emptyenv())   # node id -> [C, d1, d2] batched value

    # Seed variable values from theta (inverse-transform to constrained space).
    var_constrained <- list()
    lp <- torch_zeros(C, dtype = dtype)
    for (vid in var_order) {
      info <- param_info[[vid]]
      raw <- theta_mat[, (info$offset + 1L):(info$offset + info$n_elem), drop = FALSE]
      raw <- raw$reshape(c(C, info$n_elem, 1L))    # [C, n_elem, 1]
      tr <- info$transform
      con <- if (!is.null(tr)) tr$inverse(raw) else raw
      cache[[model$free_vars[[vid]]$id]] <- con
      var_constrained[[vid]] <- list(raw = raw, con = con, info = info)
    }

    # Batched DAG evaluation, memoised.
    bcompute <- function(nid) {
      hit <- cache[[nid]]
      if (!is.null(hit)) return(hit)
      node <- dag_nodes[[nid]]
      val <- switch(node$node_type,
        data = {
          v <- node$value
          if (length(dim(v)) == 2) v$reshape(c(1L, v$shape[1], v$shape[2])) else v
        },
        variable = cache[[nid]],
        operation = .batched_op(node, bcompute, C),
        cli_abort("Batched eval: unsupported node type {.val {node$node_type}}.")
      )
      cache[[nid]] <- val
      val
    }

    # Priors + jacobians.
    for (vid in var_order) {
      vc <- var_constrained[[vid]]
      if (!is.null(vc$info$distribution)) {
        lp <- lp + .batched_density(vc$info$distribution, vc$con, bcompute)
      }
      if (!is.null(vc$info$transform) &&
          !inherits(vc$info$transform, "IdentityTransform")) {
        lp <- lp + .batched_ldj(vc$info$transform, vc$raw)
      }
    }

    # Likelihood terms.
    for (data_id in names(model$likelihood_terms)) {
      dist_array <- model$likelihood_terms[[data_id]]
      dist_node <- if (inherits(dist_array, "gretaR_array")) get_node(dist_array) else dist_array
      if (is.null(dist_node) || is.null(dist_node$distribution)) next
      data_node <- dag_nodes[[data_id]]
      if (is.null(data_node)) next
      obs <- data_node$value
      obs <- obs$reshape(c(1L, obs$shape[1], if (length(obs$shape) > 1) obs$shape[2] else 1L))
      lp <- lp + .batched_density(dist_node$distribution, obs, bcompute)
    }

    lp
  }
}

# --- batched operation dispatch (keyed on op_type) ----------------------------
.batched_op <- function(node, bcompute, C) {
  pv <- lapply(node$parents, bcompute)
  ot <- node$op_type %||% "unknown"
  if (length(pv) == 2L) {
    a <- pv[[1]]; b <- pv[[2]]
    switch(ot,
      "binary_+" = a + b, "binary_-" = a - b,
      "binary_*" = a * b, "binary_/" = a / b,
      "binary_^" = torch_pow(a, b),
      "matmul"   = torch_matmul(a, b),    # [C,i,k] x [C/1,k,j] -> [C,i,j]
      "index_select" = {
        # a: [C, n_groups, cols] gathered along dim 2 by 1-based idx in b [.,m,1]
        idx <- b$reshape(c(-1L))$to(dtype = torch_long())
        a$index_select(2L, idx)
      },
      cli_abort("Batched op not implemented: {.val {ot}}.")
    )
  } else if (length(pv) == 1L) {
    a <- pv[[1]]
    switch(ot,
      "math_exp" = torch_exp(a), "math_log" = torch_log(a),
      "math_sqrt" = torch_sqrt(a), "sigmoid" = torch_sigmoid(a),
      "transpose" = a$transpose(2L, 3L),
      cli_abort("Batched unary op not implemented: {.val {ot}}.")
    )
  } else {
    cli_abort("Batched op with {length(pv)} parents not supported.")
  }
}
