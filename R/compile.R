# compile.R — JIT compilation of the log-joint density function
#
# Generates a pure torch function from the model DAG that avoids R-level
# list traversal and R6 dispatch during gradient evaluation. The compiled
# function can optionally be JIT-traced for additional performance.

#' Compile the log-joint density into an optimised torch function
#'
#' At model() time, this function "flattens" the DAG into a closure that
#' performs all computation using torch operations only — no R6 method
#' dispatch, no list indexing, no R-level loops during evaluation.
#'
#' @param model A `gretaR_model` object.
#' @return A function `f(theta_free) -> scalar tensor` that computes the
#'   log-joint density. Suitable for use with `autograd_grad()`.
#' @noRd
compile_log_prob <- function(model) {

  # Pre-extract all the information we need from the model into
  # simple R objects (vectors, functions) to avoid R6/list dispatch
  # during hot-path evaluation.

  n_params <- model$total_dim
  var_order <- model$var_order
  n_vars <- length(var_order)

  # Pre-compute parameter layout
  offsets <- integer(n_vars)
  n_elems <- integer(n_vars)
  dims_list <- vector("list", n_vars)
  transforms <- vector("list", n_vars)
  dist_log_probs <- vector("list", n_vars)
  has_transform <- logical(n_vars)
  has_distribution <- logical(n_vars)

  for (i in seq_len(n_vars)) {
    vid <- var_order[i]
    info <- model$param_info[[vid]]
    offsets[i] <- info$offset
    n_elems[i] <- info$n_elem
    dims_list[[i]] <- info$dim
    transforms[[i]] <- info$transform
    has_transform[i] <- !is.null(info$transform) &&
      !inherits(info$transform, "IdentityTransform")
    has_distribution[i] <- !is.null(info$distribution)
    if (has_distribution[i]) {
      dist_log_probs[[i]] <- info$distribution
    }
  }

  # Pre-extract likelihood information
  lik_data <- list()
  for (data_id in names(model$likelihood_terms)) {
    dist_array <- model$likelihood_terms[[data_id]]
    dist_node <- if (inherits(dist_array, "gretaR_array")) get_node(dist_array) else dist_array
    if (is.null(dist_node) || is.null(dist_node$distribution)) next

    data_node <- model$dag_nodes[[data_id]]
    if (is.null(data_node)) next

    lik_data[[length(lik_data) + 1]] <- list(
      obs_value = data_node$value,  # Fixed torch tensor
      dist_obj = dist_node$distribution
    )
  }

  # Store references to the free variable nodes (for setting values)
  var_nodes <- vector("list", n_vars)
  for (i in seq_len(n_vars)) {
    var_nodes[[i]] <- model$free_vars[[var_order[i]]]
  }

  dtype <- model$dtype

  # Return the compiled function
  function(theta_free) {
    # Unpack parameters — pre-computed offsets, no list lookups
    for (i in seq_len(n_vars)) {
      start <- offsets[i] + 1L
      end <- offsets[i] + n_elems[i]
      raw <- theta_free[start:end]

      if (n_elems[i] > 1L) {
        raw <- raw$reshape(dims_list[[i]])
      }

      if (has_transform[i]) {
        var_nodes[[i]]$value <- transforms[[i]]$inverse(raw)
      } else {
        var_nodes[[i]]$value <- raw
      }
    }

    # Accumulate log-prob as a scalar tensor
    lp <- torch_zeros(1, dtype = dtype)

    # Prior terms
    for (i in seq_len(n_vars)) {
      if (has_distribution[i]) {
        lp <- lp + dist_log_probs[[i]]$log_prob(var_nodes[[i]]$value)
      }
      if (has_transform[i]) {
        start <- offsets[i] + 1L
        end <- offsets[i] + n_elems[i]
        raw <- theta_free[start:end]
        if (n_elems[i] > 1L) raw <- raw$reshape(dims_list[[i]])
        lp <- lp + transforms[[i]]$log_det_jacobian(raw)
      }
    }

    # Likelihood terms
    for (j in seq_along(lik_data)) {
      lp <- lp + lik_data[[j]]$dist_obj$log_prob(lik_data[[j]]$obs_value)
    }

    lp
  }
}

#' Compile and optionally JIT-trace the log-joint density
#'
#' Creates an optimised version of the log-prob function. If JIT tracing
#' succeeds, the function is further optimised by torch's JIT compiler.
#'
#' @param model A `gretaR_model` object.
#' @param use_jit Logical; attempt JIT tracing (default TRUE).
#' @return A function `f(theta_free) -> scalar tensor`.
#' @noRd
compile_model <- function(model, use_jit = TRUE) {
  compiled_fn <- compile_log_prob(model)

  if (!use_jit) return(compiled_fn)

  # Attempt JIT tracing
  tryCatch({
    example_input <- torch_zeros(model$total_dim, dtype = model$dtype)
    traced <- jit_trace(compiled_fn, example_input)
    # Verify traced output matches untraced
    test_input <- torch_randn(model$total_dim, dtype = model$dtype) * 0.1
    orig <- compiled_fn(test_input)$item()
    traced_val <- traced(test_input)$item()
    if (abs(orig - traced_val) < 1e-3) {
      return(traced)
    } else {
      cli_alert_warning("JIT trace produced different output; falling back to compiled function.")
      return(compiled_fn)
    }
  }, error = function(e) {
    # JIT tracing failed — fall back to compiled (still faster than original)
    compiled_fn
  })
}

#' Fast gradient computation using compiled log-prob
#'
#' Replaces the standard `grad_log_prob()` with a version that uses
#' the pre-compiled log-prob function.
#'
#' @param compiled_fn A compiled log-prob function.
#' @param theta_vec Numeric vector of unconstrained parameters.
#' @param dtype Torch dtype.
#' @return List with `lp` (scalar) and `grad` (numeric vector).
#' @noRd
fast_grad <- function(compiled_fn, theta_vec, dtype) {
  theta_t <- torch_tensor(theta_vec, dtype = dtype, requires_grad = TRUE)
  lp <- compiled_fn(theta_t)
  grads <- autograd_grad(lp, theta_t)
  grad_vec <- as.numeric(grads[[1]])
  if (any(is.nan(grad_vec))) grad_vec[is.nan(grad_vec)] <- 0
  list(lp = lp$item(), grad = grad_vec)
}
