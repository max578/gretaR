# stan_backend.R — Stan code generation and execution via cmdstanr
#
# Translates a compiled gretaR_model into Stan code, then uses cmdstanr
# to compile and run inference. Returns gretaR_fit objects.

# =============================================================================
# Stan distribution name mapping
# =============================================================================

.stan_dist_map <- list(
  normal = list(stan = "normal", params = c("mean", "sd")),
  half_normal = list(stan = "normal", params = c("sd"), lower = 0,
                     transform = function(p) paste0("normal(0, ", p["sd"], ")")),
  half_cauchy = list(stan = "cauchy", params = c("scale"), lower = 0,
                     transform = function(p) paste0("cauchy(0, ", p["scale"], ")")),
  student_t = list(stan = "student_t", params = c("df", "mu", "sigma")),
  uniform = list(stan = "uniform", params = c("lower", "upper")),
  bernoulli = list(stan = "bernoulli", params = c("prob")),
  binomial = list(stan = "binomial", params = c("size", "prob")),
  poisson = list(stan = "poisson", params = c("rate")),
  gamma = list(stan = "gamma", params = c("shape", "rate")),
  beta = list(stan = "beta", params = c("alpha", "beta")),
  exponential = list(stan = "exponential", params = c("rate")),
  lognormal = list(stan = "lognormal", params = c("meanlog", "sdlog")),
  cauchy = list(stan = "cauchy", params = c("location", "scale")),
  negative_binomial = list(stan = "neg_binomial", params = c("size", "prob")),
  dirichlet = list(stan = "dirichlet", params = c("concentration"))
)

# =============================================================================
# Main code generator
# =============================================================================

#' @title Generate Stan Code from a gretaR Model
#'
#' @description Translate a compiled `gretaR_model` into Stan code suitable
#'   for use with `cmdstanr`. The generated code preserves the model structure
#'   defined via the gretaR DSL.
#'
#' @param model A `gretaR_model` object created by [model()].
#' @return A character string containing valid Stan code.
#'
#' @export
#' @examples
#' \dontrun{
#' mu <- normal(0, 10)
#' sigma <- half_cauchy(2)
#' y <- as_data(rnorm(50, 3, 1.5))
#' distribution(y) <- normal(mu, sigma)
#' m <- model(mu, sigma)
#' cat(compile_to_stan(m))
#' }
compile_to_stan <- function(model) {

  # Collect information from the model
  data_block <- list()
  param_block <- list()
  tparam_block <- list()
  model_block <- list()

  data_arrays <- list()      # name → tensor value (for passing to Stan)
  param_names <- list()      # var_id → stan_name
  node_stan_names <- list()  # node_id → stan_name (lookup table, not on R6)

  # --- Identify data nodes and their Stan names ---
  # Identify which data nodes are used as indices (parents of index_select ops)
  index_data_ids <- character(0)
  for (nid in names(model$dag_nodes)) {
    node <- model$dag_nodes[[nid]]
    if (node$node_type == "operation" && identical(node$op_type, "index_select")) {
      # Second parent is the index
      if (length(node$parents) >= 2) {
        index_data_ids <- c(index_data_ids, node$parents[2])
      }
    }
  }

  # Identify data nodes used in discrete likelihoods
  discrete_data_ids <- character(0)
  for (data_id in names(model$likelihood_terms)) {
    dist_arr <- model$likelihood_terms[[data_id]]
    dist_nd <- if (inherits(dist_arr, "gretaR_array")) get_node(dist_arr) else dist_arr
    if (!is.null(dist_nd) && !is.null(dist_nd$distribution)) {
      dname <- dist_nd$distribution$name
      if (dname %in% c("bernoulli", "binomial", "poisson", "negative_binomial")) {
        discrete_data_ids <- c(discrete_data_ids, data_id)
      }
    }
  }

  for (nid in names(model$dag_nodes)) {
    node <- model$dag_nodes[[nid]]
    if (node$node_type == "data") {
      stan_name <- paste0("data_", gsub("node_", "", nid))
      n_rows <- node$dim_[1]
      n_cols <- node$dim_[2]
      vals <- as.numeric(node$value$cpu())

      is_index <- nid %in% index_data_ids
      is_discrete <- nid %in% discrete_data_ids

      if (is_index) {
        # Integer index array
        data_block[[length(data_block) + 1]] <- sprintf("  array[%d] int %s;", n_rows, stan_name)
        data_arrays[[stan_name]] <- as.integer(vals)
      } else if (is_discrete) {
        # Integer outcome array
        data_block[[length(data_block) + 1]] <- sprintf("  array[%d] int %s;", n_rows, stan_name)
        data_arrays[[stan_name]] <- as.integer(vals)
      } else if (n_cols == 1) {
        data_block[[length(data_block) + 1]] <- sprintf("  vector[%d] %s;", n_rows, stan_name)
        data_arrays[[stan_name]] <- vals
      } else {
        data_block[[length(data_block) + 1]] <- sprintf("  matrix[%d, %d] %s;", n_rows, n_cols, stan_name)
        data_arrays[[stan_name]] <- vals
      }
      node_stan_names[[nid]] <- stan_name
    }
  }

  # Add N (number of observations) — inferred from first likelihood data
  n_obs <- NULL
  for (data_id in names(model$likelihood_terms)) {
    data_node <- model$dag_nodes[[data_id]]
    if (!is.null(data_node)) {
      n_obs <- data_node$dim_[1]
      break
    }
  }
  if (!is.null(n_obs)) {
    data_block <- c(list(sprintf("  int<lower=1> N;")), data_block)
    data_arrays[["N"]] <- n_obs
  }

  # --- Declare parameters ---
  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    vnode <- model$free_vars[[vid]]
    stan_name <- info$name

    # Determine bounds
    constraint <- vnode$constraint
    bounds <- ""
    if (!is.null(constraint)) {
      lower <- constraint$lower
      upper <- constraint$upper
      if (is.finite(lower) && is.finite(upper)) {
        bounds <- sprintf("<lower=%s, upper=%s>", format_stan_num(lower), format_stan_num(upper))
      } else if (is.finite(lower)) {
        bounds <- sprintf("<lower=%s>", format_stan_num(lower))
      } else if (is.finite(upper)) {
        bounds <- sprintf("<upper=%s>", format_stan_num(upper))
      }
    }

    # Type declaration
    if (info$n_elem == 1) {
      param_block[[length(param_block) + 1]] <- sprintf("  real%s %s;", bounds, stan_name)
    } else {
      d <- info$dim
      if (length(d) == 2 && d[2] == 1) {
        param_block[[length(param_block) + 1]] <- sprintf("  vector%s[%d] %s;", bounds, d[1], stan_name)
      } else if (length(d) == 2) {
        param_block[[length(param_block) + 1]] <- sprintf("  matrix%s[%d, %d] %s;", bounds, d[1], d[2], stan_name)
      }
    }

    param_names[[vid]] <- stan_name
  }

  # --- Priors (model block) ---
  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    if (!is.null(info$distribution)) {
      prior_stmt <- generate_stan_prior(info$name, info$distribution, info$n_elem)
      if (!is.null(prior_stmt)) {
        model_block[[length(model_block) + 1]] <- paste0("  ", prior_stmt)
      }
    }
  }

  # --- Likelihood (model block) ---
  for (data_id in names(model$likelihood_terms)) {
    dist_array <- model$likelihood_terms[[data_id]]
    dist_node <- if (inherits(dist_array, "gretaR_array")) get_node(dist_array) else dist_array
    if (is.null(dist_node) || is.null(dist_node$distribution)) next

    data_node <- model$dag_nodes[[data_id]]
    data_stan <- node_stan_names[[data_id]] %||% paste0("data_", gsub("node_", "", data_id))

    # Generate the likelihood expression
    lik_expr <- generate_stan_likelihood(
      data_stan, dist_node$distribution, model$dag_nodes, param_names, node_stan_names
    )
    if (!is.null(lik_expr)) {
      model_block[[length(model_block) + 1]] <- paste0("  ", lik_expr)
    }
  }

  # --- Generate transformed parameters for operation nodes ---
  # (For now, inline operations in the likelihood expression)

  # --- Assemble Stan code ---
  code <- paste0(
    "data {\n",
    paste(data_block, collapse = "\n"),
    "\n}\n\n",
    "parameters {\n",
    paste(param_block, collapse = "\n"),
    "\n}\n\n",
    "model {\n",
    paste(model_block, collapse = "\n"),
    "\n}\n"
  )

  attr(code, "data") <- data_arrays
  code
}

# =============================================================================
# Generate Stan prior statement
# =============================================================================

#' @noRd
generate_stan_prior <- function(param_name, distribution, n_elem) {
  dist_name <- distribution$name
  params <- distribution$parameters

  # Resolve numeric parameters
  resolve_to_string <- function(p) {
    if (is.numeric(p)) return(format_stan_num(p))
    if (inherits(p, "torch_tensor")) return(format_stan_num(p$item()))
    if (inherits(p, "gretaR_array")) {
      node <- get_node(p)
      if (!is.null(node$node_name)) return(node$node_name)
      return("0")  # fallback
    }
    format_stan_num(as.numeric(p))
  }

  stan_dist <- switch(dist_name,
    "normal" = sprintf("normal(%s, %s)",
                       resolve_to_string(params$mean),
                       resolve_to_string(params$sd)),
    "half_normal" = sprintf("normal(0, %s)", resolve_to_string(params$sd)),
    "half_cauchy" = sprintf("cauchy(0, %s)", resolve_to_string(params$scale)),
    "student_t" = sprintf("student_t(%s, %s, %s)",
                          resolve_to_string(params$df),
                          resolve_to_string(params$mu),
                          resolve_to_string(params$sigma)),
    "exponential" = sprintf("exponential(%s)", resolve_to_string(params$rate)),
    "gamma" = sprintf("gamma(%s, %s)",
                      resolve_to_string(params$shape),
                      resolve_to_string(params$rate)),
    "beta" = sprintf("beta(%s, %s)",
                     resolve_to_string(params$alpha),
                     resolve_to_string(params$beta)),
    "cauchy" = sprintf("cauchy(%s, %s)",
                       resolve_to_string(params$location),
                       resolve_to_string(params$scale)),
    "lognormal" = sprintf("lognormal(%s, %s)",
                          resolve_to_string(params$meanlog),
                          resolve_to_string(params$sdlog)),
    "uniform" = sprintf("uniform(%s, %s)",
                        resolve_to_string(params$lower),
                        resolve_to_string(params$upper)),
    NULL
  )

  if (is.null(stan_dist)) return(NULL)

  if (n_elem == 1) {
    sprintf("%s ~ %s;", param_name, stan_dist)
  } else {
    # Vector parameter — element-wise prior
    sprintf("to_vector(%s) ~ %s;", param_name, stan_dist)
  }
}

# =============================================================================
# Generate Stan likelihood statement
# =============================================================================

#' @noRd
generate_stan_likelihood <- function(data_name, distribution, dag_nodes, param_names, node_stan_names = list()) {
  dist_name <- distribution$name
  params <- distribution$parameters

  # Build the mean expression by walking the DAG
  resolve_expr <- function(p) {
    if (is.numeric(p)) return(format_stan_num(p))
    if (inherits(p, "torch_tensor")) return(format_stan_num(p$item()))
    if (inherits(p, "gretaR_array")) {
      node <- get_node(p)
      if (!is.null(node)) return(node_to_stan_expr(node, dag_nodes, param_names, node_stan_names))
    }
    "0"
  }

  switch(dist_name,
    "normal" = sprintf("%s ~ normal(%s, %s);",
                       data_name,
                       resolve_expr(params$mean),
                       resolve_expr(params$sd)),
    "bernoulli" = sprintf("%s ~ bernoulli(%s);",
                          data_name, resolve_expr(params$prob)),
    "poisson" = sprintf("%s ~ poisson(%s);",
                        data_name, resolve_expr(params$rate)),
    "gamma" = sprintf("%s ~ gamma(%s, %s);",
                      data_name,
                      resolve_expr(params$shape),
                      resolve_expr(params$rate)),
    "student_t" = sprintf("%s ~ student_t(%s, %s, %s);",
                          data_name,
                          resolve_expr(params$df),
                          resolve_expr(params$mu),
                          resolve_expr(params$sigma)),
    "beta" = sprintf("%s ~ beta(%s, %s);",
                     data_name,
                     resolve_expr(params$alpha),
                     resolve_expr(params$beta)),
    {
      cli_alert_warning("Stan backend: unsupported likelihood distribution '{dist_name}'.")
      NULL
    }
  )
}

# =============================================================================
# Convert a DAG node to a Stan expression string
# =============================================================================

#' @noRd
node_to_stan_expr <- function(node, dag_nodes, param_names, node_stan_names = list()) {
  if (node$node_type == "data") {
    return(node_stan_names[[node$id]] %||% paste0("data_", gsub("node_", "", node$id)))
  }

  if (node$node_type == "variable") {
    return(node$node_name %||% param_names[[node$id]] %||% node$id)
  }

  if (node$node_type == "operation") {
    # Recursively resolve parent expressions
    parent_exprs <- vapply(node$parents, function(pid) {
      pnode <- dag_nodes[[pid]]
      if (is.null(pnode)) return(pid)
      node_to_stan_expr(pnode, dag_nodes, param_names, node_stan_names)
    }, character(1))

    ot <- node$op_type %||% "unknown"

    # --- Binary operations ---
    if (length(parent_exprs) == 2) {
      a <- parent_exprs[1]; b <- parent_exprs[2]
      stan_expr <- switch(ot,
        "binary_+" = sprintf("(%s + %s)", a, b),
        "binary_-" = sprintf("(%s - %s)", a, b),
        "binary_*" = sprintf("(%s .* %s)", a, b),
        "binary_/" = sprintf("(%s ./ %s)", a, b),
        "binary_^" = sprintf("pow(%s, %s)", a, b),
        "matmul"   = sprintf("(%s * %s)", a, b),
        "index_select" = sprintf("%s[%s]", a, b),
        # fallback: try deparse
        sprintf("(%s + %s)", a, b)
      )
      return(stan_expr)
    }

    # --- Unary operations ---
    if (length(parent_exprs) == 1) {
      a <- parent_exprs[1]
      stan_expr <- switch(ot,
        "math_log"     = sprintf("log(%s)", a),
        "math_exp"     = sprintf("exp(%s)", a),
        "math_sqrt"    = sprintf("sqrt(%s)", a),
        "math_abs"     = sprintf("fabs(%s)", a),
        "math_cos"     = sprintf("cos(%s)", a),
        "math_sin"     = sprintf("sin(%s)", a),
        "math_tan"     = sprintf("tan(%s)", a),
        "math_acos"    = sprintf("acos(%s)", a),
        "math_asin"    = sprintf("asin(%s)", a),
        "math_atan"    = sprintf("atan(%s)", a),
        "math_lgamma"  = sprintf("lgamma(%s)", a),
        "math_digamma" = sprintf("digamma(%s)", a),
        "sigmoid"      = sprintf("inv_logit(%s)", a),
        "transpose"    = sprintf("(%s)'", a),
        "sum"          = sprintf("sum(%s)", a),
        "mean"         = sprintf("mean(%s)", a),
        a  # fallback: pass through
      )
      return(stan_expr)
    }

    return("0")
  }

  "0"
}

# =============================================================================
# Stan sampling via cmdstanr
# =============================================================================

#' @title Run Stan Backend for a gretaR Model
#'
#' @description Compile the gretaR model to Stan code, then sample using
#'   CmdStan via the cmdstanr package.
#'
#' @param model A `gretaR_model` object.
#' @param n_samples Number of post-warmup samples per chain.
#' @param warmup Number of warmup iterations per chain.
#' @param chains Number of chains.
#' @param verbose Print progress.
#' @param ... Additional arguments passed to CmdStan's `$sample()`.
#'
#' @return A `gretaR_fit` object.
#' @noRd
stan_sample <- function(model, n_samples = 1000L, warmup = 1000L,
                        chains = 4L, verbose = TRUE, ...) {

  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    cli_abort("Package {.pkg cmdstanr} is required for the Stan backend. Install from https://mc-stan.org/cmdstanr/")
  }

  # Generate Stan code
  stan_code <- compile_to_stan(model)
  stan_data <- attr(stan_code, "data")

  if (verbose) {
    cli_alert_info("Stan backend: generating code...")
  }

  # Write to temp file and compile
  stan_file <- tempfile(fileext = ".stan")
  writeLines(stan_code, stan_file)

  if (verbose) cli_alert_info("Stan backend: compiling model...")

  t0 <- proc.time()
  stan_mod <- cmdstanr::cmdstan_model(stan_file, quiet = !verbose)

  # Prepare data list
  data_list <- stan_data
  # Convert vectors to proper Stan format
  for (nm in names(data_list)) {
    val <- data_list[[nm]]
    if (length(val) == 1 && nm == "N") {
      data_list[[nm]] <- as.integer(val)
    }
  }

  if (verbose) cli_alert_info("Stan backend: sampling...")

  fit <- stan_mod$sample(
    data = data_list,
    chains = chains,
    iter_sampling = n_samples,
    iter_warmup = warmup,
    refresh = if (verbose) 200 else 0,
    show_messages = verbose,
    ...
  )

  elapsed <- (proc.time() - t0)[["elapsed"]]

  # Convert to posterior::draws_array
  draws <- posterior::as_draws_array(fit$draws())

  # Filter to only model parameters (exclude lp__, etc.)
  param_names <- make_param_names(model)
  stan_param_names <- intersect(posterior::variables(draws), param_names)
  if (length(stan_param_names) > 0) {
    draws <- posterior::subset_draws(draws, variable = stan_param_names)
  }

  summ <- tryCatch(posterior::summarise_draws(draws), error = function(e) NULL)
  convergence <- build_convergence(draws)

  if (verbose) {
    n_div <- tryCatch(
      sum(fit$diagnostic_summary()$num_divergent),
      error = function(e) 0
    )
    if (n_div > 0) {
      cli_alert_warning("{n_div} divergent transitions (Stan).")
    }
    convergence$n_divergences <- n_div
    cli_alert_success("Stan sampling complete in {round(elapsed, 1)}s.")
  }

  new_gretaR_fit(
    draws = draws,
    model = model,
    summary = summ,
    convergence = convergence,
    call_info = list(
      n_samples = n_samples, warmup = warmup, chains = chains,
      backend = "stan"
    ),
    run_time = elapsed,
    method = "stan",
    extra = list(
      stan_code = stan_code,
      stan_fit = fit,
      par = if (!is.null(summ)) stats::setNames(summ$mean, summ$variable) else NULL
    )
  )
}

#' @title Stan MAP Estimation
#' @noRd
stan_optimize <- function(model, verbose = TRUE, ...) {

  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    cli_abort("Package {.pkg cmdstanr} is required for the Stan backend.")
  }

  stan_code <- compile_to_stan(model)
  stan_data <- attr(stan_code, "data")

  stan_file <- tempfile(fileext = ".stan")
  writeLines(stan_code, stan_file)

  t0 <- proc.time()
  stan_mod <- cmdstanr::cmdstan_model(stan_file, quiet = TRUE)

  data_list <- stan_data
  for (nm in names(data_list)) {
    if (length(data_list[[nm]]) == 1 && nm == "N") {
      data_list[[nm]] <- as.integer(data_list[[nm]])
    }
  }

  fit <- stan_mod$optimize(data = data_list, ...)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  par_vec <- fit$mle()
  param_names <- make_param_names(model)
  stan_params <- intersect(names(par_vec), param_names)
  par_filtered <- par_vec[stan_params]

  new_gretaR_fit(
    draws = NULL,
    model = model,
    summary = NULL,
    convergence = NULL,
    call_info = list(backend = "stan"),
    run_time = elapsed,
    method = "stan_map",
    extra = list(
      par = par_filtered,
      stan_code = stan_code,
      stan_fit = fit
    )
  )
}

# =============================================================================
# Helpers
# =============================================================================

#' @noRd
format_stan_num <- function(x) {
  if (is.integer(x)) return(as.character(x))
  formatted <- format(as.numeric(x), scientific = FALSE, digits = 6)
  trimws(formatted)
}
