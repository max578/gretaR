# model.R — Model compilation: DAG → log-joint density function

#' @title Create a gretaR Model
#'
#' @description Compile the computation graph defined by the specified target
#'   variables into a differentiable log-joint-density function suitable for
#'   HMC/NUTS inference.
#'
#' @param ... \code{gretaR_array} objects representing the parameters of interest.
#' @param precision Torch dtype: \code{"float32"} (default) or \code{"float64"}.
#' @return A \code{gretaR_model} object with \code{log_prob()} and
#'   \code{grad_log_prob()} methods.
#' @export
#' @examples
#' \dontrun{
#' alpha <- normal(0, 10)
#' beta <- normal(0, 5)
#' sigma <- half_cauchy(1)
#' y <- as_data(rnorm(100))
#' x <- as_data(rnorm(100))
#' mu <- alpha + beta * x
#' distribution(y) <- normal(mu, sigma)
#' m <- model(alpha, beta, sigma)
#' }
model <- function(..., precision = c("float32", "float64")) {
  precision <- match.arg(precision)
  dtype <- if (precision == "float64") torch_float64() else torch_float32()

  targets <- list(...)

  # Validate inputs
  if (length(targets) == 0L) {
    cli_abort("At least one target variable must be provided to {.fn model}.")
  }

  # Extract target names from the call
  mc <- match.call(expand.dots = FALSE)
  target_names <- vapply(mc[["..."]], deparse, character(1))

  # Collect all free variable nodes (nodes with node_type == "variable")
  free_vars <- list()
  target_ids <- character(0)

  for (i in seq_along(targets)) {
    arr <- targets[[i]]
    if (!inherits(arr, "gretaR_array")) {
      cli_abort("Argument {i} is not a gretaR_array.")
    }
    node <- get_node(arr)
    if (node$node_type != "variable") {
      cli_abort("Argument {target_names[i]} is not a variable node (it is '{node$node_type}').")
    }
    node$node_name <- target_names[i]
    free_vars[[node$id]] <- node
    target_ids <- c(target_ids, node$id)
  }

  # Collect likelihood terms early (needed to exclude distribution template nodes)
  likelihood_terms <- .gretaR_env$distributions

  # Also find any variables referenced in computation that aren't targets
  # (walk the DAG from distributions to find all variables)
  # Exclude variable nodes used as likelihood distribution templates
  likelihood_node_ids <- character(0)
  for (data_id in names(likelihood_terms)) {
    lt <- likelihood_terms[[data_id]]
    lt_node <- if (inherits(lt, "gretaR_array")) get_node(lt) else lt
    if (!is.null(lt_node)) {
      likelihood_node_ids <- c(likelihood_node_ids, lt_node$id)
    }
  }

  all_var_ids <- find_all_variables()
  for (vid in all_var_ids) {
    if (!vid %in% names(free_vars) && !vid %in% likelihood_node_ids) {
      vnode <- .gretaR_env$dag$nodes[[vid]]
      if (is.null(vnode$node_name)) {
        vnode$node_name <- vid  # Use ID as fallback name
      }
      free_vars[[vid]] <- vnode
    }
  }

  # Order: targets first, then auxiliary variables
  var_order <- c(target_ids, setdiff(names(free_vars), target_ids))
  free_vars <- free_vars[var_order]

  # Build parameter info
  param_info <- list()
  total_dim <- 0L
  for (vid in names(free_vars)) {
    vnode <- free_vars[[vid]]
    n_elem <- prod(vnode$dim_)
    param_info[[vid]] <- list(
      name = vnode$node_name,
      dim = vnode$dim_,
      n_elem = n_elem,
      offset = total_dim,
      transform = vnode$transform,
      distribution = vnode$distribution
    )
    total_dim <- total_dim + n_elem
  }

  # Build the compiled model
  compiled <- list(
    free_vars = free_vars,
    param_info = param_info,
    total_dim = total_dim,
    target_ids = target_ids,
    target_names = target_names,
    var_order = var_order,
    likelihood_terms = likelihood_terms,
    dtype = dtype,
    dag_nodes = .gretaR_env$dag$nodes
  )

  structure(compiled, class = "gretaR_model")
}

# =============================================================================
# log_prob and grad_log_prob for a compiled model
# =============================================================================

#' Evaluate the log joint density at a point in unconstrained space
#'
#' @param model A `gretaR_model`.
#' @param theta_free A 1D torch tensor of unconstrained parameter values.
#' @return Scalar torch tensor (log joint density).
#' @noRd
log_prob <- function(model, theta_free) {
  # Unpack parameters from flat vector → individual variables
  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    # Slice the relevant portion
    start <- info$offset + 1L
    end <- info$offset + info$n_elem
    raw <- theta_free[start:end]

    if (prod(info$dim) > 1L) {
      raw <- raw$reshape(info$dim)
    }

    # Apply inverse transform to get constrained value
    vnode <- model$free_vars[[vid]]
    if (!is.null(info$transform)) {
      vnode$value <- info$transform$inverse(raw)
    } else {
      vnode$value <- raw
    }
  }

  lp <- torch_zeros(1, dtype = model$dtype)

  # Prior terms: sum log_prob of each variable's distribution
  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    vnode <- model$free_vars[[vid]]

    if (!is.null(info$distribution)) {
      prior_lp <- info$distribution$log_prob(vnode$value)
      lp <- lp + prior_lp
    }

    # Jacobian adjustment for the transform
    if (!is.null(info$transform) &&
        !inherits(info$transform, "IdentityTransform")) {
      start <- info$offset + 1L
      end <- info$offset + info$n_elem
      raw <- theta_free[start:end]
      if (prod(info$dim) > 1L) raw <- raw$reshape(info$dim)
      lp <- lp + info$transform$log_det_jacobian(raw)
    }
  }

  # Likelihood terms: for each distribution(data) <- dist assignment
  for (data_id in names(model$likelihood_terms)) {
    dist_array <- model$likelihood_terms[[data_id]]
    dist_node <- if (inherits(dist_array, "gretaR_array")) {
      get_node(dist_array)
    } else {
      dist_array
    }
    if (is.null(dist_node)) next

    dist_obj <- dist_node$distribution
    if (is.null(dist_obj)) next

    data_node <- model$dag_nodes[[data_id]]
    if (is.null(data_node)) {
      cli_abort("Data node {data_id} not found in DAG.")
    }

    # Evaluate log_prob of the likelihood distribution at the observed data
    # resolve_param inside log_prob will compute values through the DAG
    obs_value <- data_node$value
    lik_lp <- dist_obj$log_prob(obs_value)
    lp <- lp + lik_lp
  }

  lp
}

#' Compute log_prob and its gradient
#'
#' @param model A `gretaR_model`.
#' @param theta_free A 1D numeric vector or torch tensor.
#' @return List with `lp` (scalar) and `grad` (vector, same length as theta_free).
#' @noRd
grad_log_prob <- function(model, theta_free) {
  if (!inherits(theta_free, "torch_tensor")) {
    theta_free <- torch_tensor(theta_free, dtype = model$dtype)
  }
  theta_free <- theta_free$detach()$requires_grad_(TRUE)

  lp <- log_prob(model, theta_free)

  # Use autograd_grad instead of backward() — avoids grad accumulation overhead
  grads <- autograd_grad(lp, theta_free)

  list(
    lp = lp$item(),
    grad = grads[[1]]
  )
}

#' @title Get the Log Joint Density Function
#'
#' @description Extract a torch-compatible function that evaluates the log
#'   joint density of a compiled \code{gretaR_model} at a given parameter
#'   vector in unconstrained space.
#'
#' @param model A \code{gretaR_model} object created by \code{\link{model}}.
#' @return A function \code{f(theta)} that takes a 1-D torch tensor of
#'   unconstrained parameter values and returns a scalar torch tensor (the
#'   log joint density).
#' @export
#' @examples
#' \dontrun{
#' mu <- normal(0, 10)
#' y <- as_data(rnorm(50, 3))
#' distribution(y) <- normal(mu, 1)
#' m <- model(mu)
#' ld <- joint_density(m)
#' ld(torch::torch_zeros(1))
#' }
joint_density <- function(model) {
  function(theta) log_prob(model, theta)
}

# =============================================================================
# Resolve distribution parameters through the DAG
# =============================================================================

#' @noRd
resolve_distribution_params <- function(dist_obj, dag_nodes) {
  # Create a copy with resolved parameters
  resolved_params <- lapply(dist_obj$parameters, function(p) {
    if (inherits(p, "gretaR_array")) {
      # This is a gretaR_array — compute its current value through the DAG
      node <- get_node(p)
      if (!is.null(node)) {
        return(node$compute())
      }
    }
    if (inherits(p, "GretaRArray")) {
      return(p$compute())
    }
    if (inherits(p, "torch_tensor")) return(p)
    if (is.numeric(p)) return(torch_tensor(p, dtype = torch_float32()))
    p
  })

  # Create a thin wrapper that uses resolved params
  structure(
    list(
      log_prob = function(x) {
        # Re-evaluate using the original dist but with current DAG state
        dist_obj$log_prob(x)
      }
    ),
    class = "resolved_distribution"
  )
}

# =============================================================================
# Find all variable nodes reachable from the current DAG
# =============================================================================

#' @noRd
find_all_variables <- function() {
  var_ids <- character(0)
  for (nid in names(.gretaR_env$dag$nodes)) {
    node <- .gretaR_env$dag$nodes[[nid]]
    if (node$node_type == "variable") {
      var_ids <- c(var_ids, nid)
    }
  }
  var_ids
}

# =============================================================================
# S3 methods for gretaR_model
# =============================================================================

#' @export
print.gretaR_model <- function(x, ...) {
  cat("gretaR model\n")
  cat(sprintf("  Free parameters: %d (%d total elements)\n",
              length(x$var_order), x$total_dim))
  cat("  Variables:\n")
  for (vid in x$var_order) {
    info <- x$param_info[[vid]]
    dist_name <- if (!is.null(info$distribution)) {
      info$distribution$name
    } else {
      "free"
    }
    dim_str <- paste(info$dim, collapse = " x ")
    cat(sprintf("    %s ~ %s [%s]\n", info$name, dist_name, dim_str))
  }
  cat(sprintf("  Likelihood terms: %d\n", length(x$likelihood_terms)))
  invisible(x)
}

#' @export
summary.gretaR_model <- function(object, ...) {
  print(object)
}
