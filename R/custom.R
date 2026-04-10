# custom.R — Custom distributions and mixture model helpers

# =============================================================================
# Custom distribution (user-defined log_prob)
# =============================================================================

CustomDistribution <- R6::R6Class(
  "CustomDistribution",
  inherit = GretaRDistribution,

  public = list(
    log_prob_fn = NULL,

    initialize = function(log_prob_fn, constraint = NULL, dim = NULL,
                          name = "custom") {
      constraint <- constraint %||% list(lower = -Inf, upper = Inf)
      self$log_prob_fn <- log_prob_fn
      super$initialize(
        name = name,
        parameters = list(),
        constraint = constraint,
        dim = dim
      )
    },

    log_prob = function(x) {
      self$log_prob_fn(x)
    },

    sample = function(n = 1L) {
      cli_abort("Sampling not available for custom distributions.")
    }
  )
)

#' @title Custom Distribution
#'
#' @description Define a distribution with a user-supplied log-probability
#'   density function. The function must accept a torch tensor and return a
#'   scalar torch tensor. It must be differentiable via torch autograd.
#'
#' @param log_prob_fn A function `f(x) -> scalar torch tensor` computing
#'   the log-probability density at `x`.
#' @param constraint Optional list with `lower` and `upper` bounds for the
#'   parameter space. Determines the transform for HMC sampling.
#' @param dim Dimensions of the variable.
#' @param name Optional name for display purposes.
#'
#' @return A `gretaR_array` representing a variable with the custom distribution.
#' @export
#' @examples
#' \dontrun{
#' # Laplace distribution (not built in)
#' x <- custom_distribution(
#'   log_prob_fn = function(x) -torch_sum(torch_abs(x)),
#'   name = "laplace"
#' )
#'
#' # Truncated normal (positive only)
#' x <- custom_distribution(
#'   log_prob_fn = function(x) {
#'     torch_sum(-0.5 * x^2)  # kernel of N(0,1)
#'   },
#'   constraint = list(lower = 0, upper = Inf),
#'   name = "truncated_normal"
#' )
#' }
custom_distribution <- function(log_prob_fn, constraint = NULL, dim = NULL,
                                name = "custom") {
  if (!is.function(log_prob_fn)) {
    cli_abort("{.arg log_prob_fn} must be a function.")
  }
  dist <- CustomDistribution$new(
    log_prob_fn = log_prob_fn,
    constraint = constraint,
    dim = dim,
    name = name
  )
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Mixture distribution (marginalising over discrete components)
# =============================================================================

MixtureDistribution <- R6::R6Class(
  "MixtureDistribution",
  inherit = GretaRDistribution,

  public = list(
    distributions = NULL,
    weights = NULL,

    initialize = function(distributions, weights, dim = NULL) {
      super$initialize(
        name = "mixture",
        parameters = list(weights = weights),
        constraint = list(lower = -Inf, upper = Inf),
        dim = dim %||% distributions[[1]]$dim
      )
      self$distributions <- distributions
      self$weights <- weights
    },

    log_prob = function(x) {
      w <- resolve_param(self$weights)
      # Ensure weights are on log scale for numerical stability
      log_w <- torch_log(torch_clamp(w, min = 1e-30))

      # Log-sum-exp over components:
      # log p(x) = log sum_k w_k * p_k(x) = logsumexp(log w_k + log p_k(x))
      k <- length(self$distributions)
      log_components <- torch_zeros(k, dtype = x$dtype)
      for (i in seq_len(k)) {
        log_components[i] <- log_w[i] + self$distributions[[i]]$log_prob(x)
      }
      torch_logsumexp(log_components, dim = 1L)
    },

    sample = function(n = 1L) {
      cli_abort("Sampling from mixture distributions not yet implemented.")
    }
  )
)

#' @title Mixture Distribution
#'
#' @description Define a finite mixture of distributions. The mixture is
#'   marginalised over the discrete component indicator using the log-sum-exp
#'   trick, enabling gradient-based inference (HMC/NUTS).
#'
#' @param distributions A list of gretaR distribution objects (e.g.,
#'   `list(normal(mu1, sigma1), normal(mu2, sigma2))`). Each must be a
#'   `gretaR_array` with a distribution attached.
#' @param weights A gretaR_array or numeric vector of mixture weights
#'   (must sum to 1). Typically from `dirichlet()` or `softmax()`.
#'
#' @return A distribution object suitable for use with `distribution()`.
#' @export
#' @examples
#' \dontrun{
#' # Two-component Gaussian mixture
#' w <- dirichlet(c(1, 1))
#' mu1 <- normal(-2, 1); mu2 <- normal(2, 1)
#' sigma <- half_cauchy(1)
#'
#' mix <- mixture(
#'   distributions = list(normal(mu1, sigma), normal(mu2, sigma)),
#'   weights = w
#' )
#' y <- as_data(rnorm(100))
#' distribution(y) <- mix
#' m <- model(w, mu1, mu2, sigma)
#' }
mixture <- function(distributions, weights) {
  if (!is.list(distributions) || length(distributions) < 2) {
    cli_abort("{.arg distributions} must be a list of at least 2 distributions.")
  }

  # Extract distribution objects from gretaR_arrays
  dist_objs <- lapply(distributions, function(d) {
    node <- get_node(d)
    if (is.null(node) || is.null(node$distribution)) {
      cli_abort("Each element of {.arg distributions} must be a gretaR distribution.")
    }
    node$distribution
  })

  mix_dist <- MixtureDistribution$new(
    distributions = dist_objs,
    weights = weights
  )

  # Create a variable node for the mixture (acts as a distribution template)
  node <- GretaRArray$new(
    node_type = "variable",
    distribution = mix_dist,
    dim = dist_objs[[1]]$dim %||% c(1L, 1L)
  )
  node$value <- torch_zeros(node$dim_, dtype = torch_float32())
  wrap_gretaR_array(node)
}
