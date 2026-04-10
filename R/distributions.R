# distributions.R — P0 probability distributions for gretaR
#
# Each distribution is an R6 class with:
#   - log_prob(x)  : differentiable log probability density/mass
#   - sample(n)    : draw n samples
#   - constraint   : list(lower, upper) for parameter transforms
#   - parameters   : named list of torch tensors or gretaR_arrays
#
# User-facing constructors return gretaR_array objects that are distribution nodes.

# =============================================================================
# Base distribution class
# =============================================================================

GretaRDistribution <- R6::R6Class(
  "GretaRDistribution",

  public = list(
    name = NULL,
    parameters = NULL,
    constraint = NULL,
    dim = NULL,
    truncation = NULL,

    initialize = function(name, parameters, constraint, dim = NULL) {
      self$name <- name
      self$parameters <- parameters
      self$constraint <- constraint
      self$dim <- dim %||% c(1L, 1L)
    },

    log_prob = function(x) {
      cli_abort("log_prob() not implemented for base distribution.")
    },

    sample = function(n = 1L) {
      cli_abort("sample() not implemented for base distribution.")
    },

    get_transform = function() {
      select_transform(self$constraint$lower, self$constraint$upper)
    }
  )
)

# =============================================================================
# Helper: resolve parameter to torch tensor
# =============================================================================

#' Resolve a parameter (numeric or gretaR_array) to its value for log_prob
#'
#' For gretaR_array parameters, this computes the current value through the DAG.
#' @noRd
resolve_param <- function(x) {
  if (inherits(x, "torch_tensor")) return(x)
  if (inherits(x, "gretaR_array")) {
    node <- get_node(x)
    if (!is.null(node)) return(node$compute())
    return(x$.node$compute())
  }
  if (inherits(x, "GretaRArray")) return(x$compute())
  if (is.numeric(x)) return(torch_tensor(x, dtype = torch_float32()))
  cli_abort("Cannot resolve parameter of class {class(x)}")
}

# =============================================================================
# Normal distribution
# =============================================================================

NormalDistribution <- R6::R6Class(
  "NormalDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(mean, sd, dim = NULL) {
      super$initialize(
        name = "normal",
        parameters = list(mean = mean, sd = sd),
        constraint = list(lower = -Inf, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      mu <- resolve_param(self$parameters$mean)
      sigma <- resolve_param(self$parameters$sd)
      # -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
      z <- (x - mu) / sigma
      torch_sum(-0.9189385 - torch_log(sigma) - 0.5 * z * z)
    },

    sample = function(n = 1L) {
      mu <- resolve_param(self$parameters$mean)
      sigma <- resolve_param(self$parameters$sd)
      mu + sigma * torch_randn(c(n, 1L))
    }
  )
)

#' @title Normal Distribution
#'
#' @description Create a normally-distributed variable with the specified mean
#'   and standard deviation. Support is on the entire real line.
#'
#' @param mean Mean of the distribution (numeric or \code{gretaR_array}).
#' @param sd Standard deviation (numeric or \code{gretaR_array}, positive).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array} representing a normally-distributed variable.
#' @export
#' @examples
#' \dontrun{
#' x <- normal(0, 1)
#' beta <- normal(0, 5, dim = c(3, 1))
#' }
normal <- function(mean = 0, sd = 1, dim = NULL) {
  dist <- NormalDistribution$new(mean = mean, sd = sd, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Half-Normal distribution
# =============================================================================

HalfNormalDistribution <- R6::R6Class(
  "HalfNormalDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(sd, dim = NULL) {
      super$initialize(
        name = "half_normal",
        parameters = list(sd = sd),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      sigma <- resolve_param(self$parameters$sd)
      # log(sqrt(2/pi)) - log(sigma) - 0.5 * (x/sigma)^2, for x >= 0
      z <- x / sigma
      torch_sum(-0.2257914 - torch_log(sigma) - 0.5 * z * z)
    },

    sample = function(n = 1L) {
      sigma <- resolve_param(self$parameters$sd)
      torch_abs(sigma * torch_randn(c(n, 1L)))
    }
  )
)

#' @title Half-Normal Distribution
#'
#' @description Create a half-normal-distributed variable. The half-normal is
#'   the absolute value of a normal distribution, with support on the positive
#'   reals.
#'
#' @param sd Scale parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array} with support on the positive reals.
#' @export
#' @examples
#' \dontrun{
#' sigma <- half_normal(1)
#' }
half_normal <- function(sd = 1, dim = NULL) {
  dist <- HalfNormalDistribution$new(sd = sd, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Half-Cauchy distribution
# =============================================================================

HalfCauchyDistribution <- R6::R6Class(
  "HalfCauchyDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(scale, dim = NULL) {
      super$initialize(
        name = "half_cauchy",
        parameters = list(scale = scale),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      gamma_val <- resolve_param(self$parameters$scale)
      # log(2/(pi*gamma)) - log(1 + (x/gamma)^2), for x >= 0
      torch_sum(
        log(2) - log(pi) - torch_log(gamma_val) -
          torch_log(1 + (x / gamma_val)^2)
      )
    },

    sample = function(n = 1L) {
      gamma_val <- resolve_param(self$parameters$scale)
      torch_abs(gamma_val * torch_tan(torch_rand(c(n, 1L)) * (pi / 2)))
    }
  )
)

#' @title Half-Cauchy Distribution
#'
#' @description Create a half-Cauchy-distributed variable. A popular weakly
#'   informative prior for scale parameters (Gelman, 2006), with support on the
#'   positive reals.
#'
#' @param scale Scale parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array} with support on the positive reals.
#' @export
#' @examples
#' \dontrun{
#' sigma <- half_cauchy(1)
#' tau <- half_cauchy(5)
#' }
half_cauchy <- function(scale = 1, dim = NULL) {
  dist <- HalfCauchyDistribution$new(scale = scale, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Student-t distribution
# =============================================================================

StudentTDistribution <- R6::R6Class(
  "StudentTDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(df, mu, sigma, dim = NULL) {
      super$initialize(
        name = "student_t",
        parameters = list(df = df, mu = mu, sigma = sigma),
        constraint = list(lower = -Inf, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      nu <- resolve_param(self$parameters$df)
      mu <- resolve_param(self$parameters$mu)
      sigma <- resolve_param(self$parameters$sigma)
      z <- (x - mu) / sigma
      # log-pdf of Student-t
      torch_sum(
        torch_lgamma((nu + 1) / 2) - torch_lgamma(nu / 2) -
          0.5 * torch_log(nu * pi) - torch_log(sigma) -
          (nu + 1) / 2 * torch_log(1 + z * z / nu)
      )
    },

    sample = function(n = 1L) {
      nu <- resolve_param(self$parameters$df)
      mu <- resolve_param(self$parameters$mu)
      sigma <- resolve_param(self$parameters$sigma)
      # Use the ratio of normal / sqrt(chi2/df)
      z <- torch_randn(c(n, 1L))
      chi2 <- torch_sum(torch_randn(c(n, as.integer(nu$item())))^2, dim = 2, keepdim = TRUE)
      mu + sigma * z / torch_sqrt(chi2 / nu)
    }
  )
)

#' @title Student-t Distribution
#'
#' @description Create a Student-t-distributed variable with the specified
#'   degrees of freedom, location, and scale. Useful as a robust alternative to
#'   the normal distribution.
#'
#' @param df Degrees of freedom (positive numeric or \code{gretaR_array}).
#' @param mu Location parameter (numeric or \code{gretaR_array}).
#' @param sigma Scale parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' x <- student_t(df = 3, mu = 0, sigma = 1)
#' }
student_t <- function(df = 3, mu = 0, sigma = 1, dim = NULL) {
  dist <- StudentTDistribution$new(df = df, mu = mu, sigma = sigma, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Uniform distribution
# =============================================================================

UniformDistribution <- R6::R6Class(
  "UniformDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(lower, upper, dim = NULL) {
      super$initialize(
        name = "uniform",
        parameters = list(lower = lower, upper = upper),
        constraint = list(lower = lower, upper = upper),
        dim = dim
      )
    },

    log_prob = function(x) {
      a <- resolve_param(self$parameters$lower)
      b <- resolve_param(self$parameters$upper)
      torch_sum(-torch_log(b - a))
    },

    sample = function(n = 1L) {
      a <- resolve_param(self$parameters$lower)
      b <- resolve_param(self$parameters$upper)
      a + (b - a) * torch_rand(c(n, 1L))
    }
  )
)

#' @title Uniform Distribution
#'
#' @description Create a uniformly-distributed variable on the interval
#'   \code{[lower, upper]}.
#'
#' @param lower Lower bound (numeric).
#' @param upper Upper bound (numeric).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' p <- uniform(0, 1)
#' }
uniform <- function(lower = 0, upper = 1, dim = NULL) {
  dist <- UniformDistribution$new(lower = lower, upper = upper, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Bernoulli distribution
# =============================================================================

BernoulliDistribution <- R6::R6Class(
  "BernoulliDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(prob, dim = NULL) {
      super$initialize(
        name = "bernoulli",
        parameters = list(prob = prob),
        constraint = list(lower = 0, upper = 1),
        dim = dim
      )
    },

    log_prob = function(x) {
      p <- resolve_param(self$parameters$prob)
      # Clamp to avoid log(0)
      p <- torch_clamp(p, min = 1e-7, max = 1 - 1e-7)
      torch_sum(x * torch_log(p) + (1 - x) * torch_log(1 - p))
    },

    sample = function(n = 1L) {
      p <- resolve_param(self$parameters$prob)
      torch_bernoulli(p$expand(c(n, 1L)))
    }
  )
)

#' @title Bernoulli Distribution
#'
#' @description Create a Bernoulli-distributed variable with the specified
#'   probability of success. Support is on \code{{0, 1}}.
#'
#' @param prob Probability of success (numeric or \code{gretaR_array}, 0 to 1).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' z <- bernoulli(0.5)
#' }
bernoulli <- function(prob, dim = NULL) {
  dist <- BernoulliDistribution$new(prob = prob, dim = dim)
  create_variable_node(distribution = dist, dim = dim, is_discrete = TRUE)
}

# =============================================================================
# Binomial distribution
# =============================================================================

BinomialDistribution <- R6::R6Class(
  "BinomialDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(size, prob, dim = NULL) {
      super$initialize(
        name = "binomial",
        parameters = list(size = size, prob = prob),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      n <- resolve_param(self$parameters$size)
      p <- resolve_param(self$parameters$prob)
      p <- torch_clamp(p, min = 1e-7, max = 1 - 1e-7)
      # log C(n,x) + x*log(p) + (n-x)*log(1-p)
      torch_sum(
        torch_lgamma(n + 1) - torch_lgamma(x + 1) - torch_lgamma(n - x + 1) +
          x * torch_log(p) + (n - x) * torch_log(1 - p)
      )
    },

    sample = function(n_samples = 1L) {
      n <- resolve_param(self$parameters$size)
      p <- resolve_param(self$parameters$prob)
      torch_binomial(n$expand(c(n_samples, 1L)), p$expand(c(n_samples, 1L)))
    }
  )
)

#' @title Binomial Distribution
#'
#' @description Create a binomially-distributed variable with the specified
#'   number of trials and probability of success per trial.
#'
#' @param size Number of trials (positive integer or \code{gretaR_array}).
#' @param prob Probability of success per trial (numeric or \code{gretaR_array}, 0 to 1).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' y <- binomial(size = 10, prob = 0.3)
#' }
binomial <- function(size, prob, dim = NULL) {
  dist <- BinomialDistribution$new(size = size, prob = prob, dim = dim)
  create_variable_node(distribution = dist, dim = dim, is_discrete = TRUE)
}

# =============================================================================
# Poisson distribution
# =============================================================================

PoissonDistribution <- R6::R6Class(
  "PoissonDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(rate, dim = NULL) {
      super$initialize(
        name = "poisson",
        parameters = list(rate = rate),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      lambda <- resolve_param(self$parameters$rate)
      lambda <- torch_clamp(lambda, min = 1e-7)
      # x*log(lambda) - lambda - log(x!)
      torch_sum(x * torch_log(lambda) - lambda - torch_lgamma(x + 1))
    },

    sample = function(n = 1L) {
      lambda <- resolve_param(self$parameters$rate)
      torch_poisson(lambda$expand(c(n, 1L)))
    }
  )
)

#' @title Poisson Distribution
#'
#' @description Create a Poisson-distributed variable with the specified rate.
#'   Named \code{poisson_dist} to avoid conflict with \code{stats::poisson}.
#'
#' @param rate Rate parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' y <- poisson_dist(rate = 5)
#' }
poisson_dist <- function(rate, dim = NULL) {
  dist <- PoissonDistribution$new(rate = rate, dim = dim)
  create_variable_node(distribution = dist, dim = dim, is_discrete = TRUE)
}

# =============================================================================
# Gamma distribution
# =============================================================================

GammaDistribution <- R6::R6Class(
  "GammaDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(shape, rate, dim = NULL) {
      super$initialize(
        name = "gamma",
        parameters = list(shape = shape, rate = rate),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      alpha <- resolve_param(self$parameters$shape)
      beta <- resolve_param(self$parameters$rate)
      x <- torch_clamp(x, min = 1e-30)
      # alpha*log(beta) - lgamma(alpha) + (alpha-1)*log(x) - beta*x
      torch_sum(
        alpha * torch_log(beta) - torch_lgamma(alpha) +
          (alpha - 1) * torch_log(x) - beta * x
      )
    },

    sample = function(n = 1L) {
      alpha <- resolve_param(self$parameters$shape)
      beta <- resolve_param(self$parameters$rate)
      # torch uses concentration/rate parameterisation
      dist <- torch::distr_gamma(concentration = alpha, rate = beta)
      dist$sample(c(n, 1L))
    }
  )
)

#' @title Gamma Distribution
#'
#' @description Create a gamma-distributed variable with the specified shape
#'   and rate. Support is on the positive reals. Named \code{gamma_dist} to
#'   avoid conflict with \code{base::gamma}.
#'
#' @param shape Shape parameter (positive numeric or \code{gretaR_array}).
#' @param rate Rate parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' tau <- gamma_dist(shape = 2, rate = 1)
#' }
gamma_dist <- function(shape, rate, dim = NULL) {
  dist <- GammaDistribution$new(shape = shape, rate = rate, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Beta distribution
# =============================================================================

BetaDistribution <- R6::R6Class(
  "BetaDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(alpha, beta, dim = NULL) {
      super$initialize(
        name = "beta",
        parameters = list(alpha = alpha, beta = beta),
        constraint = list(lower = 0, upper = 1),
        dim = dim
      )
    },

    log_prob = function(x) {
      a <- resolve_param(self$parameters$alpha)
      b <- resolve_param(self$parameters$beta)
      x <- torch_clamp(x, min = 1e-7, max = 1 - 1e-7)
      torch_sum(
        torch_lgamma(a + b) - torch_lgamma(a) - torch_lgamma(b) +
          (a - 1) * torch_log(x) + (b - 1) * torch_log(1 - x)
      )
    },

    sample = function(n = 1L) {
      a <- resolve_param(self$parameters$alpha)
      b <- resolve_param(self$parameters$beta)
      # Beta via Gamma: X ~ Ga(a,1), Y ~ Ga(b,1), X/(X+Y) ~ Beta(a,b)
      dist_a <- torch::distr_gamma(concentration = a, rate = torch_ones(1))
      dist_b <- torch::distr_gamma(concentration = b, rate = torch_ones(1))
      xa <- dist_a$sample(c(n, 1L))
      xb <- dist_b$sample(c(n, 1L))
      xa / (xa + xb)
    }
  )
)

#' @title Beta Distribution
#'
#' @description Create a beta-distributed variable with the specified shape
#'   parameters. Support is on the interval \code{(0, 1)}. Named
#'   \code{beta_dist} to avoid conflict with \code{base::beta}.
#'
#' @param alpha First shape parameter (positive numeric or \code{gretaR_array}).
#' @param beta Second shape parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' p <- beta_dist(alpha = 2, beta = 5)
#' }
beta_dist <- function(alpha, beta, dim = NULL) {
  dist <- BetaDistribution$new(alpha = alpha, beta = beta, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Exponential distribution
# =============================================================================

ExponentialDistribution <- R6::R6Class(
  "ExponentialDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(rate, dim = NULL) {
      super$initialize(
        name = "exponential",
        parameters = list(rate = rate),
        constraint = list(lower = 0, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      lambda <- resolve_param(self$parameters$rate)
      torch_sum(torch_log(lambda) - lambda * x)
    },

    sample = function(n = 1L) {
      lambda <- resolve_param(self$parameters$rate)
      # Inverse CDF: -log(U)/lambda
      -torch_log(torch_rand(c(n, 1L))) / lambda
    }
  )
)

#' @title Exponential Distribution
#'
#' @description Create an exponentially-distributed variable with the specified
#'   rate. Support is on the positive reals.
#'
#' @param rate Rate parameter (positive numeric or \code{gretaR_array}).
#' @param dim Integer vector of dimensions (default scalar).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' lambda <- exponential(rate = 1)
#' }
exponential <- function(rate = 1, dim = NULL) {
  dist <- ExponentialDistribution$new(rate = rate, dim = dim)
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Multivariate Normal distribution
# =============================================================================

MultivariateNormalDistribution <- R6::R6Class(
  "MultivariateNormalDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(mean, covariance, dim = NULL) {
      super$initialize(
        name = "multivariate_normal",
        parameters = list(mean = mean, covariance = covariance),
        constraint = list(lower = -Inf, upper = Inf),
        dim = dim
      )
    },

    log_prob = function(x) {
      mu <- resolve_param(self$parameters$mean)
      sigma <- resolve_param(self$parameters$covariance)
      k <- mu$shape[1]
      # Cholesky for numerical stability
      L <- torch_linalg_cholesky(sigma)
      diff <- x - mu
      # solve L z = diff
      z <- torch_linalg_solve_triangular(L, diff$unsqueeze(-1), upper = FALSE)
      log_det <- 2 * torch_sum(torch_log(torch_diag(L)))
      # -0.5 * (k*log(2*pi) + log|Sigma| + z^T z)
      -0.5 * (k * 1.8378771 + log_det + torch_sum(z * z))
    },

    sample = function(n = 1L) {
      mu <- resolve_param(self$parameters$mean)
      sigma <- resolve_param(self$parameters$covariance)
      dist <- torch::distr_multivariate_normal(loc = mu, covariance_matrix = sigma)
      dist$sample(n)
    }
  )
)

#' @title Multivariate Normal Distribution
#'
#' @description Create a multivariate-normal-distributed variable with the
#'   specified mean vector and covariance matrix. If \code{dim} is \code{NULL},
#'   it is inferred from the length of \code{mean}.
#'
#' @param mean Numeric mean vector or \code{gretaR_array}.
#' @param covariance Covariance matrix (positive definite numeric matrix or
#'   \code{gretaR_array}).
#' @param dim Integer vector of dimensions (inferred from \code{mean} if
#'   \code{NULL}).
#' @return A \code{gretaR_array}.
#' @export
#' @examples
#' \dontrun{
#' mu <- c(0, 0)
#' Sigma <- diag(2)
#' x <- multivariate_normal(mean = mu, covariance = Sigma)
#' }
multivariate_normal <- function(mean, covariance, dim = NULL) {
  dist <- MultivariateNormalDistribution$new(
    mean = mean, covariance = covariance, dim = dim
  )
  if (is.null(dim)) {
    if (is.numeric(mean)) dim <- c(length(mean), 1L)
  }
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Dirichlet distribution
# =============================================================================

DirichletDistribution <- R6::R6Class(
  "DirichletDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(concentration, dim = NULL) {
      k <- if (is.numeric(concentration)) length(concentration) else NULL
      super$initialize(
        name = "dirichlet",
        parameters = list(concentration = concentration),
        # Simplex constraint — identity transform for now (Phase 3)
        constraint = list(lower = 0, upper = 1, type = "simplex"),
        dim = dim %||% if (!is.null(k)) c(k, 1L) else NULL
      )
    },

    log_prob = function(x) {
      alpha <- resolve_param(self$parameters$concentration)
      # Ensure alpha is 1-D
      if (alpha$dim() > 1) alpha <- alpha$squeeze()
      if (x$dim() > 1) x <- x$squeeze()
      x <- torch_clamp(x, min = 1e-30)
      # log B(alpha) = sum(lgamma(alpha)) - lgamma(sum(alpha))
      # log p(x) = -log B(alpha) + sum((alpha - 1) * log(x))
      torch_lgamma(torch_sum(alpha)) - torch_sum(torch_lgamma(alpha)) +
        torch_sum((alpha - 1) * torch_log(x))
    },

    sample = function(n = 1L) {
      alpha <- resolve_param(self$parameters$concentration)
      if (alpha$dim() > 1) alpha <- alpha$squeeze()
      k <- alpha$shape[1]
      # Sample via normalised Gamma variates
      # Each component: Y_i ~ Gamma(alpha_i, 1), then X = Y / sum(Y)
      samples <- torch_zeros(c(n, k))
      for (i in seq_len(k)) {
        g <- torch::distr_gamma(
          concentration = alpha[i]$unsqueeze(1)$expand(c(n)),
          rate = torch_ones(n)
        )
        samples[, i] <- g$sample()
      }
      row_sums <- torch_sum(samples, dim = 2, keepdim = TRUE)
      samples / row_sums
    }
  )
)

#' Dirichlet distribution
#'
#' Creates a variable distributed according to the Dirichlet distribution,
#' the multivariate generalisation of the Beta distribution. Values lie on
#' the probability simplex (non-negative, sum to one).
#'
#' @param concentration Concentration parameter vector (positive numeric or
#'   gretaR_array). Length determines the dimensionality of the simplex.
#' @param dim Dimensions of the variable (inferred from concentration if NULL).
#' @return A `gretaR_array` with support on the simplex.
#' @export
#' @examples
#' \dontrun{
#' theta <- dirichlet(c(1, 1, 1))
#' theta <- dirichlet(c(2, 5, 1))
#' }
dirichlet <- function(concentration, dim = NULL) {
  dist <- DirichletDistribution$new(concentration = concentration, dim = dim)
  if (is.null(dim) && is.numeric(concentration)) {
    dim <- c(length(concentration), 1L)
  }
  create_variable_node(distribution = dist, dim = dim)
}

# =============================================================================
# Negative Binomial distribution
# =============================================================================

NegativeBinomialDistribution <- R6::R6Class(
  "NegativeBinomialDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(size, prob, dim = NULL) {
      super$initialize(
        name = "negative_binomial",
        parameters = list(size = size, prob = prob),
        constraint = list(lower = 0, upper = Inf, type = "discrete"),
        dim = dim
      )
    },

    log_prob = function(x) {
      r <- resolve_param(self$parameters$size)
      p <- resolve_param(self$parameters$prob)
      p <- torch_clamp(p, min = 1e-7, max = 1 - 1e-7)
      # lgamma(x + r) - lgamma(x + 1) - lgamma(r) + r*log(p) + x*log(1 - p)
      torch_sum(
        torch_lgamma(x + r) - torch_lgamma(x + 1) - torch_lgamma(r) +
          r * torch_log(p) + x * torch_log(1 - p)
      )
    },

    sample = function(n = 1L) {
      r <- resolve_param(self$parameters$size)
      p <- resolve_param(self$parameters$prob)
      # NB via Poisson-Gamma mixture: lambda ~ Gamma(r, p/(1-p)), x ~ Poisson(lambda)
      rate <- p / (1 - torch_clamp(p, max = 1 - 1e-7))
      g <- torch::distr_gamma(concentration = r, rate = rate)
      lambda <- g$sample(c(n, 1L))
      torch_poisson(lambda)
    }
  )
)

#' Negative Binomial distribution
#'
#' Parameterised by the number of successes `size` (r) and the probability
#' of success `prob` (p). Models the number of failures before `size`
#' successes occur.
#'
#' @param size Target number of successes (positive numeric or gretaR_array).
#' @param prob Probability of success per trial (0 to 1).
#' @param dim Dimensions.
#' @return A `gretaR_array` (discrete, non-negative integers).
#' @export
#' @examples
#' \dontrun{
#' y <- negative_binomial(size = 5, prob = 0.5)
#' }
negative_binomial <- function(size, prob, dim = NULL) {
  dist <- NegativeBinomialDistribution$new(size = size, prob = prob, dim = dim)
  create_variable_node(distribution = dist, dim = dim, is_discrete = TRUE)
}

# =============================================================================
# LKJ Correlation distribution
# =============================================================================

LKJDistribution <- R6::R6Class(
  "LKJDistribution",
  inherit = GretaRDistribution,

  public = list(
    initialize = function(eta, dim_mat) {
      super$initialize(
        name = "lkj",
        parameters = list(eta = eta, dim_mat = dim_mat),
        # Correlation matrix constraint — identity transform for now (Phase 3)
        constraint = list(lower = -1, upper = 1, type = "correlation"),
        dim = c(dim_mat, dim_mat)
      )
    },

    log_prob = function(x) {
      eta <- resolve_param(self$parameters$eta)
      # x is a correlation matrix; compute (eta - 1) * log(det(R))
      # Use slogdet for numerical stability
      log_det <- torch_slogdet(x)
      # torch_slogdet returns list(sign, logabsdet)
      log_det_val <- log_det[[2]]
      (eta - 1) * log_det_val
    },

    sample = function(n = 1L) {
      d <- self$parameters$dim_mat
      eta <- resolve_param(self$parameters$eta)
      # Simple fallback: return identity matrix (proper sampling is complex)
      # Full onion/vine method deferred to Phase 3
      cli_alert_warning(
        "LKJ sampling not yet fully implemented; returning identity matrix."
      )
      torch_eye(d)$unsqueeze(1)$expand(c(n, d, d))
    }
  )
)

#' LKJ Correlation distribution
#'
#' The LKJ distribution over correlation matrices (Lewandowski, Kurowicka,
#' and Joe, 2009). The density is proportional to \eqn{\det(R)^{\eta - 1}}
#' where \eqn{R} is a correlation matrix. When \eqn{\eta = 1}, the
#' distribution is uniform over valid correlation matrices.
#'
#' @param eta Shape parameter (positive). Values > 1 concentrate mass
#'   around the identity matrix; values < 1 favour extreme correlations.
#' @param dim Dimension of the correlation matrix (integer >= 2).
#' @return A `gretaR_array` representing a correlation matrix.
#' @note Simplex/correlation transforms and efficient sampling are
#'   deferred to Phase 3. The current implementation uses an identity
#'   transform and stub sampling (returns identity matrices).
#' @export
#' @examples
#' \dontrun{
#' R <- lkj_correlation(eta = 2, dim = 3)
#' }
lkj_correlation <- function(eta = 1, dim = 2L) {
  if (dim < 2L) cli_abort("LKJ dimension must be >= 2.")
  dist <- LKJDistribution$new(eta = eta, dim_mat = as.integer(dim))
  create_variable_node(distribution = dist, dim = c(dim, dim))
}
