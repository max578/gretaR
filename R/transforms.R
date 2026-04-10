# transforms.R — Bijectors for constrained ↔ unconstrained parameterisation
#
# Each transform has:
#   forward(x)           : constrained → unconstrained
#   inverse(y)           : unconstrained → constrained
#   log_det_jacobian(y)  : log |d inverse(y) / dy|  (for HMC correction)

#' @title Parameter Transforms (Bijectors)
#' @name transforms
#' @description
#' Transforms map between constrained parameter spaces and unconstrained
#' real-valued spaces for HMC/NUTS sampling.
#' @noRd
NULL

# --- Identity transform (unconstrained parameters) ---

IdentityTransform <- R6::R6Class(

  "IdentityTransform",
  public = list(
    forward = function(x) x,
    inverse = function(y) y,
    log_det_jacobian = function(y) {
      torch_zeros(1, dtype = y$dtype, device = y$device)
    }
  )
)

# --- Log transform (positive parameters: rate, sd, scale) ---

LogTransform <- R6::R6Class(
  "LogTransform",
  public = list(
    forward = function(x) torch_log(x),
    inverse = function(y) torch_exp(y),
    log_det_jacobian = function(y) {
      # d/dy exp(y) = exp(y), so log|det J| = sum(y)
      torch_sum(y)
    }
  )
)

# --- Logit transform (parameters in (0, 1)) ---

LogitTransform <- R6::R6Class(
  "LogitTransform",
  public = list(
    forward = function(x) torch_log(x / (1 - x)),
    inverse = function(y) torch_sigmoid(y),
    log_det_jacobian = function(y) {
      # d/dy sigmoid(y) = sigmoid(y) * (1 - sigmoid(y))
      # log|det J| = sum(log(sigmoid(y)) + log(1 - sigmoid(y)))
      s <- torch_sigmoid(y)
      torch_sum(torch_log(s) + torch_log(1 - s))
    }
  )
)

# --- Scaled logit transform (parameters in (lower, upper)) ---

ScaledLogitTransform <- R6::R6Class(
  "ScaledLogitTransform",
  public = list(
    lower = NULL,
    upper = NULL,

    initialize = function(lower, upper) {
      self$lower <- lower
      self$upper <- upper
    },

    forward = function(x) {
      # Map (lower, upper) → R
      z <- (x - self$lower) / (self$upper - self$lower)
      torch_log(z / (1 - z))
    },

    inverse = function(y) {
      # Map R → (lower, upper)
      s <- torch_sigmoid(y)
      self$lower + (self$upper - self$lower) * s
    },

    log_det_jacobian = function(y) {
      s <- torch_sigmoid(y)
      range_val <- self$upper - self$lower
      # log|dx/dy| = log(range) + log(sigmoid(y)) + log(1 - sigmoid(y))
      torch_sum(
        log(range_val) + torch_log(s) + torch_log(1 - s)
      )
    }
  )
)

# --- Softplus transform (positive, smoother than exp) ---

SoftplusTransform <- R6::R6Class(
  "SoftplusTransform",
  public = list(
    forward = function(x) {
      # softplus_inv(x) = log(exp(x) - 1)
      torch_log(torch_exp(x) - 1)
    },
    inverse = function(y) {
      torch_nn_functional_softplus(y)
    },
    log_det_jacobian = function(y) {
      # d/dy softplus(y) = sigmoid(y)
      torch_sum(torch_log(torch_sigmoid(y)))
    }
  )
)

# --- Lower-bounded transform ---

LowerBoundTransform <- R6::R6Class(
  "LowerBoundTransform",
  public = list(
    lower = NULL,

    initialize = function(lower = 0) {
      self$lower <- lower
    },

    forward = function(x) torch_log(x - self$lower),

    inverse = function(y) torch_exp(y) + self$lower,

    log_det_jacobian = function(y) {
      # d/dy (exp(y) + lower) = exp(y), so log|det J| = sum(y)
      torch_sum(y)
    }
  )
)

# --- Helper: select transform from constraints ---

#' Select an appropriate transform given constraints
#' @param lower Lower bound (NULL or -Inf for unconstrained below)
#' @param upper Upper bound (NULL or Inf for unconstrained above)
#' @return A transform object
#' @noRd
select_transform <- function(lower = NULL, upper = NULL) {
  has_lower <- !is.null(lower) && is.finite(lower)
  has_upper <- !is.null(upper) && is.finite(upper)

  if (!has_lower && !has_upper) {
    IdentityTransform$new()
  } else if (has_lower && !has_upper) {
    if (lower == 0) {
      LogTransform$new()
    } else {
      LowerBoundTransform$new(lower)
    }
  } else if (!has_lower && has_upper) {
    # Reflect: upper-bounded → lower-bounded via negation
    # Not common, but handle it
    cli_abort("Upper-bounded-only transforms not yet implemented.")
  } else {
    # Both bounded
    if (lower == 0 && upper == 1) {
      LogitTransform$new()
    } else {
      ScaledLogitTransform$new(lower, upper)
    }
  }
}
