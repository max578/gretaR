# inference_utils.R — Shared utilities for HMC and NUTS samplers

# Mass-matrix metric abstraction (diagonal or dense) -------------------------
#
# The Hamiltonian dynamics need three operations on the momentum: draw it from
# its kinetic distribution, form the velocity (the position update direction),
# and evaluate the kinetic energy. Writing these as `metric_*` helpers lets the
# leapfrog, the NUTS tree, and the HMC trajectory be metric-agnostic, so a dense
# mass matrix drops in beside the diagonal with no change to the integrators.
#
# Convention (Stan's): mass matrix `M`, covariance `Sigma = M^-1`. The momentum
# is `p ~ N(0, M)`; the velocity is `v = M^-1 p = Sigma p`; the kinetic energy
# is `K = 0.5 p^T Sigma p`. The diagonal case stores `M` as a vector and reduces
# to the historical `p / M`, `0.5 sum(p^2 / M)`, `rnorm * sqrt(M)`.

#' Diagonal metric from a mass vector `M` (= 1 / posterior variance).
#' @noRd
metric_diag <- function(m_vec) {
  list(type = "diag", M = m_vec)
}

#' Dense metric from a covariance estimate `Sigma` (= M^-1).
#'
#' Regularises `Sigma` to positive-definite and pre-factors the mass `M` for
#' momentum sampling. Returns `NULL` if the Cholesky fails so the caller can
#' fall back to a diagonal metric (never sample with a broken metric).
#' @noRd
metric_dense <- function(sigma) {
  p <- nrow(sigma)
  # Symmetrise (guard against tiny asymmetry from estimation) then ridge.
  sigma <- (sigma + t(sigma)) / 2
  ridge <- 1e-8 * mean(diag(sigma))
  sigma_r <- sigma + diag(ridge, p)
  # M = Sigma^-1; pre-factor R (upper) with M = R^T R for momentum draws.
  r_chol <- tryCatch(chol(chol2inv(chol(sigma_r))), error = function(e) NULL)
  if (is.null(r_chol)) {
    return(NULL)
  }
  list(type = "dense", Sigma = sigma_r, R = r_chol)
}

#' Velocity `v = M^-1 p` (the position-update direction).
#' @noRd
metric_velocity <- function(metric, p) {
  if (metric$type == "diag") {
    p / metric$M
  } else {
    as.numeric(metric$Sigma %*% p)
  }
}

#' Kinetic energy `K = 0.5 p^T M^-1 p`.
#' @noRd
metric_kinetic <- function(metric, p) {
  0.5 * sum(p * metric_velocity(metric, p))
}

#' Draw momentum `p ~ N(0, M)`.
#' @noRd
metric_draw_momentum <- function(metric, n) {
  if (metric$type == "diag") {
    rnorm(n) * sqrt(metric$M)
  } else {
    # p = R^T z with z ~ N(0, I): Cov(p) = R^T R = M.
    as.numeric(t(metric$R) %*% rnorm(n))
  }
}

#' Estimate a metric from warmup draws, honouring the requested kind.
#'
#' `kind` is `"diag"` (default) or `"dense"`. A dense metric is built only when
#' it is estimable: dimension within `dense_max_dim` (caps the O(P^2) Cholesky
#' cost) and at least `P + 5` warmup draws (so the covariance is not
#' rank-deficient), and the regularised covariance factors. Otherwise -- or for
#' `"diag"` -- the diagonal mass is the inverse posterior variance (GS3:
#' `M = 1 / Var`). The dense mass is the inverse of a Stan-style regularised
#' covariance.
#' @noRd
estimate_metric <- function(theta_mat, kind = "diag", dense_max_dim = 75L) {
  n_s <- nrow(theta_mat)
  p <- ncol(theta_mat)

  if (identical(kind, "dense")) {
    if (p <= dense_max_dim && n_s >= p + 5L) {
      # Stan's regularised covariance: shrink the sample cov toward a small
      # diagonal so it stays well-conditioned when warmup draws are few.
      s_cov <- stats::cov(theta_mat)
      sigma_hat <- (n_s / (n_s + 5)) * s_cov +
        1e-3 * (5 / (n_s + 5)) * diag(p)
      dense <- metric_dense(sigma_hat)
      if (!is.null(dense)) {
        return(dense)
      }
    }
    cli_alert_warning(paste(
      "Dense metric not estimable (P = {p}, warmup draws = {n_s});",
      "using a diagonal metric."
    ))
  }

  theta_var <- apply(theta_mat, 2, stats::var)
  theta_var[theta_var < 1e-3] <- 1e-3
  metric_diag(1 / theta_var)
}

#' Find a reasonable initial step size
#'
#' Uses the algorithm from Stan (Carpenter et al. 2017, Algorithm 4):
#' find epsilon such that the acceptance probability of a single leapfrog
#' step is approximately 0.5.
#'
#' @noRd
find_reasonable_epsilon <- function(model, theta_vec, metric) {
  eps <- 1.0
  n_params <- length(theta_vec)

  mom_vec <- metric_draw_momentum(metric, n_params)
  eg <- eval_grad(model, theta_vec)

  K0 <- metric_kinetic(metric, mom_vec)
  joint0 <- eg$lp - K0

  # One leapfrog step
  lf <- tryCatch(
    leapfrog_vec(model, theta_vec, mom_vec, eg$grad, eps, metric),
    error = function(e) NULL
  )

  if (is.null(lf) || is.nan(lf$lp)) {
    return(0.001)
  }

  K1 <- metric_kinetic(metric, lf$momentum)
  joint1 <- lf$lp - K1
  log_ratio <- joint1 - joint0

  if (is.nan(log_ratio)) return(0.001)

  # Determine direction: increase or decrease epsilon
  direction <- if (log_ratio > log(0.5)) 1 else -1

  for (i in 1:100) {
    if (direction == 1) {
      eps <- eps * 2
    } else {
      eps <- eps / 2
    }

    if (eps < 1e-7 || eps > 1e4) break

    lf <- tryCatch(
      leapfrog_vec(model, theta_vec, mom_vec, eg$grad, eps, metric),
      error = function(e) NULL
    )

    if (is.null(lf) || is.nan(lf$lp)) break

    K1 <- metric_kinetic(metric, lf$momentum)
    joint1 <- lf$lp - K1
    log_ratio <- joint1 - joint0

    if (is.nan(log_ratio)) break

    if (direction == 1 && log_ratio < log(0.5)) break
    if (direction == -1 && log_ratio > log(0.5)) break
  }

  max(1e-7, min(eps, 1e4))
}

#' Find reasonable initial values via gradient ascent toward the MAP
#'
#' Runs Adam-like gradient ascent on the log-joint density to find a
#' starting point near the posterior mode.
#'
#' @noRd
find_initial_values <- function(model, n_params, n_steps = 200,
                                learning_rate = 0.1) {
  # Start from a random point
  theta_vec <- rnorm(n_params, 0, 0.5)

  # Simple gradient ascent with momentum (Adam-lite)
  m_vec <- rep(0, n_params)  # first moment
  v_vec <- rep(0, n_params)  # second moment
  beta1 <- 0.9
  beta2 <- 0.999
  eps_adam <- 1e-8

  best_theta <- theta_vec
  best_lp <- -Inf

  for (step in seq_len(n_steps)) {
    eg <- tryCatch(eval_grad(model, theta_vec), error = function(e) NULL)
    if (is.null(eg) || is.nan(eg$lp) || any(is.nan(eg$grad))) {
      # Reset to a random point if we hit NaN
      theta_vec <- rnorm(n_params, 0, 0.5)
      next
    }

    if (eg$lp > best_lp) {
      best_lp <- eg$lp
      best_theta <- theta_vec
    }

    # Adam update (gradient ascent, so we ADD)
    m_vec <- beta1 * m_vec + (1 - beta1) * eg$grad
    v_vec <- beta2 * v_vec + (1 - beta2) * eg$grad^2
    m_hat <- m_vec / (1 - beta1^step)
    v_hat <- v_vec / (1 - beta2^step)

    theta_vec <- theta_vec + learning_rate * m_hat / (sqrt(v_hat) + eps_adam)

    # Clamp to prevent explosion
    theta_vec <- pmax(pmin(theta_vec, 20), -20)
  }

  # Add small jitter so chains don't all start at the same point
  best_theta + rnorm(n_params, 0, 0.1)
}

#' Compute log_prob and gradient at a numeric vector position
#'
#' Uses the active gradient function stored in `.gretaR_env$active_grad_fn`
#' if available (set by the sampler when a compiled function is provided),
#' otherwise falls back to the standard `grad_log_prob()`.
#' @noRd
eval_grad <- function(model, theta_vec) {
  # Fast path: use compiled gradient function if active
  if (!is.null(.gretaR_env$active_grad_fn)) {
    return(.gretaR_env$active_grad_fn(theta_vec))
  }
  # Standard path
  theta_t <- torch_tensor(theta_vec, dtype = model$dtype)
  glp <- grad_log_prob(model, theta_t)
  grad_vec <- as.numeric(glp$grad)
  if (any(is.nan(grad_vec))) {
    cli_alert_warning("NaN gradient detected; replacing with 0 (model may be misspecified)")
    grad_vec[is.nan(grad_vec)] <- 0
  }
  list(lp = glp$lp, grad = grad_vec)
}

#' Single leapfrog step (numeric vectors)
#' @noRd
leapfrog_vec <- function(model, theta, momentum, grad, epsilon, metric) {
  # Half step for momentum
  momentum <- momentum + 0.5 * epsilon * grad

  # Full step for position along the velocity v = M^-1 p
  theta <- theta + epsilon * metric_velocity(metric, momentum)

  # Evaluate gradient at new position
  eg <- eval_grad(model, theta)

  # Half step for momentum
  momentum <- momentum + 0.5 * epsilon * eg$grad

  list(theta = theta, momentum = momentum, lp = eg$lp, grad = eg$grad)
}

#' Make parameter names from model
#' @noRd
make_param_names <- function(model) {
  param_names <- character(model$total_dim)
  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    # Caller-supplied per-element names (via `model_from_arrays(names = ...)`)
    # take precedence over the positional `name[j]` convention, so a host
    # package gets exactly its own labels in the draws without a relabel pass.
    elem_names <- info$element_names
    if (!is.null(elem_names) && length(elem_names) == info$n_elem) {
      idx <- (info$offset + 1L):(info$offset + info$n_elem)
      param_names[idx] <- elem_names
    } else if (info$n_elem == 1L) {
      param_names[info$offset + 1L] <- info$name
    } else {
      for (j in seq_len(info$n_elem)) {
        param_names[info$offset + j] <- paste0(info$name, "[", j, "]")
      }
    }
  }
  param_names
}

#' Convert unconstrained → constrained parameters
#' @noRd
unconstrained_to_constrained <- function(model, theta_free) {
  theta_vec <- as.numeric(theta_free$cpu())
  result <- numeric(length(theta_vec))

  for (vid in model$var_order) {
    info <- model$param_info[[vid]]
    start <- info$offset + 1L
    end <- info$offset + info$n_elem
    raw_vec <- theta_vec[start:end]
    raw <- torch_tensor(raw_vec, dtype = model$dtype)

    if (!is.null(info$transform)) {
      if (prod(info$dim) > 1L) raw <- raw$reshape(info$dim)
      constrained <- info$transform$inverse(raw)
      result[start:end] <- as.numeric(constrained$detach()$cpu())
    } else {
      result[start:end] <- raw_vec
    }
  }
  torch_tensor(result, dtype = model$dtype)
}
