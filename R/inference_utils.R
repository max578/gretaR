# inference_utils.R — Shared utilities for HMC and NUTS samplers

#' Find a reasonable initial step size
#'
#' Uses the algorithm from Stan (Carpenter et al. 2017, Algorithm 4):
#' find epsilon such that the acceptance probability of a single leapfrog
#' step is approximately 0.5.
#'
#' @noRd
find_reasonable_epsilon <- function(model, theta_vec, inv_mass_vec) {
  eps <- 1.0
  n_params <- length(theta_vec)

  mom_vec <- rnorm(n_params) * sqrt(inv_mass_vec)
  eg <- eval_grad(model, theta_vec)

  K0 <- 0.5 * sum(mom_vec^2 / inv_mass_vec)
  joint0 <- eg$lp - K0

  # One leapfrog step
  lf <- tryCatch(
    leapfrog_vec(model, theta_vec, mom_vec, eg$grad, eps, inv_mass_vec),
    error = function(e) NULL
  )

  if (is.null(lf) || is.nan(lf$lp)) {
    return(0.001)
  }

  K1 <- 0.5 * sum(lf$momentum^2 / inv_mass_vec)
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
      leapfrog_vec(model, theta_vec, mom_vec, eg$grad, eps, inv_mass_vec),
      error = function(e) NULL
    )

    if (is.null(lf) || is.nan(lf$lp)) break

    K1 <- 0.5 * sum(lf$momentum^2 / inv_mass_vec)
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
#' @noRd
eval_grad <- function(model, theta_vec) {
  theta_t <- torch_tensor(theta_vec, dtype = model$dtype)
  glp <- grad_log_prob(model, theta_t)
  grad_vec <- as.numeric(glp$grad$detach()$cpu())
  if (any(is.nan(grad_vec))) grad_vec[is.nan(grad_vec)] <- 0
  list(lp = glp$lp, grad = grad_vec)
}

#' Single leapfrog step (numeric vectors)
#' @noRd
leapfrog_vec <- function(model, theta, momentum, grad, epsilon, inv_mass) {
  # Half step for momentum
  momentum <- momentum + 0.5 * epsilon * grad

  # Full step for position
  theta <- theta + epsilon * momentum / inv_mass

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
    if (info$n_elem == 1L) {
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
