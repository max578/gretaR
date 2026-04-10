# inference_nuts.R — No-U-Turn Sampler (Hoffman & Gelman, 2014)

#' NUTS sampler
#' @noRd
nuts_sampler <- function(model, n_samples = 1000L, warmup = 500L,
                         chains = 4L, step_size = NULL,
                         max_treedepth = 10L, target_accept = 0.8,
                         init_values = NULL, verbose = TRUE) {

  n_params <- model$total_dim
  total_iter <- n_samples + warmup

  samples <- array(NA_real_, dim = c(n_samples, chains, n_params))
  acceptance_rates <- matrix(NA_real_, nrow = total_iter, ncol = chains)
  divergences <- matrix(FALSE, nrow = total_iter, ncol = chains)
  treedepths <- matrix(0L, nrow = total_iter, ncol = chains)

  param_names <- make_param_names(model)

  if (verbose) {
    cli_alert_info("Running NUTS with {chains} chain{?s}, {warmup} warmup + {n_samples} samples")
    cli_alert_info("Parameters: {n_params}, max tree depth: {max_treedepth}")
  }

  for (chain in seq_len(chains)) {
    if (verbose) cli_alert_info("Chain {chain}/{chains}")

    # Initialise
    theta_vec <- if (!is.null(init_values) && length(init_values) >= chain) {
      as.numeric(init_values[[chain]])
    } else {
      find_initial_values(model, n_params)
    }

    inv_mass_vec <- rep(1.0, n_params)

    # Find reasonable step size
    eps <- if (!is.null(step_size)) {
      step_size
    } else {
      find_reasonable_epsilon(model, theta_vec, inv_mass_vec)
    }

    if (verbose) cli_alert_info("  Initial step size: {round(eps, 5)}")

    # Dual averaging state
    log_eps_bar <- log(eps)
    H_bar <- 0
    mu <- log(10 * eps)
    gamma_da <- 0.05
    t0 <- 10
    kappa <- 0.75

    warmup_thetas <- list()

    # Windowed warmup: 3 phases
    # Phase 1 (first 15%): step-size adaptation only
    # Phase 2 (15%-90%): collect samples for mass matrix
    # Phase 3 (last 10%): re-adapt step size with new mass matrix
    phase2_start <- max(1L, as.integer(warmup * 0.15))
    phase3_start <- max(phase2_start + 1L, as.integer(warmup * 0.9))

    for (iter in seq_len(total_iter)) {
      # Sample momentum
      mom_vec <- rnorm(n_params) * sqrt(inv_mass_vec)

      # Current energy
      eg <- eval_grad(model, theta_vec)
      current_K <- 0.5 * sum(mom_vec^2 / inv_mass_vec)
      joint0 <- eg$lp - current_K

      # Build NUTS tree
      tree <- build_nuts_tree_vec(
        model, theta_vec, mom_vec, eg$grad, eps,
        max_treedepth, joint0, inv_mass_vec
      )

      theta_vec <- tree$theta_new
      acceptance_rates[iter, chain] <- tree$accept_stat
      divergences[iter, chain] <- tree$divergent
      treedepths[iter, chain] <- tree$depth

      # Warmup adaptation
      if (iter <= warmup) {
        # Step-size dual averaging
        m_iter <- iter
        w <- 1 / (m_iter + t0)
        H_bar <- (1 - w) * H_bar + w * (target_accept - tree$accept_stat)
        log_eps <- mu - (sqrt(m_iter) / gamma_da) * H_bar
        eps <- exp(log_eps)
        m_w <- m_iter^(-kappa)
        log_eps_bar <- m_w * log_eps + (1 - m_w) * log_eps_bar

        # Phase 2: collect samples for mass matrix
        if (iter >= phase2_start && iter < phase3_start) {
          warmup_thetas[[length(warmup_thetas) + 1]] <- theta_vec
        }

        # Transition to Phase 3: update mass matrix and re-init step-size
        if (iter == phase3_start) {
          if (length(warmup_thetas) > 2) {
            theta_mat <- do.call(rbind, warmup_thetas)
            theta_var <- apply(theta_mat, 2, var)
            theta_var[theta_var < 1e-3] <- 1e-3
            inv_mass_vec <- theta_var
          }
          # Re-find reasonable step size with new mass matrix
          eps <- find_reasonable_epsilon(model, theta_vec, inv_mass_vec)
          # Reset dual averaging for phase 3
          mu <- log(10 * eps)
          log_eps_bar <- log(eps)
          H_bar <- 0
        }

        # End of warmup: finalise step size
        if (iter == warmup) {
          eps <- exp(log_eps_bar)
          if (verbose) {
            cli_alert_info("  Adapted step size: {round(eps, 5)}")
          }
        }
      }

      # Store post-warmup
      if (iter > warmup) {
        theta_t <- torch_tensor(theta_vec, dtype = model$dtype)
        constrained <- unconstrained_to_constrained(model, theta_t)
        samples[iter - warmup, chain, ] <- as.numeric(constrained$cpu())
      }
    }

    if (verbose) {
      n_div <- sum(divergences[(warmup + 1):total_iter, chain])
      mean_td <- mean(treedepths[(warmup + 1):total_iter, chain])
      cli_alert_success(
        "  Chain {chain} done. Post-warmup divergences: {n_div}, mean tree depth: {round(mean_td, 1)}"
      )
    }
  }

  list(
    samples = samples,
    param_names = param_names,
    acceptance_rates = acceptance_rates,
    divergences = divergences,
    treedepths = treedepths,
    warmup = warmup,
    n_samples = n_samples,
    chains = chains,
    sampler = "nuts"
  )
}

# =============================================================================
# NUTS tree building (numeric vectors, iterative doubling)
# =============================================================================

#' Build NUTS tree
#' @noRd
build_nuts_tree_vec <- function(model, theta, momentum, grad, epsilon,
                                max_depth, joint0, inv_mass) {

  theta_minus <- theta
  theta_plus <- theta
  mom_minus <- momentum
  mom_plus <- momentum
  grad_minus <- grad
  grad_plus <- grad

  theta_new <- theta
  depth <- 0L
  n_valid <- 1L
  sum_accept <- 0
  n_accept <- 0L
  divergent <- FALSE

  for (j in seq_len(max_depth)) {
    direction <- if (runif(1) < 0.5) 1 else -1

    if (direction == -1) {
      result <- build_subtree_vec(
        model, theta_minus, mom_minus, grad_minus,
        epsilon * direction, j - 1L, joint0, inv_mass
      )
      theta_minus <- result$theta_minus
      mom_minus <- result$momentum_minus
      grad_minus <- result$grad_minus
    } else {
      result <- build_subtree_vec(
        model, theta_plus, mom_plus, grad_plus,
        epsilon * direction, j - 1L, joint0, inv_mass
      )
      theta_plus <- result$theta_plus
      mom_plus <- result$momentum_plus
      grad_plus <- result$grad_plus
    }

    if (result$divergent) divergent <- TRUE

    if (!result$s_prime) {
      depth <- j
      break
    }

    if (result$n_prime > 0 && runif(1) < result$n_prime / n_valid) {
      theta_new <- result$theta_prime
    }

    n_valid <- n_valid + result$n_prime
    sum_accept <- sum_accept + result$sum_accept
    n_accept <- n_accept + result$n_accept

    # U-turn criterion
    delta_theta <- theta_plus - theta_minus
    u_turn <- (sum(delta_theta * mom_minus) < 0) ||
              (sum(delta_theta * mom_plus) < 0)

    if (u_turn) {
      depth <- j
      break
    }

    depth <- j
  }

  accept_stat <- if (n_accept > 0) sum_accept / n_accept else 0

  list(
    theta_new = theta_new,
    accept_stat = accept_stat,
    divergent = divergent,
    depth = depth
  )
}

#' Build subtree recursively
#' @noRd
build_subtree_vec <- function(model, theta, momentum, grad, epsilon,
                              depth, joint0, inv_mass) {

  if (depth == 0L) {
    result <- tryCatch({
      lf <- leapfrog_vec(model, theta, momentum, grad, epsilon, inv_mass)

      joint_new <- lf$lp - 0.5 * sum(lf$momentum^2 / inv_mass)
      delta <- joint_new - joint0

      div <- is.nan(delta) || delta < -1000
      n_prime <- if (!div && delta > -1000) 1L else 0L
      accept <- min(1, exp(min(0, delta)))
      if (is.nan(accept)) accept <- 0

      list(
        theta_minus = lf$theta, theta_plus = lf$theta,
        momentum_minus = lf$momentum, momentum_plus = lf$momentum,
        grad_minus = lf$grad, grad_plus = lf$grad,
        theta_prime = lf$theta,
        n_prime = n_prime, s_prime = !div,
        sum_accept = accept, n_accept = 1L,
        divergent = div
      )
    }, error = function(e) {
      list(
        theta_minus = theta, theta_plus = theta,
        momentum_minus = momentum, momentum_plus = momentum,
        grad_minus = grad, grad_plus = grad,
        theta_prime = theta,
        n_prime = 0L, s_prime = FALSE,
        sum_accept = 0, n_accept = 1L,
        divergent = TRUE
      )
    })

    return(result)
  }

  # Build left subtree
  left <- build_subtree_vec(model, theta, momentum, grad, epsilon,
                            depth - 1L, joint0, inv_mass)

  if (!left$s_prime) return(left)

  # Build right subtree from appropriate endpoint
  if (epsilon > 0) {
    right <- build_subtree_vec(
      model, left$theta_plus, left$momentum_plus, left$grad_plus,
      epsilon, depth - 1L, joint0, inv_mass
    )
  } else {
    right <- build_subtree_vec(
      model, left$theta_minus, left$momentum_minus, left$grad_minus,
      epsilon, depth - 1L, joint0, inv_mass
    )
  }

  # Combine subtrees
  n_prime <- left$n_prime + right$n_prime
  if (n_prime > 0 && runif(1) < right$n_prime / n_prime) {
    theta_prime <- right$theta_prime
  } else {
    theta_prime <- left$theta_prime
  }

  if (epsilon > 0) {
    theta_plus <- right$theta_plus
    mom_plus <- right$momentum_plus
    grad_plus <- right$grad_plus
    theta_minus <- left$theta_minus
    mom_minus <- left$momentum_minus
    grad_minus <- left$grad_minus
  } else {
    theta_minus <- right$theta_minus
    mom_minus <- right$momentum_minus
    grad_minus <- right$grad_minus
    theta_plus <- left$theta_plus
    mom_plus <- left$momentum_plus
    grad_plus <- left$grad_plus
  }

  # U-turn check
  delta_theta <- theta_plus - theta_minus
  s_prime <- left$s_prime && right$s_prime &&
    (sum(delta_theta * mom_minus) >= 0) &&
    (sum(delta_theta * mom_plus) >= 0)

  list(
    theta_minus = theta_minus, theta_plus = theta_plus,
    momentum_minus = mom_minus, momentum_plus = mom_plus,
    grad_minus = grad_minus, grad_plus = grad_plus,
    theta_prime = theta_prime,
    n_prime = n_prime, s_prime = s_prime,
    sum_accept = left$sum_accept + right$sum_accept,
    n_accept = left$n_accept + right$n_accept,
    divergent = left$divergent || right$divergent
  )
}
