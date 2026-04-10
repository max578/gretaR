# inference_hmc.R — Hamiltonian Monte Carlo with dual averaging

#' Static HMC sampler
#' @noRd
hmc_sampler <- function(model, n_samples = 1000L, warmup = 500L,
                        chains = 4L, step_size = NULL,
                        n_leapfrog = 25L, target_accept = 0.65,
                        compiled_fn = NULL,
                        init_values = NULL, verbose = TRUE) {

  n_params <- model$total_dim
  total_iter <- n_samples + warmup

  samples <- array(NA_real_, dim = c(n_samples, chains, n_params))
  acceptance_rates <- matrix(NA_real_, nrow = total_iter, ncol = chains)
  divergences <- matrix(FALSE, nrow = total_iter, ncol = chains)

  param_names <- make_param_names(model)

  if (verbose) {
    cli_alert_info("Running HMC with {chains} chain{?s}, {warmup} warmup + {n_samples} samples")
    cli_alert_info("Parameters: {n_params}")
  }

  # Set up compiled gradient function if available
  if (!is.null(compiled_fn)) {
    .gretaR_env$active_grad_fn <- function(theta_vec) {
      fast_grad(compiled_fn, theta_vec, model$dtype)
    }
  } else {
    .gretaR_env$active_grad_fn <- NULL
  }
  on.exit(.gretaR_env$active_grad_fn <- NULL, add = TRUE)

  for (chain in seq_len(chains)) {
    if (verbose) cli_alert_info("Chain {chain}/{chains}")

    # Initialise
    theta_vec <- if (!is.null(init_values) && length(init_values) >= chain) {
      as.numeric(init_values[[chain]])
    } else {
      find_initial_values(model, n_params)
    }

    inv_mass_vec <- rep(1.0, n_params)

    # Find reasonable step size if not provided
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
    gamma <- 0.05
    t0 <- 10
    kappa <- 0.75

    warmup_thetas <- list()
    phase2_start <- max(1L, as.integer(warmup * 0.15))
    phase3_start <- max(phase2_start + 1L, as.integer(warmup * 0.9))

    for (iter in seq_len(total_iter)) {
      mom_vec <- rnorm(n_params) * sqrt(inv_mass_vec)

      eg <- eval_grad(model, theta_vec)
      current_lp <- eg$lp
      current_K <- 0.5 * sum(mom_vec^2 / inv_mass_vec)

      # Leapfrog trajectory
      theta_prop <- theta_vec
      mom_prop <- mom_vec
      grad_prop <- eg$grad
      divergent <- FALSE

      for (step in seq_len(n_leapfrog)) {
        lf <- tryCatch(
          leapfrog_vec(model, theta_prop, mom_prop, grad_prop, eps, inv_mass_vec),
          error = function(e) NULL
        )

        if (is.null(lf) || is.nan(lf$lp) || any(is.nan(lf$grad))) {
          divergent <- TRUE
          break
        }

        theta_prop <- lf$theta
        mom_prop <- lf$momentum
        grad_prop <- lf$grad
      }

      if (!divergent) {
        proposed_lp <- lf$lp
        proposed_K <- 0.5 * sum(mom_prop^2 / inv_mass_vec)
        delta_H <- (proposed_lp - proposed_K) - (current_lp - current_K)

        if (is.nan(delta_H) || abs(delta_H) > 1000) {
          divergent <- TRUE
          delta_H <- -Inf
        }
      } else {
        delta_H <- -Inf
      }

      divergences[iter, chain] <- divergent

      accept_prob <- min(1, exp(delta_H))
      if (is.nan(accept_prob)) accept_prob <- 0

      if (runif(1) < accept_prob && !divergent) {
        theta_vec <- theta_prop
      }

      acceptance_rates[iter, chain] <- accept_prob

      # Warmup adaptation (windowed)
      if (iter <= warmup) {
        m_iter <- iter
        w <- 1 / (m_iter + t0)
        H_bar <- (1 - w) * H_bar + w * (target_accept - accept_prob)
        log_eps <- mu - (sqrt(m_iter) / gamma) * H_bar
        eps <- exp(log_eps)
        m_w <- m_iter^(-kappa)
        log_eps_bar <- m_w * log_eps + (1 - m_w) * log_eps_bar

        if (iter >= phase2_start && iter < phase3_start) {
          warmup_thetas[[length(warmup_thetas) + 1]] <- theta_vec
        }

        if (iter == phase3_start) {
          if (length(warmup_thetas) > 2) {
            theta_mat <- do.call(rbind, warmup_thetas)
            theta_var <- apply(theta_mat, 2, var)
            theta_var[theta_var < 1e-3] <- 1e-3
            inv_mass_vec <- theta_var
          }
          eps <- find_reasonable_epsilon(model, theta_vec, inv_mass_vec)
          mu <- log(10 * eps)
          log_eps_bar <- log(eps)
          H_bar <- 0
        }

        if (iter == warmup) {
          eps <- exp(log_eps_bar)
          if (verbose) {
            cli_alert_info("  Adapted step size: {round(eps, 5)}")
          }
        }
      }

      # Store post-warmup samples
      if (iter > warmup) {
        theta_t <- torch_tensor(theta_vec, dtype = model$dtype)
        constrained <- unconstrained_to_constrained(model, theta_t)
        samples[iter - warmup, chain, ] <- as.numeric(constrained$cpu())
      }
    }
  }

  list(
    samples = samples,
    param_names = param_names,
    acceptance_rates = acceptance_rates,
    divergences = divergences,
    warmup = warmup,
    n_samples = n_samples,
    chains = chains,
    sampler = "hmc"
  )
}
