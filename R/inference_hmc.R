# inference_hmc.R — Hamiltonian Monte Carlo with dual averaging

#' Static HMC sampler
#' @noRd
hmc_sampler <- function(model, n_samples = 1000L, warmup = 500L,
                        chains = 4L, step_size = NULL,
                        n_leapfrog = 25L, target_accept = 0.65,
                        compiled_fn = NULL, metric = "diag",
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

    # Metric (mass matrix): identity until warmup estimates it.
    mtr <- metric_diag(rep(1.0, n_params))

    # Find reasonable step size if not provided
    eps <- if (!is.null(step_size)) {
      step_size
    } else {
      find_reasonable_epsilon(model, theta_vec, mtr)
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
      mom_vec <- metric_draw_momentum(mtr, n_params)

      eg <- eval_grad(model, theta_vec)
      current_lp <- eg$lp
      current_K <- metric_kinetic(mtr, mom_vec)

      # Leapfrog trajectory
      joint0 <- current_lp - current_K
      theta_prop <- theta_vec
      mom_prop <- mom_vec
      grad_prop <- eg$grad
      divergent <- FALSE

      # Average Metropolis acceptance over the trajectory's states -- the
      # dual-averaging control signal (Stan's accept_stat, = NUTS's
      # sum_accept/n_accept), NOT the single end-of-trajectory acceptance.
      # Leapfrog energy error oscillates and can return near zero at the
      # endpoint, so the endpoint acceptance saturates at 1 over a wide step-
      # size band and drives eps up without bound (HB1: chains adapted to
      # eps 1.8-4.4, ess ~45). The trajectory average is smooth in eps and
      # stabilises adaptation, matching the robust NUTS path.
      sum_accept <- 0
      n_accept <- 0L

      # Integration-time HMC. Fixed-length HMC resonates with the target's
      # periodic Hamiltonian flow: a trajectory that is a near-multiple of the
      # oscillation period returns close to its start, giving near-zero net
      # moves and catastrophic autocorrelation (HB1: ess ~10-50, fragile to
      # the exact L and step size). After mass adaptation the metric is
      # M = 1/Var, so the target is approximately isotropic with oscillation
      # period 2*pi. Each iteration we integrate for a *random time*
      # T ~ U(0, 2*pi] and take L = round(T / eps) steps: this controls the
      # trajectory length in time (independent of the adapted eps) and
      # randomises it, averaging over the resonance (Neal 2011, sec 4.2-4.4).
      # Capped for safety. NUTS achieves the same via dynamic termination.
      traj_steps <- max(1L, as.integer(round(runif(1, 0, 2 * pi) / eps)))
      n_lf_iter <- min(traj_steps, 10L * n_leapfrog)

      for (step in seq_len(n_lf_iter)) {
        lf <- tryCatch(
          leapfrog_vec(model, theta_prop, mom_prop, grad_prop, eps, mtr),
          error = function(e) NULL
        )

        if (is.null(lf) || is.nan(lf$lp) || any(is.nan(lf$grad))) {
          divergent <- TRUE
          break
        }

        theta_prop <- lf$theta
        mom_prop <- lf$momentum
        grad_prop <- lf$grad

        joint_step <- lf$lp - metric_kinetic(mtr, mom_prop)
        a_step <- min(1, exp(min(0, joint_step - joint0)))
        if (is.nan(a_step)) a_step <- 0
        sum_accept <- sum_accept + a_step
        n_accept <- n_accept + 1L
      }

      # Divergent trajectories contribute 0 acceptance so adaptation lowers eps.
      accept_stat <- if (divergent || n_accept == 0L) 0 else sum_accept / n_accept

      if (!divergent) {
        proposed_lp <- lf$lp
        proposed_K <- metric_kinetic(mtr, mom_prop)
        delta_H <- (proposed_lp - proposed_K) - joint0

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

      # Report the averaged accept_stat (the adaptation/diagnostic statistic,
      # as NUTS does); the actual move above uses the exact endpoint MH ratio.
      acceptance_rates[iter, chain] <- accept_stat

      # Warmup adaptation (windowed)
      if (iter <= warmup) {
        m_iter <- iter
        w <- 1 / (m_iter + t0)
        H_bar <- (1 - w) * H_bar + w * (target_accept - accept_stat)
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
            mtr <- estimate_metric(theta_mat, kind = metric)
          }
          eps <- find_reasonable_epsilon(model, theta_vec, mtr)
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
