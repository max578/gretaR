# inference_batched.R -- M1: batched (all-chains-at-once) leapfrog integrator.
#
# Builds on compile_log_prob_batched (M0b): the gradient of all chains is one
# [C,P] tensor, so a single leapfrog advances every chain together. Ragged
# per-chain trajectory lengths (the integration-time HMC of HB1 draws a
# different L per chain) are handled by an active-mask: a chain that has used
# its L_c steps is frozen (mask 0) while the others keep going.
#
# Each iteration is a complete kick-drift-kick (velocity Verlet) step, which is
# mathematically identical to one call of the single-chain leapfrog_vec(); L
# such steps therefore equal L sequential leapfrog_vec() calls per chain (the
# half-kicks of adjacent steps combine into the full kicks of the bracketed
# form). This file is build-beside: it does not touch the single-chain samplers.

# Batched gradient and leapfrog primitives -----------------------------------

#' Batched gradient function for a model
#'
#' @param model A `gretaR_model`.
#' @return `function(theta_mat)` taking `[C,P]` -> `list(lp = [C], grad = [C,P])`.
#'   Chains are independent, so the gradient of `sum(lp)` is the per-chain
#'   gradient. NaN gradients are zeroed (mirrors the single-chain path).
#' @noRd
batched_grad_fn <- function(model, example = NULL) {
  fB <- compile_log_prob_batched(model)
  # JIT-trace at the (fixed) batch shape to remove R-level DAG-walking overhead,
  # mirroring compile_model() for the single-chain path. Falls back to the
  # untraced closure if tracing fails (correctness is identical either way).
  if (!is.null(example)) {
    traced <- tryCatch(torch::jit_trace(fB, example), error = function(e) NULL)
    if (!is.null(traced)) {
      v <- tryCatch(max(abs(as.numeric((traced(example) - fB(example))$cpu()))),
                    error = function(e) Inf)
      if (is.finite(v) && v < 1e-4) fB <- traced
    }
  }
  function(theta_mat) {
    t <- theta_mat$clone()$detach()$requires_grad_(TRUE)
    lp <- fB(t)
    g <- torch::autograd_grad(lp$sum(), t)[[1]]$detach()
    nan <- torch::torch_isnan(g)
    # torch_any() is not exposed by the R torch binding; sum the bool tensor.
    if (torch::torch_sum(nan$to(dtype = torch::torch_int64()))$item() > 0) {
      g <- torch::torch_where(nan, torch::torch_zeros_like(g), g)
    }
    list(lp = lp$detach(), grad = g)
  }
}

#' Batched leapfrog integrator
#'
#' Advances `C` chains, each by its own number of steps, in one batched loop.
#'
#' @param bgrad A batched gradient function (from [batched_grad_fn()]).
#' @param theta `[C, P]` positions.
#' @param mom `[C, P]` momenta.
#' @param grad `[C, P]` gradient at `theta` (from `bgrad`).
#' @param eps Scalar or `[C, 1]` step size.
#' @param n_steps Integer scalar or length-`C` vector of per-chain step counts.
#' @param inv_mass `[P]` or `[C, P]` diagonal metric (used as the mass `M`,
#'   matching `leapfrog_vec`: the drift is `eps * mom / inv_mass`).
#' @param eps Scalar or `[C, 1]` step size.
#' @param joint0 Optional `[C]` initial joint `lp0 - K0`. When supplied, the
#'   trajectory-averaged Metropolis acceptance (Stan's `accept_stat`, the HB1
#'   dual-averaging control signal) is accumulated per chain and returned.
#' @return `list(theta, mom, lp, grad[, accept_stat])` -- final `[C,P]` state +
#'   `[C]` log-density (+ `[C]` accept_stat when `joint0` is given).
#' @noRd
batched_leapfrog <- function(bgrad, theta, mom, grad, eps, n_steps, inv_mass,
                             joint0 = NULL) {
  C <- theta$shape[1]
  if (length(n_steps) == 1L) n_steps <- rep(as.integer(n_steps), C)
  l_max <- max(n_steps)
  track <- !is.null(joint0)
  dev <- theta$device
  if (l_max < 1L) {
    res <- list(theta = theta, mom = mom, lp = bgrad(theta)$lp, grad = grad)
    if (track) res$accept_stat <- torch::torch_zeros(C, dtype = theta$dtype, device = dev)
    return(res)
  }
  ns <- torch::torch_tensor(as.numeric(n_steps), dtype = theta$dtype, device = dev)  # [C]
  if (track) {
    sum_acc <- torch::torch_zeros(C, dtype = theta$dtype, device = dev)
    n_acc <- torch::torch_zeros(C, dtype = theta$dtype, device = dev)
  }
  lp <- NULL
  for (s in seq_len(l_max)) {
    active <- (ns >= s)$to(dtype = theta$dtype)           # [C]
    m <- active$unsqueeze(2)                              # [C,1] active mask
    mom <- mom + 0.5 * eps * grad * m                     # half kick
    theta <- theta + eps * (mom / inv_mass) * m           # drift
    bg <- bgrad(theta)                                    # grad at new theta
    grad <- bg$grad
    lp <- bg$lp
    mom <- mom + 0.5 * eps * grad * m                     # half kick
    if (track) {
      kk <- 0.5 * torch::torch_sum(mom * mom / inv_mass, dim = 2L)   # [C]
      d <- torch::torch_clamp(lp - kk - joint0, max = 0)            # min(0, .)
      a <- torch::torch_exp(d)                                       # min(1, exp(.))
      sum_acc <- sum_acc + a * active
      n_acc <- n_acc + active
    }
  }
  res <- list(theta = theta, mom = mom, lp = lp, grad = grad)
  if (track) res$accept_stat <- sum_acc / torch::torch_clamp(n_acc, min = 1)
  res
}

#' Constrain a batched unconstrained position to the model's parameter space
#' @param model A `gretaR_model`.
#' @param theta `[C, P]` unconstrained.
#' @return `[C, P]` constrained (per-variable inverse transform applied).
#' @noRd
.batched_constrain <- function(model, theta) {
  C <- theta$shape[1]
  cols <- lapply(model$var_order, function(vid) {
    info <- model$param_info[[vid]]
    sl <- (info$offset + 1L):(info$offset + info$n_elem)
    raw <- theta[, sl, drop = FALSE]$reshape(c(C, info$n_elem, 1L))
    con <- if (!is.null(info$transform)) info$transform$inverse(raw) else raw
    con$reshape(c(C, info$n_elem))
  })
  torch::torch_cat(cols, dim = 2L)
}

# Batched integration-time HMC sampler ---------------------------------------

#' Batched integration-time HMC sampler (all chains at once)
#'
#' Vectorises the single-chain integration-time HMC (the HB1 fix) over chains:
#' per-chain `[C]` dual averaging on the trajectory-averaged accept_stat,
#' integration-time trajectories `T ~ U(0, 2*pi]` -> `L = round(T/eps)` fed to
#' [batched_leapfrog()], a shared diagonal mass pooled from warmup. Build-beside;
#' does not touch the single-chain samplers.
#'
#' @return A raw list (samples `[n_samples, C, P]`, param_names, acceptance_rates,
#'   divergences, warmup, n_samples, chains, sampler) -- the `hmc_sampler` shape.
#' @noRd
batched_hmc_sampler <- function(model, n_samples = 1000L, warmup = 1000L,
                                chains = 4L, n_leapfrog = 25L,
                                target_accept = 0.8, seed = NULL,
                                device = "cpu", verbose = FALSE) {
  if (!is.null(seed)) {
    set.seed(seed)
    if (requireNamespace("torch", quietly = TRUE)) torch::torch_manual_seed(seed)
  }
  P <- model$total_dim
  dtype <- model$dtype
  bgrad <- batched_grad_fn(
    model,
    example = torch::torch_zeros(chains, P, dtype = dtype, device = device))

  init <- lapply(seq_len(chains), function(i) find_initial_values(model, P))
  theta <- torch::torch_tensor(do.call(rbind, init), dtype = dtype)$to(device = device)
  inv_mass <- torch::torch_ones(1L, P, dtype = dtype, device = device)   # shared mass M [1,P]

  total_iter <- warmup + n_samples
  samples <- array(NA_real_, dim = c(n_samples, chains, P))
  acceptance_rates <- matrix(NA_real_, total_iter, chains)
  divergences <- matrix(FALSE, total_iter, chains)
  param_names <- make_param_names(model)

  # per-chain dual-averaging state
  eps <- rep(0.25, chains)
  H_bar <- rep(0, chains)
  log_eps_bar <- log(eps)
  mu <- log(10 * eps)
  gamma <- 0.05
  t0 <- 10
  kappa <- 0.75
  phase2 <- max(1L, as.integer(warmup * 0.15))
  phase3 <- max(phase2 + 1L, as.integer(warmup * 0.9))
  warm_draws <- list()

  for (iter in seq_len(total_iter)) {
    eg <- bgrad(theta)
    mom <- torch::torch_randn(chains, P, dtype = dtype, device = device) *
      torch::torch_sqrt(inv_mass)
    K0 <- 0.5 * torch::torch_sum(mom * mom / inv_mass, dim = 2L)      # [C]
    joint0 <- eg$lp - K0
    eps_t <- torch::torch_tensor(eps, dtype = dtype)$unsqueeze(2)$to(device = device)  # [C,1]

    # integration time T ~ U(0, 2pi], L = round(T/eps) (HB1 design). One L per
    # iteration shared across chains (avoids advance-to-max-L waste while keeping
    # per-chain eps + per-iteration length randomisation that breaks resonance);
    # the per-chain integration time L*eps_c still varies.
    l_iter <- max(1L, min(as.integer(round(stats::runif(1, 0, 2 * pi) / stats::median(eps))),
                          10L * n_leapfrog))
    lf <- batched_leapfrog(bgrad, theta, mom, eg$grad, eps_t, l_iter, inv_mass,
                           joint0 = joint0)

    Kp <- 0.5 * torch::torch_sum(lf$mom * lf$mom / inv_mass, dim = 2L)
    dH <- as.numeric(((lf$lp - Kp) - joint0)$cpu())                  # [C]
    a_stat <- as.numeric(lf$accept_stat$cpu())
    divergent <- is.nan(dH) | abs(dH) > 1000
    a_prob <- pmin(1, exp(pmin(0, dH)))
    a_prob[divergent | is.nan(a_prob)] <- 0
    a_stat[divergent] <- 0

    accepted <- (stats::runif(chains) < a_prob) & !divergent
    mask <- torch::torch_tensor(accepted, dtype = torch::torch_bool())$unsqueeze(2)$to(device = device)
    theta <- torch::torch_where(mask, lf$theta, theta)
    acceptance_rates[iter, ] <- a_stat
    divergences[iter, ] <- divergent

    if (iter <= warmup) {
      w <- 1 / (iter + t0)
      H_bar <- (1 - w) * H_bar + w * (target_accept - a_stat)
      log_eps <- mu - (sqrt(iter) / gamma) * H_bar
      eps <- exp(log_eps)
      m_w <- iter^(-kappa)
      log_eps_bar <- m_w * log_eps + (1 - m_w) * log_eps_bar
      if (iter >= phase2 && iter < phase3) {
        warm_draws[[length(warm_draws) + 1]] <- as.matrix(as_array(theta$cpu()))
      }
      if (iter == phase3 && length(warm_draws) > 2) {
        pooled <- do.call(rbind, warm_draws)            # [(nw*C), P]
        v <- apply(pooled, 2, stats::var)
        v[v < 1e-3] <- 1e-3
        inv_mass <- torch::torch_tensor(matrix(1 / v, 1L, P), dtype = dtype)$to(device = device)
        mu <- log(10 * eps)
        log_eps_bar <- log(eps)
        H_bar <- rep(0, chains)
      }
      if (iter == warmup) eps <- exp(log_eps_bar)
    } else {
      samples[iter - warmup, , ] <- as.matrix(as_array(.batched_constrain(model, theta)$cpu()))
    }
  }

  list(samples = samples, param_names = param_names,
       acceptance_rates = acceptance_rates, divergences = divergences,
       warmup = warmup, n_samples = n_samples, chains = chains,
       sampler = "hmc_batched")
}

# Batched ChEES-HMC sampler and helpers --------------------------------------

#' Batched Hamiltonian (per chain)
#'
#' @param lp `[C]` log-density.
#' @param mom `[C, P]` momenta.
#' @param inv_mass `[P]` or `[C, P]` metric (mass `M`); kinetic energy is
#'   `0.5 * sum(mom^2 / M)`.
#' @return `[C]` Hamiltonian `H = -lp + K`.
#' @noRd
batched_hamiltonian <- function(lp, mom, inv_mass) {
  k <- 0.5 * torch::torch_sum(mom * mom / inv_mass, dim = 2L)
  -lp + k
}

#' van der Corput / base-2 radical inverse (a 1-D low-discrepancy sequence).
#'
#' Used to jitter the per-iteration trajectory length: a Halton-style draw in
#' `[0, 1)` (mapped to `[0.5, 1.5)`) breaks the resonance a fixed trajectory
#' length suffers on periodic targets, while keeping the mean length on target.
#' @noRd
.radical_inverse_2 <- function(i) {
  f <- 1
  r <- 0
  while (i > 0) {
    f <- f / 2
    r <- r + f * (i %% 2)
    i <- i %/% 2
  }
  r
}

#' Batched ChEES-HMC sampler (all chains at once, adaptive trajectory length)
#'
#' The batchable alternative to NUTS: keeps fixed-shape HMC trajectories (so all
#' chains advance together on [batched_leapfrog()], unlike NUTS's ragged
#' per-chain tree recursion) and recovers NUTS-quality adaptive trajectory length
#' by tuning the integration time `T` with the ChEES criterion (Change in the
#' Estimator of the Expected Square; Hoffman, Radul & Sountsov 2021). The
#' criterion is computed across the chain ensemble, so its gradient in `T` is the
#' chain rule with `d theta / d T = velocity` at the trajectory endpoint -- no
#' backpropagation through the leapfrog. `T` is adapted by Adam on `log T` during
#' warmup against a running ensemble-mean centre (stable at small chain counts);
#' the per-iteration step count is Halton-jittered to break resonance. Everything
#' else -- per-chain step-size dual averaging, the shared diagonal mass pooled
#' from warmup, the Metropolis correction -- matches [batched_hmc_sampler()].
#' Build-beside; does not touch the single-chain samplers or batched HMC.
#'
#' @param model A `gretaR_model`.
#' @param n_samples,warmup,chains Sampling budget.
#' @param target_accept Dual-averaging step-size target (default 0.8).
#' @param seed Optional integer seed.
#' @param device Torch device for the batched path (`"cpu"`/`"mps"`/`"cuda"`).
#' @param max_steps Cap on the number of leapfrog steps per trajectory (bounds
#'   the adapted `T`; analogous to NUTS `max_treedepth`).
#' @param chees_lr Adam learning rate for the `log T` adaptation.
#' @param verbose Logical; report the adapted trajectory length.
#' @return A raw list in the `batched_hmc_sampler()` shape, with
#'   `sampler = "chees_batched"`.
#' @noRd
batched_chees_sampler <- function(model, n_samples = 1000L, warmup = 1000L,
                                  chains = 4L, target_accept = 0.8, seed = NULL,
                                  device = "cpu", max_steps = 512L,
                                  chees_lr = 0.05, verbose = FALSE) {
  if (!is.null(seed)) {
    set.seed(seed)
    if (requireNamespace("torch", quietly = TRUE)) torch::torch_manual_seed(seed)
  }
  P <- model$total_dim
  dtype <- model$dtype
  bgrad <- batched_grad_fn(
    model,
    example = torch::torch_zeros(chains, P, dtype = dtype, device = device))

  init <- lapply(seq_len(chains), function(i) find_initial_values(model, P))
  theta <- torch::torch_tensor(do.call(rbind, init), dtype = dtype)$to(device = device)
  inv_mass <- torch::torch_ones(1L, P, dtype = dtype, device = device)   # shared mass M [1,P]

  total_iter <- warmup + n_samples
  samples <- array(NA_real_, dim = c(n_samples, chains, P))
  acceptance_rates <- matrix(NA_real_, total_iter, chains)
  divergences <- matrix(FALSE, total_iter, chains)
  param_names <- make_param_names(model)

  # per-chain dual-averaging state (step size eps)
  eps <- rep(0.25, chains)
  H_bar <- rep(0, chains)
  log_eps_bar <- log(eps)
  mu <- log(10 * eps)
  gamma <- 0.05
  t0 <- 10
  kappa <- 0.75
  phase2 <- max(1L, as.integer(warmup * 0.15))
  phase3 <- max(phase2 + 1L, as.integer(warmup * 0.9))
  warm_draws <- list()

  # ChEES trajectory-time adaptation state (scalar log_T + Adam + running mean).
  log_T <- log(1.0)
  m_adam <- 0
  v_adam <- 0
  b1 <- 0.9
  b2 <- 0.999
  adam_eps <- 1e-8
  theta_bar <- theta$mean(dim = 1L, keepdim = TRUE)   # [1,P] running ensemble mean
  bar_decay <- 0.9
  halton_i <- 1L
  T_hist <- numeric(warmup)
  T_final <- NA_real_

  for (iter in seq_len(total_iter)) {
    eg <- bgrad(theta)
    mom <- torch::torch_randn(chains, P, dtype = dtype, device = device) *
      torch::torch_sqrt(inv_mass)
    K0 <- 0.5 * torch::torch_sum(mom * mom / inv_mass, dim = 2L)      # [C]
    joint0 <- eg$lp - K0
    eps_t <- torch::torch_tensor(eps, dtype = dtype)$unsqueeze(2)$to(device = device)  # [C,1]
    med_eps <- stats::median(eps)

    # Trajectory length from the current (warmup) or adapted (sampling) T, mapped
    # through the shared step size and Halton-jittered, then bounded by max_steps.
    T_cur <- if (iter <= warmup) exp(log_T) else T_final
    jitter <- 0.5 + .radical_inverse_2(halton_i)
    halton_i <- halton_i + 1L
    l_iter <- max(1L, min(as.integer(round(T_cur * jitter / med_eps)), max_steps))

    lf <- batched_leapfrog(bgrad, theta, mom, eg$grad, eps_t, l_iter, inv_mass,
                           joint0 = joint0)

    # ChEES gradient at the trajectory endpoint (ensemble-centred). The
    # criterion d/dT (1/4) E[(||q(T)-qbar||^2 - ||q(0)-qbar||^2)^2] reduces to
    # E[(||q(T)-qbar||^2 - ||q(0)-qbar||^2) * (q(T)-qbar) . v(T)], v = mom / M.
    if (iter <= warmup) {
      dq <- lf$theta - theta_bar
      dq0 <- theta - theta_bar
      q_T_sq <- torch::torch_sum(dq * dq, dim = 2L)                  # [C]
      q_0_sq <- torch::torch_sum(dq0 * dq0, dim = 2L)                # [C]
      vel <- lf$mom / inv_mass
      dot <- torch::torch_sum(dq * vel, dim = 2L)                    # [C]
      g_T <- torch::torch_mean((q_T_sq - q_0_sq) * dot)$item()
      # Adam ascent on log_T (chain rule: d/d log T = T * dChEES/dT).
      if (is.finite(g_T)) {
        grad_logT <- exp(log_T) * g_T
        m_adam <- b1 * m_adam + (1 - b1) * grad_logT
        v_adam <- b2 * v_adam + (1 - b2) * grad_logT^2
        mhat <- m_adam / (1 - b1^iter)
        vhat <- v_adam / (1 - b2^iter)
        log_T <- log_T + chees_lr * mhat / (sqrt(vhat) + adam_eps)
        log_T <- max(log(med_eps), min(log_T, log(max_steps * med_eps)))
      }
      T_hist[iter] <- exp(log_T)
    }

    Kp <- 0.5 * torch::torch_sum(lf$mom * lf$mom / inv_mass, dim = 2L)
    dH <- as.numeric(((lf$lp - Kp) - joint0)$cpu())                  # [C]
    a_stat <- as.numeric(lf$accept_stat$cpu())
    divergent <- is.nan(dH) | abs(dH) > 1000
    a_prob <- pmin(1, exp(pmin(0, dH)))
    a_prob[divergent | is.nan(a_prob)] <- 0
    a_stat[divergent] <- 0

    accepted <- (stats::runif(chains) < a_prob) & !divergent
    mask <- torch::torch_tensor(accepted, dtype = torch::torch_bool())$unsqueeze(2)$to(device = device)
    theta <- torch::torch_where(mask, lf$theta, theta)
    acceptance_rates[iter, ] <- a_stat
    divergences[iter, ] <- divergent

    # Update the running ensemble-mean centre from the post-move state.
    theta_bar <- bar_decay * theta_bar + (1 - bar_decay) * theta$mean(dim = 1L, keepdim = TRUE)

    if (iter <= warmup) {
      w <- 1 / (iter + t0)
      H_bar <- (1 - w) * H_bar + w * (target_accept - a_stat)
      log_eps <- mu - (sqrt(iter) / gamma) * H_bar
      eps <- exp(log_eps)
      m_w <- iter^(-kappa)
      log_eps_bar <- m_w * log_eps + (1 - m_w) * log_eps_bar
      if (iter >= phase2 && iter < phase3) {
        warm_draws[[length(warm_draws) + 1]] <- as.matrix(as_array(theta$cpu()))
      }
      if (iter == phase3 && length(warm_draws) > 2) {
        pooled <- do.call(rbind, warm_draws)            # [(nw*C), P]
        v <- apply(pooled, 2, stats::var)
        v[v < 1e-3] <- 1e-3
        inv_mass <- torch::torch_tensor(matrix(1 / v, 1L, P), dtype = dtype)$to(device = device)
        mu <- log(10 * eps)
        log_eps_bar <- log(eps)
        H_bar <- rep(0, chains)
      }
      if (iter == warmup) {
        eps <- exp(log_eps_bar)
        # Final trajectory time: mean of log_T's EWMA-stable second-half values.
        half <- seq.int(as.integer(warmup / 2) + 1L, warmup)
        T_final <- mean(T_hist[half])
        if (verbose) {
          n_steps_est <- round(T_final / stats::median(eps))
          cli_alert_info(
            "ChEES adapted trajectory time T = {round(T_final, 3)} (~{n_steps_est} steps)"
          )
        }
      }
    } else {
      samples[iter - warmup, , ] <- as.matrix(as_array(.batched_constrain(model, theta)$cpu()))
    }
  }

  list(samples = samples, param_names = param_names,
       acceptance_rates = acceptance_rates, divergences = divergences,
       warmup = warmup, n_samples = n_samples, chains = chains,
       sampler = "chees_batched")
}
