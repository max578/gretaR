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
#' @param model A `gretaR_model`; @param theta `[C, P]` unconstrained.
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
  H_bar <- rep(0, chains); log_eps_bar <- log(eps); mu <- log(10 * eps)
  gamma <- 0.05; t0 <- 10; kappa <- 0.75
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
    a_prob <- pmin(1, exp(pmin(0, dH))); a_prob[divergent | is.nan(a_prob)] <- 0
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
        v <- apply(pooled, 2, stats::var); v[v < 1e-3] <- 1e-3
        inv_mass <- torch::torch_tensor(matrix(1 / v, 1L, P), dtype = dtype)$to(device = device)
        mu <- log(10 * eps); log_eps_bar <- log(eps); H_bar <- rep(0, chains)
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
