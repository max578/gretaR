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
batched_grad_fn <- function(model) {
  fB <- compile_log_prob_batched(model)
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
#' @return `list(theta, mom, lp, grad)` -- final `[C,P]` state + `[C]` log-density.
#' @noRd
batched_leapfrog <- function(bgrad, theta, mom, grad, eps, n_steps, inv_mass) {
  C <- theta$shape[1]
  if (length(n_steps) == 1L) n_steps <- rep(as.integer(n_steps), C)
  l_max <- max(n_steps)
  if (l_max < 1L) {
    return(list(theta = theta, mom = mom, lp = bgrad(theta)$lp, grad = grad))
  }
  ns <- torch::torch_tensor(as.numeric(n_steps), dtype = theta$dtype)  # [C]
  lp <- NULL
  for (s in seq_len(l_max)) {
    m <- (ns >= s)$to(dtype = theta$dtype)$unsqueeze(2)   # [C,1] active mask
    mom <- mom + 0.5 * eps * grad * m                     # half kick
    theta <- theta + eps * (mom / inv_mass) * m           # drift
    bg <- bgrad(theta)                                     # grad at new theta
    grad <- bg$grad
    lp <- bg$lp
    mom <- mom + 0.5 * eps * grad * m                      # half kick
  }
  list(theta = theta, mom = mom, lp = lp, grad = grad)
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
