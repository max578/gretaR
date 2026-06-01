# M1: the batched leapfrog must equal looping the single-chain leapfrog per
# chain (bit-identical to float tolerance), conserve energy, and give each chain
# exactly its own number of steps (ragged). compile_log_prob is the reference.

.glm_model <- function() {
  reset_gretaR_env()
  set.seed(1); x <- rnorm(40); X <- as_data(cbind(1, x))
  b <- normal(0, 5, dim = 2); s <- half_cauchy(2)
  eta <- X %*% b; y <- as_data(1.5 - 0.8 * x + rnorm(40))
  distribution(y) <- normal(eta, s)
  model(b, s)
}

# single-chain reference leapfrog over compile_log_prob ([P] tensors)
.sc_leapfrog <- function(f1, theta, mom, grad, eps, L, inv_mass) {
  g <- function(th) { t <- th$clone()$detach()$requires_grad_(TRUE); o <- f1(t); torch::autograd_grad(o, t)[[1]]$detach() }
  for (s in seq_len(L)) {
    mom <- mom + 0.5 * eps * grad
    theta <- theta + eps * mom / inv_mass
    grad <- g(theta)
    mom <- mom + 0.5 * eps * grad
  }
  list(theta = as.numeric(theta$cpu()), mom = as.numeric(mom$cpu()))
}

test_that("batched leapfrog equals looping single-chain leapfrog (theta + mom)", {
  skip_if_not(torch::torch_is_installed())
  m <- .glm_model(); P <- 3L; C <- 5L
  f1 <- compile_log_prob(m); bgrad <- batched_grad_fn(m)

  torch::torch_manual_seed(2)
  theta0 <- torch::torch_randn(C, P) * 0.4; mom0 <- torch::torch_randn(C, P)
  inv_mass <- torch::torch_tensor(c(1.3, 0.7, 2.1)); eps <- 0.02; L <- 25L

  out <- batched_leapfrog(bgrad, theta0, mom0, bgrad(theta0)$grad, eps, L, inv_mass)
  th_b <- as.matrix(as_array(out$theta$cpu())); mo_b <- as.matrix(as_array(out$mom$cpu()))
  for (c in seq_len(C)) {
    g0 <- bgrad(theta0[c, ]$unsqueeze(1))$grad$squeeze(1)
    r <- .sc_leapfrog(f1, theta0[c, ], mom0[c, ], g0, eps, L, inv_mass)
    expect_lt(max(abs(th_b[c, ] - r$theta)), 1e-4)
    expect_lt(max(abs(mo_b[c, ] - r$mom)), 1e-4)
  }
})

test_that("batched leapfrog conserves energy and honours ragged step counts", {
  skip_if_not(torch::torch_is_installed())
  m <- .glm_model(); P <- 3L; C <- 6L
  f1 <- compile_log_prob(m); bgrad <- batched_grad_fn(m)

  torch::torch_manual_seed(3)
  theta0 <- torch::torch_randn(C, P) * 0.4; mom0 <- torch::torch_randn(C, P)
  inv_mass <- torch::torch_tensor(c(1.3, 0.7, 2.1))

  # energy conservation (small eps)
  g0 <- bgrad(theta0)
  H0 <- batched_hamiltonian(g0$lp, mom0, inv_mass)
  oe <- batched_leapfrog(bgrad, theta0, mom0, g0$grad, 0.005, 60L, inv_mass)
  H1 <- batched_hamiltonian(oe$lp, oe$mom, inv_mass)
  expect_lt(max(abs(as.numeric((H1 - H0)$cpu()))), 0.1)

  # ragged: each chain matches its own L
  n_steps <- c(3L, 7L, 12L, 20L, 25L, 1L)
  outr <- batched_leapfrog(bgrad, theta0, mom0, g0$grad, 0.02, n_steps, inv_mass)
  thr <- as.matrix(as_array(outr$theta$cpu()))
  for (c in seq_len(C)) {
    g0c <- bgrad(theta0[c, ]$unsqueeze(1))$grad$squeeze(1)
    r <- .sc_leapfrog(f1, theta0[c, ], mom0[c, ], g0c, 0.02, n_steps[c], inv_mass)
    expect_lt(max(abs(thr[c, ] - r$theta)), 1e-4)
  }
})
