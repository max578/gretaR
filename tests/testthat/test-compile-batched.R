# M0b: the batched (all-chains-at-once) log-density must equal stacking the
# single-chain compile_log_prob, in both value and gradient, across the
# workhorse model classes. compile_log_prob (single-chain) is the untouched
# correctness reference; compile_log_prob_batched is the new batched path.

.batched_matches_stacked <- function(model, P, C = 5L, tol = 1e-4) {
  f1 <- compile_log_prob(model)
  fB <- compile_log_prob_batched(model)
  th <- torch::torch_randn(C, P) * 0.4

  thb <- th$clone()$detach()$requires_grad_(TRUE)
  lpb <- fB(thb)
  gb <- as.matrix(as_array(torch::autograd_grad(lpb$sum(), thb)[[1]]$detach()))
  lpb <- as.numeric(lpb$detach())

  lps <- numeric(C)
  gs <- matrix(0, C, P)
  for (c in seq_len(C)) {
    t1 <- th[c, ]$clone()$detach()$requires_grad_(TRUE)
    o <- f1(t1)
    gs[c, ] <- as.numeric(torch::autograd_grad(o, t1)[[1]])
    lps[c] <- as.numeric(o)
  }
  list(dlp = max(abs(lpb - lps)), dg = max(abs(gb - gs)))
}

test_that("batched log-density matches stacked single-chain (lp + grad)", {
  skip_if_not(torch::torch_is_installed())

  # Conjugate Normal (normal density, identity transform).
  reset_gretaR_env()
  y <- as_data(rnorm(8, 3, 2))
  mu <- normal(0, 10)
  distribution(y) <- normal(mu, 2)
  d <- .batched_matches_stacked(model(mu), 1L)
  expect_lt(d$dlp, 1e-4)
  expect_lt(d$dg, 1e-4)

  # Gaussian GLM (matmul, log-transform jacobian on sigma, half-Cauchy prior).
  reset_gretaR_env()
  set.seed(1)
  x <- rnorm(40)
  X <- as_data(cbind(1, x))
  b <- normal(0, 5, dim = 2)
  s <- half_cauchy(2)
  eta <- X %*% b
  yv <- as_data(1.5 - 0.8 * x + rnorm(40))
  distribution(yv) <- normal(eta, s)
  d <- .batched_matches_stacked(model(b, s), 3L)
  expect_lt(d$dlp, 1e-4)
  expect_lt(d$dg, 1e-4)

  # Hierarchical intercepts (index_select).
  reset_gretaR_env()
  G <- 3L
  n <- 30L
  grp <- rep(1:G, each = n / G)
  alpha <- normal(0, 2, dim = G)
  a_obs <- alpha[grp]
  yh <- as_data(rnorm(n))
  distribution(yh) <- normal(a_obs, 1)
  d <- .batched_matches_stacked(model(alpha), 3L)
  expect_lt(d$dlp, 1e-4)
  expect_lt(d$dg, 1e-4)
})

test_that("batched density covers the common univariate families", {
  skip_if_not(torch::torch_is_installed())

  mk <- function(prior) {
    reset_gretaR_env()
    p <- prior()
    y <- as_data(rnorm(5))
    distribution(y) <- normal(0, 1)
    model(p)
  }
  priors <- list(
    half_normal = function() half_normal(2),
    half_cauchy = function() half_cauchy(2),
    gamma = function() gamma_dist(2, 1),
    exponential = function() exponential(1),
    lognormal = function() lognormal(0, 1),
    beta = function() beta_dist(2, 2)
  )
  for (nm in names(priors)) {
    d <- .batched_matches_stacked(mk(priors[[nm]]), 1L)
    expect_lt(d$dlp, 1e-4, label = paste0("dlp[", nm, "]"))
    expect_lt(d$dg, 1e-4, label = paste0("dg[", nm, "]"))
  }
})
