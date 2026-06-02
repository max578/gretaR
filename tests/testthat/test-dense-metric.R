# Dense mass-matrix metric (N3). The metric abstraction must reproduce the
# diagonal dynamics exactly, implement the dense dynamics correctly (velocity
# Sigma p, kinetic 0.5 p^T Sigma p, momentum ~ N(0, Sigma^-1)), estimate a
# sensible metric from warmup draws with a safe diagonal fallback, and -- the
# payoff -- mix a correlated posterior better than the diagonal metric.

test_that("diagonal metric reproduces the historical formulas exactly", {
  m <- metric_diag(c(2, 0.5, 4))
  p <- c(1, -2, 0.5)
  expect_equal(metric_velocity(m, p), p / c(2, 0.5, 4))
  expect_equal(metric_kinetic(m, p), 0.5 * sum(p^2 / c(2, 0.5, 4)))
})

test_that("dense velocity and kinetic use Sigma = M^-1", {
  sigma <- matrix(c(1, 0.8, 0.8, 1), 2, 2)
  m <- metric_dense(sigma)
  expect_equal(m$type, "dense")
  p <- c(1, -1)
  expect_equal(metric_velocity(m, p), as.numeric(sigma %*% p), tolerance = 1e-5)
  expect_equal(metric_kinetic(m, p),
               0.5 * sum(p * (sigma %*% p)), tolerance = 1e-5)
})

test_that("dense momentum draws have covariance M = Sigma^-1", {
  set.seed(1)
  sigma <- matrix(c(1, 0.7, 0.7, 1.5), 2, 2)
  m <- metric_dense(sigma)
  draws <- vapply(seq_len(40000L), function(i) metric_draw_momentum(m, 2L),
                  numeric(2))
  expect_equal(stats::cov(t(draws)), solve(sigma), tolerance = 0.05)
})

test_that("metric_dense returns NULL on a non-PD matrix (defensive fallback)", {
  not_pd <- matrix(c(1, 2, 2, 1), 2, 2)  # eigenvalues 3, -1
  expect_null(metric_dense(not_pd))
})

test_that("estimate_metric chooses dense for correlated draws, diag when told", {
  set.seed(1)
  x1 <- rnorm(500)
  x2 <- 0.9 * x1 + sqrt(1 - 0.81) * rnorm(500)
  X <- cbind(x1, x2)

  md <- estimate_metric(X, kind = "dense")
  expect_equal(md$type, "dense")
  expect_gt(md$Sigma[1, 2], 0.5)            # recovers the positive covariance

  expect_equal(estimate_metric(X, kind = "diag")$type, "diag")
})

test_that("estimate_metric dense falls back to diagonal above the dimension cap", {
  set.seed(1)
  big <- matrix(rnorm(200 * 80), 200, 80)
  # 80 > the default cap of 75: dense is refused (cli message) and diagonal used.
  m_big <- suppressMessages(
    estimate_metric(big, kind = "dense", dense_max_dim = 75L)
  )
  expect_equal(m_big$type, "diag")
  # Within the cap, dense is built.
  small <- matrix(rnorm(200 * 40), 200, 40)
  expect_equal(estimate_metric(small, kind = "dense")$type, "dense")
})

test_that("dense metric recovers a correlated posterior and out-mixes diagonal", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(7)
  torch::torch_manual_seed(7)
  n <- 200L
  x1 <- rnorm(n)
  x2 <- 0.97 * x1 + sqrt(1 - 0.97^2) * rnorm(n)   # strongly correlated predictors
  y <- 1.0 * x1 + 0.5 * x2 + rnorm(n, 0, 0.5)
  build <- function() {
    reset_gretaR_env()
    b1 <- normal(0, 5)
    b2 <- normal(0, 5)
    s <- half_normal(2)
    eta <- b1 * as_data(x1) + b2 * as_data(x2)
    yd <- as_data(y)
    distribution(yd) <- normal(eta, s)
    model_from_arrays(list(b1, b2, s), likelihood = yd, names = c("b1", "b2", "s"))
  }

  fd <- mcmc(build(), n_samples = 400L, warmup = 600L, chains = 2L,
             sampler = "nuts", metric = "diag", seed = 7L, verbose = FALSE)
  fe <- mcmc(build(), n_samples = 400L, warmup = 600L, chains = 2L,
             sampler = "nuts", metric = "dense", seed = 7L, verbose = FALSE)
  sd_ <- posterior::summarise_draws(fd$draws)
  se <- posterior::summarise_draws(fe$draws)

  # Correctness: dense recovers the truth and converges.
  b1e <- se[se$variable == "b1", ]
  b2e <- se[se$variable == "b2", ]
  expect_lt(abs(b1e$mean - 1.0), 0.2)
  expect_lt(abs(b2e$mean - 0.5), 0.2)
  expect_lt(max(se$rhat, na.rm = TRUE), 1.1)

  # Payoff: dense mixes the correlated posterior better than diagonal.
  expect_gt(min(se$ess_bulk, na.rm = TRUE), min(sd_$ess_bulk, na.rm = TRUE))
})

test_that("mcmc rejects an unknown metric", {
  skip_if_not(torch::torch_is_installed())
  reset_gretaR_env()
  mu <- normal(0, 1)
  y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, 1)
  expect_error(mcmc(model(mu), metric = "bogus"), "metric|arg")
})
