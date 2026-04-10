# Tests for MAP estimation and Laplace approximation

test_that("opt() finds the MAP for a simple normal model", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(100, 5, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- opt(m, verbose = FALSE)
  expect_true(fit$convergence)
  expect_true(abs(fit$par["mu"] - mean(y_obs)) < 0.5)
})

test_that("opt() works with two parameters", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(100, 3, 1.5)
  mu <- normal(0, 10)
  sigma <- half_cauchy(5)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, sigma)
  m <- model(mu, sigma)

  fit <- opt(m, verbose = FALSE)
  expect_true(abs(fit$par["mu"] - 3) < 1)
  expect_true(abs(fit$par["sigma"] - 1.5) < 1)
})

test_that("laplace() returns posterior approximation", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(50, 5, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  la <- laplace(m, verbose = FALSE)
  expect_true(abs(la$mean["mu"] - mean(y_obs)) < 0.5)
  expect_true(la$sd["mu"] > 0)
  # Analytical SD = 1/sqrt(50) ≈ 0.14
  expect_true(abs(la$sd["mu"] - 1/sqrt(50)) < 0.1)
})
