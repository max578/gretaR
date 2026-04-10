# Tests for variational inference

test_that("vi() fits a simple normal model (meanfield)", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(100, 5, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- vi(m, max_iter = 2000, verbose = FALSE)
  expect_s3_class(fit, "gretaR_vi")
  expect_true(abs(fit$mean["mu"] - mean(y_obs)) < 1)
  expect_true(fit$sd["mu"] > 0)
  expect_true(length(fit$elbo) > 0)
})

test_that("vi() returns posterior draws", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(50, 3, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- vi(m, max_iter = 1000, verbose = FALSE)
  expect_s3_class(fit$draws, "draws_array")
})
