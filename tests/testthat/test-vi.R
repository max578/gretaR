# Tests for variational inference

test_that("variational() fits a simple normal model (meanfield)", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(100, 5, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- variational(m, max_iter = 2000, verbose = FALSE)
  expect_s3_class(fit, "gretaR_fit")
  expect_true(abs(coef(fit)["mu"] - mean(y_obs)) < 1)
  expect_true(length(fit$elbo) > 0)
})

test_that("variational() returns posterior draws", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  y_obs <- rnorm(50, 3, 1)
  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- variational(m, max_iter = 1000, verbose = FALSE)
  expect_s3_class(fit$draws, "draws_array")
})
