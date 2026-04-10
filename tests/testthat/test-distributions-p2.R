# Tests for Phase 2 distributions: LogNormal, Cauchy, Wishart

test_that("LogNormal log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- LogNormalDistribution$new(meanlog = 0, sdlog = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dlnorm(1, 0, 1, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("LogNormal log_prob with non-default params", {
  skip_if_not_installed("torch")
  dist <- LogNormalDistribution$new(meanlog = 1, sdlog = 0.5)
  x <- torch_tensor(3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dlnorm(3, 1, 0.5, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Cauchy log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- CauchyDistribution$new(location = 0, scale = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dcauchy(1, 0, 1, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Cauchy log_prob with non-default params", {
  skip_if_not_installed("torch")
  dist <- CauchyDistribution$new(location = 2, scale = 3)
  x <- torch_tensor(5, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dcauchy(5, 2, 3, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Wishart log_prob evaluates for identity", {
  skip_if_not_installed("torch")
  V <- diag(3)
  dist <- WishartDistribution$new(df = 5, scale_matrix = V)
  x <- torch_eye(3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  # Should be finite
  expect_true(is.finite(lp))
})

test_that("lognormal() creates variable with distribution", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- lognormal(0, 1)
  node <- get_node(x)
  expect_equal(node$distribution$name, "lognormal")
})

test_that("cauchy() creates variable with distribution", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- cauchy(0, 1)
  node <- get_node(x)
  expect_equal(node$distribution$name, "cauchy")
})
