# Tests for custom distributions and mixture models

test_that("custom_distribution creates a variable with custom log_prob", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- custom_distribution(
    log_prob_fn = function(x) -torch_sum(torch_abs(x)),
    name = "laplace"
  )
  node <- get_node(x)
  expect_equal(node$node_type, "variable")
  expect_equal(node$distribution$name, "laplace")
})

test_that("custom_distribution log_prob evaluates correctly", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # Standard normal kernel
  dist <- CustomDistribution$new(
    log_prob_fn = function(x) torch_sum(-0.5 * x * x - 0.9189385),
    name = "custom_normal"
  )
  x <- torch_tensor(0, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expect_equal(lp, dnorm(0, log = TRUE), tolerance = 1e-3)
})

test_that("custom_distribution with constraints applies transform", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- custom_distribution(
    log_prob_fn = function(x) -torch_sum(x),
    constraint = list(lower = 0, upper = Inf),
    name = "exp_like"
  )
  node <- get_node(x)
  expect_true(inherits(node$transform, "LogTransform"))
})

test_that("custom_distribution rejects non-functions", {
  skip_if_not_installed("torch")
  expect_error(custom_distribution("not a function"), "must be a function")
})

test_that("MixtureDistribution computes log_prob via log-sum-exp", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  d1 <- NormalDistribution$new(mean = -2, sd = 1)
  d2 <- NormalDistribution$new(mean = 2, sd = 1)
  w <- torch_tensor(c(0.5, 0.5), dtype = torch_float32())

  mix <- MixtureDistribution$new(
    distributions = list(d1, d2),
    weights = w
  )

  x <- torch_tensor(0, dtype = torch_float32())
  lp <- mix$log_prob(x)$item()

  # Manual: log(0.5*dnorm(0,-2,1) + 0.5*dnorm(0,2,1))
  expected <- log(0.5 * dnorm(0, -2, 1) + 0.5 * dnorm(0, 2, 1))
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("mixture() rejects insufficient components", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  expect_error(mixture(list(normal(0, 1)), c(1)), "at least 2")
})
