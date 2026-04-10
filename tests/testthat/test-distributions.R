# Tests for P0 distributions

test_that("Normal log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- NormalDistribution$new(mean = 0, sd = 1)
  x <- torch_tensor(0, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dnorm(0, 0, 1, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-4)

  x2 <- torch_tensor(1.5, dtype = torch_float32())
  lp2 <- dist$log_prob(x2)$item()
  expect_equal(lp2, dnorm(1.5, 0, 1, log = TRUE), tolerance = 1e-4)
})

test_that("Normal with non-standard parameters", {
  skip_if_not_installed("torch")
  dist <- NormalDistribution$new(mean = 3, sd = 2)
  x <- torch_tensor(3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expect_equal(lp, dnorm(3, 3, 2, log = TRUE), tolerance = 1e-4)
})

test_that("HalfNormal log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- HalfNormalDistribution$new(sd = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  # HalfNormal(1): 2 * dnorm(1, 0, 1) for x >= 0
  expected <- log(2) + dnorm(1, 0, 1, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("HalfCauchy log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- HalfCauchyDistribution$new(scale = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  # 2/(pi * scale) * 1/(1 + (x/scale)^2)
  expected <- log(2 / (pi * 1) * 1 / (1 + 1^2))
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("Gamma log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- GammaDistribution$new(shape = 2, rate = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dgamma(1, shape = 2, rate = 1, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Beta log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- BetaDistribution$new(alpha = 2, beta = 5)
  x <- torch_tensor(0.3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dbeta(0.3, 2, 5, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Exponential log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- ExponentialDistribution$new(rate = 2)
  x <- torch_tensor(0.5, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dexp(0.5, rate = 2, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("Poisson log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- PoissonDistribution$new(rate = 3)
  x <- torch_tensor(2, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dpois(2, lambda = 3, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Bernoulli log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- BernoulliDistribution$new(prob = 0.7)
  x1 <- torch_tensor(1, dtype = torch_float32())
  x0 <- torch_tensor(0, dtype = torch_float32())
  expect_equal(dist$log_prob(x1)$item(), log(0.7), tolerance = 1e-4)
  expect_equal(dist$log_prob(x0)$item(), log(0.3), tolerance = 1e-4)
})

test_that("StudentT log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- StudentTDistribution$new(df = 5, mu = 0, sigma = 1)
  x <- torch_tensor(1, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dt(1, df = 5, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Uniform log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- UniformDistribution$new(lower = 0, upper = 5)
  x <- torch_tensor(2.5, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dunif(2.5, 0, 5, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("Binomial log_prob is correct", {
  skip_if_not_installed("torch")
  dist <- BinomialDistribution$new(size = 10, prob = 0.3)
  x <- torch_tensor(3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dbinom(3, 10, 0.3, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Distribution sampling produces tensors", {
  skip_if_not_installed("torch")

  dists <- list(
    NormalDistribution$new(0, 1),
    HalfNormalDistribution$new(1),
    HalfCauchyDistribution$new(1),
    ExponentialDistribution$new(1),
    GammaDistribution$new(2, 1),
    BetaDistribution$new(2, 2),
    PoissonDistribution$new(3),
    BernoulliDistribution$new(0.5),
    UniformDistribution$new(0, 1)
  )

  for (d in dists) {
    s <- d$sample(5L)
    expect_true(inherits(s, "torch_tensor"), label = d$name)
  }
})
