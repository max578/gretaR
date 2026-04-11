# Tests for truncation support

test_that("normal with truncation sets correct constraint", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- normal(0, 1, truncation = c(0, Inf))
  node <- get_node(x)
  expect_equal(node$constraint$lower, 0)
  expect_equal(node$constraint$upper, Inf)
  expect_true(inherits(node$transform, "LogTransform"))
})

test_that("normal with two-sided truncation", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- normal(0, 1, truncation = c(-2, 2))
  node <- get_node(x)
  expect_equal(node$constraint$lower, -2)
  expect_equal(node$constraint$upper, 2)
  expect_true(inherits(node$transform, "ScaledLogitTransform"))
})

test_that("student_t with truncation", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- student_t(df = 3, mu = 0, sigma = 1, truncation = c(0, Inf))
  node <- get_node(x)
  expect_equal(node$constraint$lower, 0)
})

test_that("gamma_dist with truncation tightens upper bound", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- gamma_dist(shape = 2, rate = 1, truncation = c(0, 5))
  node <- get_node(x)
  expect_equal(node$constraint$lower, 0)
  expect_equal(node$constraint$upper, 5)
})

test_that("beta_dist with truncation narrows bounds", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- beta_dist(alpha = 2, beta = 5, truncation = c(0.1, 0.9))
  node <- get_node(x)
  expect_equal(node$constraint$lower, 0.1)
  expect_equal(node$constraint$upper, 0.9)
})

test_that("exponential with truncation", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- exponential(rate = 1, truncation = c(0, 10))
  node <- get_node(x)
  expect_equal(node$constraint$lower, 0)
  expect_equal(node$constraint$upper, 10)
})

test_that("truncated normal log_prob evaluates correctly", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # Truncated normal should still compute log_prob (untruncated kernel)
  x <- normal(0, 1, truncation = c(0, Inf))
  dist <- get_node(x)$distribution
  val <- torch_tensor(1.0, dtype = torch_float32())
  lp <- dist$log_prob(val)$item()
  expect_true(is.finite(lp))
  # Should equal the untruncated log_prob (normalising constant is not included)
  expect_equal(lp, dnorm(1, 0, 1, log = TRUE), tolerance = 1e-3)
})

test_that("truncated normal in a model compiles and runs", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  # Positive normal
  mu <- normal(0, 5, truncation = c(0, Inf))
  y <- as_data(abs(rnorm(50, 3, 1)))
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- opt(m, verbose = FALSE)
  expect_true(coef(fit)["mu"] > 0)
})

test_that("no truncation by default", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- normal(0, 1)
  node <- get_node(x)
  expect_null(node$distribution$truncation)
  expect_equal(node$constraint$lower, -Inf)
  expect_equal(node$constraint$upper, Inf)
})
