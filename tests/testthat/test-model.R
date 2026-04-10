# Tests for model compilation

test_that("model() compiles a simple model", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  y <- as_data(rnorm(20, 3, 1))
  distribution(y) <- normal(mu, sigma)

  m <- model(mu, sigma)
  expect_s3_class(m, "gretaR_model")
  expect_equal(m$total_dim, 2L)
  expect_true(length(m$likelihood_terms) > 0)
})

test_that("model() identifies all variables", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10)
  beta <- normal(0, 5)
  sigma <- half_cauchy(1)

  m <- model(alpha, beta, sigma)
  expect_equal(length(m$var_order), 3L)
})

test_that("log_prob evaluates without error", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 10)
  y <- as_data(c(1, 2, 3))
  distribution(y) <- normal(mu, 1)

  m <- model(mu)
  theta <- torch_zeros(1L)

  lp <- log_prob(m, theta)
  expect_true(inherits(lp, "torch_tensor"))
  expect_true(is.finite(lp$item()))
})

test_that("grad_log_prob returns gradient", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 10)
  y <- as_data(c(1, 2, 3))
  distribution(y) <- normal(mu, 1)

  m <- model(mu)
  theta <- torch_zeros(1L)

  result <- grad_log_prob(m, theta)
  expect_true(is.finite(result$lp))
  expect_true(inherits(result$grad, "torch_tensor"))
  expect_equal(length(as.numeric(result$grad)), 1L)
  expect_true(is.finite(as.numeric(result$grad)))
})

test_that("print.gretaR_model works", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, sigma)

  m <- model(mu, sigma)
  expect_output(print(m), "gretaR model")
})
