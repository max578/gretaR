# Tests for P1 distributions: Dirichlet, Negative Binomial, LKJ Correlation

# =============================================================================
# Dirichlet distribution
# =============================================================================

test_that("Dirichlet log_prob matches known values", {
  skip_if_not_installed("torch")

  # Dirichlet(1, 1, 1) is uniform on the 2-simplex
  # log p(x) = lgamma(3) - 3*lgamma(1) + 0 = log(2) = 0.6931472
  dist <- DirichletDistribution$new(concentration = c(1, 1, 1))
  x <- torch_tensor(c(1/3, 1/3, 1/3), dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  # For Dir(1,1,1): log(Gamma(3)) - 3*log(Gamma(1)) + sum(0*log(x)) = log(2!)
  expected <- lgamma(3) - 3 * lgamma(1)  # = log(2)
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("Dirichlet log_prob with non-uniform concentration", {
  skip_if_not_installed("torch")

  alpha <- c(2, 5, 1)
  x_val <- c(0.2, 0.7, 0.1)
  dist <- DirichletDistribution$new(concentration = alpha)
  x <- torch_tensor(x_val, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()

  # Manual computation in R
  expected <- lgamma(sum(alpha)) - sum(lgamma(alpha)) +
    sum((alpha - 1) * log(x_val))
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Dirichlet log_prob with 2-element concentration matches Beta", {
  skip_if_not_installed("torch")

  # Dirichlet(a, b) on (x, 1-x) should match Beta(a, b) at x
  a <- 3
  b <- 7
  x_val <- 0.4
  dist <- DirichletDistribution$new(concentration = c(a, b))
  x <- torch_tensor(c(x_val, 1 - x_val), dtype = torch_float32())
  lp <- dist$log_prob(x)$item()

  expected <- dbeta(x_val, a, b, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Dirichlet sampling produces valid simplex output", {
  skip_if_not_installed("torch")

  dist <- DirichletDistribution$new(concentration = c(2, 3, 5))
  s <- dist$sample(10L)
  expect_true(inherits(s, "torch_tensor"))
  # Check shape: 10 x 3

  expect_equal(s$shape, c(10, 3))
  # Check all values >= 0
  expect_true(all(as.numeric(s) >= 0))
  # Check rows sum to ~1
  row_sums <- as.numeric(torch_sum(s, dim = 2))
  expect_equal(row_sums, rep(1, 10), tolerance = 1e-5)
})

# =============================================================================
# Negative Binomial distribution
# =============================================================================

test_that("Negative Binomial log_prob matches R dnbinom", {

  skip_if_not_installed("torch")

  # NB(size = 5, prob = 0.4): P(X = 3)
  dist <- NegativeBinomialDistribution$new(size = 5, prob = 0.4)
  x <- torch_tensor(3, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dnbinom(3, size = 5, prob = 0.4, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Negative Binomial log_prob at x = 0", {
  skip_if_not_installed("torch")

  dist <- NegativeBinomialDistribution$new(size = 3, prob = 0.7)
  x <- torch_tensor(0, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dnbinom(0, size = 3, prob = 0.7, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Negative Binomial log_prob with large x", {
  skip_if_not_installed("torch")

  dist <- NegativeBinomialDistribution$new(size = 10, prob = 0.5)
  x <- torch_tensor(20, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dnbinom(20, size = 10, prob = 0.5, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("Negative Binomial log_prob with size = 1 matches Geometric", {
  skip_if_not_installed("torch")

  # NB(1, p) is the Geometric distribution
  p <- 0.3
  x_val <- 5
  dist <- NegativeBinomialDistribution$new(size = 1, prob = p)
  x <- torch_tensor(x_val, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expected <- dgeom(x_val, prob = p, log = TRUE)
  expect_equal(lp, expected, tolerance = 1e-3)
})

# =============================================================================
# LKJ Correlation distribution
# =============================================================================

test_that("LKJ log_prob with eta = 1 gives log_det = 0 for identity", {
  skip_if_not_installed("torch")

  # For eta = 1: (1 - 1) * log(det(I)) = 0

  dist <- LKJDistribution$new(eta = 1, dim_mat = 3L)
  R <- torch_eye(3, dtype = torch_float32())
  lp <- dist$log_prob(R)$item()
  expect_equal(lp, 0, tolerance = 1e-6)
})

test_that("LKJ log_prob with eta = 2 and identity matrix", {
  skip_if_not_installed("torch")

  # (2 - 1) * log(det(I)) = 1 * 0 = 0
  dist <- LKJDistribution$new(eta = 2, dim_mat = 3L)
  R <- torch_eye(3, dtype = torch_float32())
  lp <- dist$log_prob(R)$item()
  expect_equal(lp, 0, tolerance = 1e-6)
})

test_that("LKJ log_prob with non-identity correlation matrix", {
  skip_if_not_installed("torch")

  # Build a valid 2x2 correlation matrix: [[1, rho], [rho, 1]]
  rho <- 0.5
  R_mat <- matrix(c(1, rho, rho, 1), nrow = 2)
  R <- torch_tensor(R_mat, dtype = torch_float32())
  eta <- 2

  dist <- LKJDistribution$new(eta = eta, dim_mat = 2L)
  lp <- dist$log_prob(R)$item()

  # Expected: (eta - 1) * log(det(R)) = 1 * log(1 - 0.25) = log(0.75)
  expected <- (eta - 1) * log(det(R_mat))
  expect_equal(lp, expected, tolerance = 1e-4)
})

test_that("LKJ log_prob with larger correlation matrix", {
  skip_if_not_installed("torch")

  # 3x3 correlation matrix with known determinant
  R_mat <- matrix(c(
    1.0, 0.3, 0.1,
    0.3, 1.0, 0.2,
    0.1, 0.2, 1.0
  ), nrow = 3, byrow = TRUE)
  R <- torch_tensor(R_mat, dtype = torch_float32())
  eta <- 3

  dist <- LKJDistribution$new(eta = eta, dim_mat = 3L)
  lp <- dist$log_prob(R)$item()

  expected <- (eta - 1) * log(det(R_mat))
  expect_equal(lp, expected, tolerance = 1e-3)
})

test_that("LKJ rejects dimension < 2", {
  skip_if_not_installed("torch")
  expect_error(lkj_correlation(eta = 1, dim = 1L), "dimension must be >= 2")
})

test_that("LKJ higher eta penalises off-diagonal correlations more", {
  skip_if_not_installed("torch")

  # With det < 1, higher eta should give lower log_prob
  rho <- 0.6
  R_mat <- matrix(c(1, rho, rho, 1), nrow = 2)
  R <- torch_tensor(R_mat, dtype = torch_float32())

  dist_low <- LKJDistribution$new(eta = 1, dim_mat = 2L)
  dist_high <- LKJDistribution$new(eta = 5, dim_mat = 2L)

  lp_low <- dist_low$log_prob(R)$item()
  lp_high <- dist_high$log_prob(R)$item()

  # eta = 1 gives 0 (uniform); eta = 5 gives 4*log(det) which is negative
  expect_equal(lp_low, 0, tolerance = 1e-6)
  expect_true(lp_high < lp_low)
})
