# Regression coverage for the multivariate-normal log-density.
#
# The MVN log_prob had no test, which let two non-existent torch symbols
# (torch_linalg_cholesky / torch_linalg_solve_triangular) survive: any MVN or
# GP-prior model erred at evaluation. These tests pin the density against an
# independent base-R reference (chol + backsolve), so the Cholesky path cannot
# silently regress again. The base-R reference needs no extra dependency.

mvn_logprob_ref <- function(x, mu, sigma) {
  R <- chol(sigma)
  z <- backsolve(R, x - mu, transpose = TRUE)
  -0.5 * (length(mu) * log(2 * pi) + 2 * sum(log(diag(R))) + sum(z * z))
}

test_that("MultivariateNormal log_prob matches a base-R reference (3-D)", {
  skip_if_not_installed("torch")
  mu <- c(0.5, -1, 2)
  sigma <- matrix(c(2, 0.3, 0.1,
                    0.3, 1, -0.2,
                    0.1, -0.2, 1.5), 3L, 3L)
  dist <- MultivariateNormalDistribution$new(mean = mu, covariance = sigma)
  xv <- c(0.2, -0.5, 1.1)
  x <- torch_tensor(xv, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expect_equal(lp, mvn_logprob_ref(xv, mu, sigma), tolerance = 1e-4)
})

test_that("MultivariateNormal log_prob matches a base-R reference (2-D, correlated)", {
  skip_if_not_installed("torch")
  mu <- c(0, 0)
  sigma <- matrix(c(1, 0.8, 0.8, 1), 2L, 2L)
  dist <- MultivariateNormalDistribution$new(mean = mu, covariance = sigma)
  xv <- c(1.3, -0.4)
  x <- torch_tensor(xv, dtype = torch_float32())
  lp <- dist$log_prob(x)$item()
  expect_equal(lp, mvn_logprob_ref(xv, mu, sigma), tolerance = 1e-4)
})
