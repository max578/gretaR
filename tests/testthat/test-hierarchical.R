# Tests for hierarchical (multi-level) model support
# Requires the [.gretaR_array indexing operator

# =============================================================================
# Basic indexing tests
# =============================================================================

test_that("[.gretaR_array works with integer vector indexing", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # Create a group-level parameter vector
  alpha <- normal(0, 10, dim = c(5L, 1L))
  node_alpha <- get_node(alpha)

  # Set known values for testing
  node_alpha$value <- torch_tensor(matrix(c(1, 2, 3, 4, 5), ncol = 1L))

  # Index with a plain R integer vector (the hierarchical model pattern)
  group_id <- c(1L, 1L, 2L, 3L, 3L, 4L, 5L)
  result <- alpha[group_id]

  expect_s3_class(result, "gretaR_array")
  expect_equal(dim(result), c(7L, 1L))

  # Verify compute() returns the correct values
  result_node <- get_node(result)
  computed <- result_node$compute()
  expected <- torch_tensor(matrix(c(1, 1, 2, 3, 3, 4, 5), ncol = 1L))
  expect_equal(as.numeric(computed), as.numeric(expected))
})

test_that("[.gretaR_array works with logical indexing", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10, dim = c(5L, 1L))
  node_alpha <- get_node(alpha)
  node_alpha$value <- torch_tensor(matrix(c(10, 20, 30, 40, 50), ncol = 1L))

  mask <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
  result <- alpha[mask]

  expect_s3_class(result, "gretaR_array")
  expect_equal(dim(result), c(3L, 1L))

  computed <- get_node(result)$compute()
  expect_equal(as.numeric(computed), c(10, 30, 50))
})

test_that("[.gretaR_array creates proper operation node in DAG", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10, dim = c(3L, 1L))
  result <- alpha[c(1L, 2L, 1L, 3L)]

  result_node <- get_node(result)
  expect_equal(result_node$node_type, "operation")
  expect_equal(length(result_node$parents), 2L)

  # First parent should be alpha, second should be the index data node
  parent1 <- .gretaR_env$dag$nodes[[result_node$parents[1]]]
  parent2 <- .gretaR_env$dag$nodes[[result_node$parents[2]]]
  expect_equal(parent1$node_type, "variable")
  expect_equal(parent2$node_type, "data")
})

test_that("[.gretaR_array handles 2D arrays (selecting rows)", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # 4 groups, 2 columns (e.g., random intercept + slope per group)
  alpha <- normal(0, 10, dim = c(4L, 2L))
  node_alpha <- get_node(alpha)
  node_alpha$value <- torch_tensor(matrix(1:8, nrow = 4L, ncol = 2L,
                                          byrow = FALSE))

  idx <- c(1L, 3L, 3L, 2L, 4L, 1L)
  result <- alpha[idx]

  expect_equal(dim(result), c(6L, 2L))
  computed <- get_node(result)$compute()
  expect_equal(computed$shape, c(6L, 2L))
})

# =============================================================================
# Hierarchical model: random intercepts
# =============================================================================

test_that("random intercepts model compiles correctly", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # Simulate grouped data
  set.seed(42)
  n_groups <- 5L
  n_per_group <- 20L
  n <- n_groups * n_per_group
  group_id <- rep(1:n_groups, each = n_per_group)
  true_mu <- 5
  true_tau <- 2
  true_sigma <- 1
  true_alpha <- rnorm(n_groups, true_mu, true_tau)
  y_obs <- rnorm(n, true_alpha[group_id], true_sigma)

  # Define gretaR model

  mu <- normal(0, 10)
  tau <- half_cauchy(5)
  alpha <- normal(mu, tau, dim = c(n_groups, 1L))
  sigma <- half_cauchy(5)

  y <- as_data(y_obs)
  fitted <- alpha[group_id]  # The key hierarchical indexing operation
  distribution(y) <- normal(fitted, sigma)

  # Compile
  m <- model(mu, tau, sigma, alpha)

  expect_s3_class(m, "gretaR_model")

  # Should have 4 named targets: mu, tau, sigma, alpha
  expect_true("mu" %in% unlist(lapply(m$param_info, `[[`, "name")))
  expect_true("tau" %in% unlist(lapply(m$param_info, `[[`, "name")))
  expect_true("sigma" %in% unlist(lapply(m$param_info, `[[`, "name")))
  expect_true("alpha" %in% unlist(lapply(m$param_info, `[[`, "name")))

  # Total dim: mu(1) + tau(1) + sigma(1) + alpha(5) = 8
  expect_equal(m$total_dim, 8L)
})

test_that("random intercepts model log_prob evaluates without error", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  set.seed(42)
  n_groups <- 5L
  n_per_group <- 20L
  n <- n_groups * n_per_group
  group_id <- rep(1:n_groups, each = n_per_group)
  true_mu <- 5
  true_tau <- 2
  true_sigma <- 1
  true_alpha <- rnorm(n_groups, true_mu, true_tau)
  y_obs <- rnorm(n, true_alpha[group_id], true_sigma)

  mu <- normal(0, 10)
  tau <- half_cauchy(5)
  alpha <- normal(mu, tau, dim = c(n_groups, 1L))
  sigma <- half_cauchy(5)

  y <- as_data(y_obs)
  fitted <- alpha[group_id]
  distribution(y) <- normal(fitted, sigma)

  m <- model(mu, tau, sigma, alpha)

  # Evaluate log_prob at a reasonable point in unconstrained space
  theta <- torch_zeros(m$total_dim, dtype = torch_float32())
  lp <- log_prob(m, theta)

  expect_true(inherits(lp, "torch_tensor"))
  expect_true(is.finite(lp$item()))
})

test_that("random intercepts model grad_log_prob evaluates without error", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  set.seed(42)
  n_groups <- 3L
  n_per_group <- 10L
  n <- n_groups * n_per_group
  group_id <- rep(1:n_groups, each = n_per_group)
  y_obs <- rnorm(n, rep(c(2, 5, 8), each = n_per_group), 1)

  mu <- normal(0, 10)
  tau <- half_cauchy(5)
  alpha <- normal(mu, tau, dim = c(n_groups, 1L))
  sigma <- half_cauchy(5)

  y <- as_data(y_obs)
  fitted <- alpha[group_id]
  distribution(y) <- normal(fitted, sigma)

  m <- model(mu, tau, sigma, alpha)

  # Evaluate gradient
  theta <- torch_zeros(m$total_dim, dtype = torch_float32())
  result <- grad_log_prob(m, theta)

  expect_true(is.finite(result$lp))
  expect_true(inherits(result$grad, "torch_tensor"))
  expect_equal(length(as.numeric(result$grad)), m$total_dim)
  expect_true(all(is.finite(as.numeric(result$grad))))
})

test_that("indexing result participates in arithmetic operations", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # alpha[group_id] + beta * x should work
  alpha <- normal(0, 10, dim = c(3L, 1L))
  beta <- normal(0, 5)
  x <- as_data(rnorm(9))
  group_id <- rep(1:3, each = 3L)

  fitted <- alpha[group_id] + beta * x
  expect_s3_class(fitted, "gretaR_array")
  expect_equal(dim(fitted), c(9L, 1L))
})

# =============================================================================
# Edge cases
# =============================================================================

test_that("[.gretaR_array rejects non-numeric index", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10, dim = c(3L, 1L))
  expect_error(alpha["a"], "must be integer")
})

test_that("[.gretaR_array works with single index", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10, dim = c(5L, 1L))
  node_alpha <- get_node(alpha)
  node_alpha$value <- torch_tensor(matrix(c(10, 20, 30, 40, 50), ncol = 1L))

  result <- alpha[3L]
  expect_equal(dim(result), c(1L, 1L))
  expect_equal(as.numeric(get_node(result)$compute()), 30)
})

test_that("[.gretaR_array works with repeated indices", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  alpha <- normal(0, 10, dim = c(2L, 1L))
  node_alpha <- get_node(alpha)
  node_alpha$value <- torch_tensor(matrix(c(100, 200), ncol = 1L))

  result <- alpha[c(1L, 1L, 2L, 2L, 1L)]
  expect_equal(dim(result), c(5L, 1L))
  expect_equal(as.numeric(get_node(result)$compute()), c(100, 100, 200, 200, 100))
})
