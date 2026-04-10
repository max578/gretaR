# Tests for gretaR_array and DSL

test_that("as_data creates data nodes", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- as_data(c(1, 2, 3))
  expect_s3_class(x, "gretaR_array")
  node <- get_node(x)
  expect_equal(node$node_type, "data")
  expect_equal(dim(x), c(3L, 1L))
})

test_that("as_data handles matrices", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  m <- matrix(1:6, nrow = 3, ncol = 2)
  x <- as_data(m)
  expect_equal(dim(x), c(3L, 2L))
})

test_that("variable creates free variable nodes", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  v <- variable()
  node <- get_node(v)
  expect_equal(node$node_type, "variable")
  expect_equal(dim(v), c(1L, 1L))
})

test_that("variable with bounds creates constrained node", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  v <- variable(lower = 0)
  node <- get_node(v)
  expect_equal(node$constraint$lower, 0)
  expect_true(inherits(node$transform, "LogTransform"))
})

test_that("arithmetic on gretaR_arrays builds operation nodes", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  a <- as_data(c(1, 2, 3))
  b <- as_data(c(4, 5, 6))
  c_arr <- a + b

  node <- get_node(c_arr)
  expect_equal(node$node_type, "operation")
  expect_equal(length(node$parents), 2L)
})

test_that("scalar-array arithmetic works", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  a <- as_data(c(1, 2, 3))
  b <- a * 2
  node <- get_node(b)
  expect_equal(node$node_type, "operation")
})

test_that("math functions work on gretaR_arrays", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  a <- as_data(c(1, 2, 3))
  b <- log(a)
  node <- get_node(b)
  expect_equal(node$node_type, "operation")
})

test_that("distribution<- assigns likelihood", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  y <- as_data(rnorm(10))
  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  distribution(y) <- normal(mu, sigma)

  expect_true(length(.gretaR_env$distributions) > 0)
})

test_that("normal() creates a variable with distribution", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- normal(0, 1)
  node <- get_node(x)
  expect_equal(node$node_type, "variable")
  expect_equal(node$distribution$name, "normal")
})

test_that("print.gretaR_array works", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- normal(0, 1)
  expect_output(print(x), "gretaR array")
})
