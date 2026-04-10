# Tests for sparse matrix support

test_that("as_data handles sparse matrices", {
  skip_if_not_installed("torch")
  skip_if_not_installed("Matrix")
  reset_gretaR_env()

  m <- Matrix::sparseMatrix(
    i = c(1, 2, 3), j = c(1, 2, 3),
    x = c(1.0, 2.0, 3.0), dims = c(3, 3)
  )
  x <- as_data(m)
  expect_s3_class(x, "gretaR_array")
  node <- get_node(x)
  expect_true(node$value$is_sparse())
  expect_equal(dim(x), c(3L, 3L))
})

test_that("sparse %*% dense works correctly", {
  skip_if_not_installed("torch")
  skip_if_not_installed("Matrix")
  reset_gretaR_env()

  # Create sparse design matrix
  X_sparse <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(1, 2, 1, 2, 1, 2),
    x = c(1.0, 0.5, 0.5, 1.0, 1.0, 1.5),
    dims = c(3, 2)
  )
  X <- as_data(X_sparse)
  beta <- as_data(c(2, 3))

  result <- X %*% beta
  node <- get_node(result)
  expect_equal(node$node_type, "operation")
  expect_equal(dim(result), c(3L, 1L))
})

test_that("as_data rejects sparse matrices with NA", {
  skip_if_not_installed("torch")
  skip_if_not_installed("Matrix")
  reset_gretaR_env()

  m <- Matrix::sparseMatrix(
    i = c(1, 2), j = c(1, 2),
    x = c(1.0, NA), dims = c(2, 2)
  )
  expect_error(as_data(m), "Missing values")
})
