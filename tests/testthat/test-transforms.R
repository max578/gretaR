# Tests for parameter transforms (bijectors)

test_that("IdentityTransform is a no-op", {
  skip_if_not_installed("torch")
  t <- IdentityTransform$new()
  x <- torch_tensor(c(1, 2, 3))
  expect_equal(as.numeric(t$forward(x)), c(1, 2, 3))
  expect_equal(as.numeric(t$inverse(x)), c(1, 2, 3))
  expect_equal(t$log_det_jacobian(x)$item(), 0)
})

test_that("LogTransform maps positive reals to reals", {
  skip_if_not_installed("torch")
  t <- LogTransform$new()
  x <- torch_tensor(c(0.5, 1, 2))
  y <- t$forward(x)
  x_back <- t$inverse(y)
  expect_equal(as.numeric(x_back), as.numeric(x), tolerance = 1e-5)

  # log|det J| = sum(y) for exp transform
  y_test <- torch_tensor(c(0.0, 1.0))
  ldj <- t$log_det_jacobian(y_test)$item()
  expect_equal(ldj, 1.0, tolerance = 1e-5)
})

test_that("LogitTransform maps (0,1) to reals", {
  skip_if_not_installed("torch")
  t <- LogitTransform$new()
  x <- torch_tensor(c(0.2, 0.5, 0.8))
  y <- t$forward(x)
  x_back <- t$inverse(y)
  expect_equal(as.numeric(x_back), as.numeric(x), tolerance = 1e-5)
})

test_that("ScaledLogitTransform maps (a,b) to reals", {
  skip_if_not_installed("torch")
  t <- ScaledLogitTransform$new(lower = 2, upper = 5)
  x <- torch_tensor(c(2.5, 3.5, 4.5))
  y <- t$forward(x)
  x_back <- t$inverse(y)
  expect_equal(as.numeric(x_back), as.numeric(x), tolerance = 1e-4)
})

test_that("select_transform picks correct transform", {
  skip_if_not_installed("torch")
  expect_s3_class(select_transform(), "IdentityTransform")
  expect_s3_class(select_transform(lower = 0), "LogTransform")
  expect_s3_class(select_transform(lower = 0, upper = 1), "LogitTransform")
  expect_s3_class(select_transform(lower = 2, upper = 5), "ScaledLogitTransform")
  expect_s3_class(select_transform(lower = -1), "LowerBoundTransform")
})

test_that("round-trip transform preserves values for all types", {
  skip_if_not_installed("torch")

  transforms_and_values <- list(
    list(t = LogTransform$new(), x = torch_tensor(c(0.1, 1, 10))),
    list(t = LogitTransform$new(), x = torch_tensor(c(0.1, 0.5, 0.9))),
    list(t = ScaledLogitTransform$new(-2, 2), x = torch_tensor(c(-1.5, 0, 1.5))),
    list(t = LowerBoundTransform$new(3), x = torch_tensor(c(3.1, 5, 10)))
  )

  for (tv in transforms_and_values) {
    y <- tv$t$forward(tv$x)
    x_back <- tv$t$inverse(y)
    expect_equal(as.numeric(x_back), as.numeric(tv$x), tolerance = 1e-4,
                 label = class(tv$t)[1])
  }
})
