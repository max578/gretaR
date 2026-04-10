# Tests for mgcv smooth term integration

test_that("process_smooths extracts s() terms", {
  skip_if_not_installed("mgcv")
  dat <- data.frame(y = rnorm(50), x = rnorm(50))
  sm <- process_smooths(y ~ s(x, k = 8), data = dat)

  expect_true(!is.null(sm$smooth_Zs))
  expect_true(length(sm$smooth_Zs) >= 1)
  expect_true(sm$n_smooth_fixed >= 0)
  expect_equal(nrow(sm$smooth_Zs[[1]]), 50)
})

test_that("process_smooths handles multiple smooths", {
  skip_if_not_installed("mgcv")
  dat <- data.frame(y = rnorm(50), x1 = rnorm(50), x2 = rnorm(50))
  sm <- process_smooths(y ~ s(x1, k = 6) + s(x2, k = 5), data = dat)

  expect_true(length(sm$smooth_Zs) >= 2)
  expect_true(length(sm$smooth_info) >= 2)
})

test_that("process_smooths handles different basis types", {
  skip_if_not_installed("mgcv")
  dat <- data.frame(y = rnorm(50), x = rnorm(50))

  # Cubic regression spline
  sm_cr <- process_smooths(y ~ s(x, bs = "cr", k = 8), data = dat)
  expect_true(length(sm_cr$smooth_Zs) >= 1)

  # P-spline
  sm_ps <- process_smooths(y ~ s(x, bs = "ps", k = 8), data = dat)
  expect_true(length(sm_ps$smooth_Zs) >= 1)
})

test_that("gretaR_glm fits a GAM with MAP", {
  skip_if_not_installed("torch")
  skip_if_not_installed("mgcv")
  skip_on_cran()

  set.seed(42)
  dat <- data.frame(x = runif(100, 0, 2 * pi))
  dat$y <- sin(dat$x) + rnorm(100, 0, 0.3)

  fit <- gretaR_glm(y ~ s(x, k = 8), data = dat, family = "gaussian",
                     sampler = "map", verbose = FALSE)
  expect_s3_class(fit, "gretaR_fit")
  expect_true(!is.null(coef(fit)))
})

test_that("detect_formula_style identifies mgcv formulas", {
  expect_equal(detect_formula_style(y ~ s(x)), "mgcv")
  expect_equal(detect_formula_style(y ~ te(x1, x2)), "mgcv")
  expect_equal(detect_formula_style(y ~ x + ti(x1, x2)), "mgcv")
})

test_that("process_smooths errors without mgcv-style terms", {
  skip_if_not_installed("mgcv")
  dat <- data.frame(y = rnorm(10), x = rnorm(10))
  expect_error(process_smooths(y ~ x, data = dat), "No smooth terms")
})
