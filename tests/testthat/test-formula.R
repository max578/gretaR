# Tests for formula interface

test_that("detect_formula_style identifies base R formulas", {
  expect_equal(detect_formula_style(y ~ x1 + x2), "base")
  expect_equal(detect_formula_style(y ~ x1 * x2 + I(x1^2)), "base")
})

test_that("detect_formula_style identifies lme4-style formulas", {
  expect_equal(detect_formula_style(y ~ x + (1 | group)), "lme4")
  expect_equal(detect_formula_style(y ~ x + (x | group)), "lme4")
})

test_that("detect_formula_style identifies mgcv-style formulas", {
  expect_equal(detect_formula_style(y ~ s(x1) + x2), "mgcv")
  expect_equal(detect_formula_style(y ~ te(x1, x2)), "mgcv")
})

test_that("detect_formula_style respects explicit style", {
  expect_equal(detect_formula_style(y ~ x, explicit = "brms"), "brms")
})

test_that("gretaR_glm fits a Gaussian model with MAP", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(123)
  dat <- data.frame(
    y = 2 + 3 * rnorm(50) + rnorm(50, 0, 0.5),
    x = rnorm(50)
  )

  fit <- gretaR_glm(y ~ x, data = dat, family = "gaussian",
                     sampler = "map", verbose = FALSE)
  expect_s3_class(fit, "gretaR_glm_fit")
  expect_equal(fit$family, "gaussian")
  expect_true(!is.null(fit$result$par))
})

test_that("gretaR_glm rejects lme4-style formulas gracefully", {
  skip_if_not_installed("torch")
  dat <- data.frame(y = 1, x = 1, group = 1)
  expect_error(
    gretaR_glm(y ~ x + (1 | group), data = dat, sampler = "map",
               verbose = FALSE),
    "not yet supported"
  )
})

test_that("print.gretaR_glm_fit works", {
  skip_if_not_installed("torch")
  skip_on_cran()

  dat <- data.frame(y = rnorm(20), x = rnorm(20))
  fit <- gretaR_glm(y ~ x, data = dat, sampler = "map", verbose = FALSE)
  expect_output(print(fit), "gretaR GLM fit")
})
