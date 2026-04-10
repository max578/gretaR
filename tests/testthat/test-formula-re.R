# Tests for lme4-style random effects formula parsing and fitting

# =============================================================================
# parse_re_bars() tests
# =============================================================================

test_that("parse_re_bars extracts (1|group) as intercept type", {
  result <- parse_re_bars(y ~ x + (1 | group))
  expect_length(result, 1)
  expect_equal(result[[1]]$type, "intercept")
  expect_equal(result[[1]]$group, "group")
  expect_equal(result[[1]]$lhs, "1")
  expect_equal(result[[1]]$slope_vars, character(0))
})

test_that("parse_re_bars extracts (x|group) as intercept_slope type", {
  result <- parse_re_bars(y ~ x + (x | group))
  expect_length(result, 1)
  expect_equal(result[[1]]$type, "intercept_slope")
  expect_equal(result[[1]]$group, "group")
  expect_equal(result[[1]]$slope_vars, "x")
})

test_that("parse_re_bars extracts (0 + x|group) as slope_only type", {
  result <- parse_re_bars(y ~ x + (0 + x | group))
  expect_length(result, 1)

  expect_equal(result[[1]]$type, "slope_only")
  expect_equal(result[[1]]$group, "group")
  expect_equal(result[[1]]$slope_vars, "x")
})

test_that("parse_re_bars handles multiple random effect terms", {
  result <- parse_re_bars(y ~ x + (1 | site) + (1 | year))
  expect_length(result, 2)
  expect_equal(result[[1]]$group, "site")
  expect_equal(result[[2]]$group, "year")
  expect_equal(result[[1]]$type, "intercept")
  expect_equal(result[[2]]$type, "intercept")
})

test_that("parse_re_bars handles compact spacing", {
 result <- parse_re_bars(y ~ x + (1|group))
  expect_length(result, 1)
  expect_equal(result[[1]]$type, "intercept")
  expect_equal(result[[1]]$group, "group")
})

test_that("parse_re_bars errors on formula without bars", {
  expect_error(parse_re_bars(y ~ x1 + x2), "No random effect terms")
})

# =============================================================================
# remove_re_bars() tests
# =============================================================================

test_that("remove_re_bars strips bar terms and returns fixed formula", {
  result <- remove_re_bars(y ~ x + (1 | group))
  # Should be equivalent to y ~ x
  expect_true(inherits(result, "formula"))
  result_str <- deparse(result, width.cutoff = 500)
  expect_false(grepl("\\|", result_str))
  expect_true(grepl("x", result_str))
})

test_that("remove_re_bars handles formula with only random effects", {
  result <- remove_re_bars(y ~ (1 | group))
  result_str <- deparse(result, width.cutoff = 500)
  expect_false(grepl("\\|", result_str))
  # Should have an intercept at minimum
  expect_true(grepl("~", result_str))
})

test_that("remove_re_bars handles multiple bar terms", {
  result <- remove_re_bars(y ~ x + z + (1 | site) + (1 | year))
  result_str <- deparse(result, width.cutoff = 500)
  expect_false(grepl("\\|", result_str))
  expect_true(grepl("x", result_str))
  expect_true(grepl("z", result_str))
})

# =============================================================================
# detect_formula_style() recognises lme4 patterns
# =============================================================================

test_that("detect_formula_style classifies bar formulas as lme4", {
  expect_equal(detect_formula_style(y ~ x + (1 | group)), "lme4")
  expect_equal(detect_formula_style(y ~ (x | group)), "lme4")
  expect_equal(detect_formula_style(y ~ x + (0 + x | group)), "lme4")
})

# =============================================================================
# gretaR_glm() mixed-model fitting
# =============================================================================

test_that("gretaR_glm fits a random intercepts model with MAP", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(42)
  n_groups <- 5
  n_per <- 20
  group <- factor(rep(1:n_groups, each = n_per))
  group_effects <- rnorm(n_groups, 0, 2)
  x <- rnorm(n_groups * n_per)
  y <- 3 + 1.5 * x + group_effects[as.integer(group)] + rnorm(n_groups * n_per, 0, 0.5)
  dat <- data.frame(y = y, x = x, group = group)

  fit <- gretaR_glm(y ~ x + (1 | group), data = dat,
                     family = "gaussian", sampler = "map", verbose = FALSE)

  expect_s3_class(fit, "gretaR_glm_fit")
  expect_equal(fit$family, "gaussian")
  expect_false(is.null(fit$random_effects))
  expect_length(fit$random_effects, 1)
  expect_equal(fit$random_effects[[1]]$group, "group")
  expect_equal(fit$random_effects[[1]]$n_groups, n_groups)
  expect_equal(fit$random_effects[[1]]$type, "intercept")
  expect_true(!is.null(fit$result$par))
})

test_that("gretaR_glm fits multiple random intercepts with MAP", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(123)
  n <- 60
  site <- factor(rep(1:3, each = 20))
  year <- factor(rep(1:4, times = 15))
  x <- rnorm(n)
  y <- 1 + 0.5 * x + rnorm(n, 0, 0.3)
  dat <- data.frame(y = y, x = x, site = site, year = year)

  fit <- gretaR_glm(y ~ x + (1 | site) + (1 | year), data = dat,
                     family = "gaussian", sampler = "map", verbose = FALSE)

  expect_s3_class(fit, "gretaR_glm_fit")
  expect_length(fit$random_effects, 2)
  expect_equal(fit$random_effects[[1]]$group, "site")
  expect_equal(fit$random_effects[[2]]$group, "year")
})

# =============================================================================
# Graceful rejection of unsupported patterns
# =============================================================================

test_that("gretaR_glm rejects missing grouping variable", {
  skip_if_not_installed("torch")

  dat <- data.frame(y = 1:5, x = rnorm(5))
  expect_error(
    gretaR_glm(y ~ x + (1 | nonexistent), data = dat,
               sampler = "map", verbose = FALSE),
    "not found"
  )
})

test_that("mgcv-style formulas are handled", {
  skip_if_not_installed("torch")
  skip_if_not_installed("mgcv")
  skip_on_cran()

  set.seed(42)
  dat <- data.frame(y = rnorm(50), x = rnorm(50))
  fit <- gretaR_glm(y ~ s(x, k = 6), data = dat, sampler = "map",
                     verbose = FALSE)
  expect_s3_class(fit, "gretaR_fit")
})

# =============================================================================
# print/summary S3 methods for mixed models
# =============================================================================

test_that("print.gretaR_glm_fit displays random effects info", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(99)
  dat <- data.frame(
    y = rnorm(30), x = rnorm(30),
    group = factor(rep(1:3, each = 10))
  )
  fit <- gretaR_glm(y ~ x + (1 | group), data = dat,
                     sampler = "map", verbose = FALSE)

  output <- capture.output(print(fit))
  expect_true(any(grepl("mixed", output)))
  expect_true(any(grepl("Random effects", output)))
  expect_true(any(grepl("group", output)))
})
