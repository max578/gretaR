# test-refusal-contract.R — orchestra-leader-routable classed refusals
#
# Every orchestra-facing verb that declines an input must raise a condition
# whose class an orchestra leader can route on via the shared predicate
# `is_orchestra_decline()` (integration/refusal_contract.R in ORCHESTRA_dev:
# TRUE when class(x) contains "orchestra_refusal", or any class element
# matches "_(refusal|abstention)$"). Before this change every decline in
# gretaR was a bare `cli_abort()`, surfacing only as `rlang_error`/`error`/
# `condition` -- indistinguishable, by class, from any other error in the
# federation. This file grades one representative refusal per converted
# entry point against the leader's own predicate logic.

is_decline <- function(cls) {
  any(cls == "orchestra_refusal") || any(grepl("_(refusal|abstention)$", cls))
}

test_that("model() refuses with a classed condition when no targets are given", {
  err <- tryCatch(model(), error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("normal() refuses with a classed condition on an unresolvable parameter", {
  err <- tryCatch(normal(NULL, 1), error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("compile_to_stan() refuses with a classed condition on a non-model input", {
  err <- tryCatch(compile_to_stan(NULL), error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("mcmc() refuses with a classed condition when batched NUTS is requested", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  y <- as_data(rnorm(20, 3, 1))
  distribution(y) <- normal(mu, sigma)
  m <- model(mu, sigma)

  err <- tryCatch(
    mcmc(m, sampler = "nuts", batched = TRUE, verbose = FALSE),
    error = function(e) e
  )
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("gretaR_glm() refuses with a classed condition when a grouping variable is missing", {
  dat <- data.frame(y = rnorm(10), x = rnorm(10))
  err <- tryCatch(
    gretaR_glm(y ~ x + (1 | site), data = dat),
    error = function(e) e
  )
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("random_effect() refuses with a classed condition on a bad group argument", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  tau <- half_normal(2)
  err <- tryCatch(
    random_effect(group = c(1.5, 2.5), n_groups = 2, sd = tau),
    error = function(e) e
  )
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("model_from_arrays() refuses with a classed condition on no targets", {
  err <- tryCatch(model_from_arrays(targets = list()), error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("custom_distribution() refuses with a classed condition on a non-function log_prob", {
  err <- tryCatch(custom_distribution(log_prob_fn = "not a function"),
                   error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("process_smooths() refuses with a classed condition on a formula with no smooth terms", {
  err <- tryCatch(process_smooths(y ~ x, data = data.frame(y = 1, x = 1)),
                   error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_true(is_decline(class(err)))
})

test_that("distribution<-() refuses with a classed condition when the LHS is not a gretaR_array", {
  x <- 1
  err <- tryCatch(distribution(x) <- normal(0, 1), error = function(e) e)
  expect_s3_class(err, "gretaR_refusal")
  expect_s3_class(err, "gretaR_invalid_input_refusal")
  expect_true(is_decline(class(err)))
})

test_that("gretaR_abort() unites the reason-code class with the shared orchestra_refusal marker", {
  err <- tryCatch(
    gretaR_abort("test message", reason_code = "backend_unavailable"),
    error = function(e) e
  )
  expect_identical(
    class(err)[1:6],
    c("gretaR_backend_unavailable_refusal", "gretaR_refusal", "orchestra_refusal",
      "rlang_error", "error", "condition")
  )
  expect_true(is_decline(class(err)))
})

test_that("gretaR_abort() rejects an unknown reason_code rather than minting a bad class", {
  expect_error(
    gretaR_abort("test message", reason_code = "not_a_real_code"),
    class = "rlang_error"
  )
})
