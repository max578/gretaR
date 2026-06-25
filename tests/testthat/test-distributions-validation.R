# Parameter validation at the distribution-constructor boundary.
#
# Every user-facing distribution must reject a NULL, a wholly missing required
# argument, a non-numeric parameter, and an NA-bearing numeric with a single
# clean caller-facing error, while still accepting numeric and gretaR_array
# parameters. These tests guard the input-validation gap surfaced by the
# pkg-validation two-sided conformance sweep: nine distributions accepted a NULL
# silently and four leaked R's internal "argument is missing" message. The
# rejection cases need no torch backend -- validation fires before any tensor is
# built -- so only the construction cases skip when torch is absent.

test_that("every distribution constructor rejects a NULL parameter cleanly", {
  null_callers <- list(
    normal              = function() normal(NULL),
    half_normal         = function() half_normal(NULL),
    half_cauchy         = function() half_cauchy(NULL),
    student_t           = function() student_t(NULL),
    uniform             = function() uniform(NULL),
    bernoulli           = function() bernoulli(NULL),
    binomial_dist       = function() binomial_dist(NULL),
    poisson_dist        = function() poisson_dist(NULL),
    gamma_dist          = function() gamma_dist(NULL),
    beta_dist           = function() beta_dist(NULL),
    exponential         = function() exponential(NULL),
    negative_binomial   = function() negative_binomial(NULL),
    lognormal           = function() lognormal(NULL),
    cauchy              = function() cauchy(NULL),
    multivariate_normal = function() multivariate_normal(NULL),
    dirichlet           = function() dirichlet(NULL),
    lkj_correlation     = function() lkj_correlation(NULL),
    wishart             = function() wishart(NULL)
  )
  for (nm in names(null_callers)) {
    expect_error(null_callers[[nm]](), "must be numeric", info = nm)
  }
})

test_that("a wholly missing required argument is rejected, not passed through", {
  # Previously these surfaced R's opaque "argument \"x\" is missing" message.
  expect_error(bernoulli(), "must be numeric")
  expect_error(beta_dist(), "must be numeric")
  expect_error(binomial_dist(), "must be numeric")
  expect_error(gamma_dist(), "must be numeric")
  expect_error(poisson_dist(), "must be numeric")
  expect_error(negative_binomial(), "must be numeric")
  expect_error(dirichlet(), "must be numeric")
  expect_error(multivariate_normal(), "must be numeric")
  expect_error(wishart(), "must be numeric")
})

test_that("a partially supplied two-parameter distribution still rejects", {
  # beta_dist(NULL) leaves `beta` missing; the message names the offending arg.
  expect_error(beta_dist(NULL), "alpha")
  expect_error(beta_dist(2, NULL), "beta")
  expect_error(gamma_dist(2, NULL), "rate")
  expect_error(binomial_dist(10, NULL), "prob")
})

test_that("a non-numeric parameter is rejected", {
  expect_error(normal("a", 1), "must be numeric")
  expect_error(beta_dist(TRUE, 2), "must be numeric")
  expect_error(poisson_dist(list(1)), "must be numeric")
  expect_error(cauchy(0, factor("x")), "must be numeric")
})

test_that("an NA-bearing numeric parameter is rejected", {
  expect_error(normal(NA_real_, 1), "missing values")
  expect_error(gamma_dist(c(1, NA), 1), "missing values")
})

test_that("valid numeric parameters still construct a gretaR_array", {
  skip_if_not_installed("torch")
  expect_true(inherits(normal(0, 1), "gretaR_array"))
  expect_true(inherits(student_t(3, 0, 1), "gretaR_array"))
  expect_true(inherits(beta_dist(2, 2), "gretaR_array"))
  expect_true(inherits(binomial_dist(10, 0.3), "gretaR_array"))
  expect_true(inherits(poisson_dist(3), "gretaR_array"))
  expect_true(inherits(uniform(0, 1), "gretaR_array"))
  expect_true(inherits(negative_binomial(5, 0.3), "gretaR_array"))
})

test_that("a gretaR_array parameter is accepted (hierarchical priors)", {
  skip_if_not_installed("torch")
  # The Beta-Bernoulli conjugate model: a distribution's parameter is itself a
  # graph node, so a gretaR_array must pass validation untouched.
  theta <- beta_dist(2, 2)
  expect_true(inherits(bernoulli(theta), "gretaR_array"))
  mu <- normal(0, 10)
  expect_true(inherits(normal(mu, 1), "gretaR_array"))
})
