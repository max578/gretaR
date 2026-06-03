# Tests for random_effect() — partially-centred random-effect blocks.
#
# The load-bearing property is posterior invariance: the centring weight
# changes only the sampling geometry, never the target, so every weight must
# recover the same posterior on the shared top-level coordinates and the same
# known truth. The fast tests below cover construction and input validation;
# the sampling tests cover the invariance and recovery claims.

# =============================================================================
# Construction + validation (no sampling)
# =============================================================================

test_that("random_effect builds latent (J) and effect (n) blocks", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  n <- 40L
  J <- 5L
  g <- rep(seq_len(J), length.out = n)
  tau <- half_normal(2)

  re <- random_effect(g, n_groups = J, sd = tau, centring = 0.5)

  expect_s3_class(re, "gretaR_random_effect")
  expect_equal(dim(re$latent), c(J, 1L))
  expect_equal(dim(re$effect), c(n, 1L))
  expect_equal(re$n_groups, J)
  expect_length(re$centring, J)
  expect_error(print(re), NA)
})

test_that("centring resolves: scalar recycles, vector passes through", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  J <- 4L
  tau <- half_normal(2)
  g <- rep(seq_len(J), length.out = 12L)

  re_scalar <- random_effect(g, n_groups = J, sd = tau, centring = 0.3)
  expect_equal(re_scalar$centring, rep(0.3, J))

  wv <- c(0, 0.4, 0.8, 1)
  re_vec <- random_effect(g, n_groups = J, sd = tau, centring = wv)
  expect_equal(re_vec$centring, wv)
})

test_that("random_effect rejects malformed inputs", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  J <- 3L
  tau <- half_normal(2)
  g <- rep(seq_len(J), length.out = 9L)

  # sd must be a gretaR_array, not a number
  expect_error(random_effect(g, J, sd = 2), "gretaR_array")
  # centring out of [0, 1]
  expect_error(random_effect(g, J, sd = tau, centring = 1.2), "\\[0, 1\\]")
  expect_error(random_effect(g, J, sd = tau, centring = -0.1), "\\[0, 1\\]")
  # centring wrong length
  expect_error(random_effect(g, J, sd = tau, centring = c(0.2, 0.8)), "length 1 or 3")
  # group ids out of range
  expect_error(random_effect(c(1L, 2L, 4L), J, sd = tau), "1 to 3")
  # bad n_groups
  expect_error(random_effect(g, n_groups = 2.5, sd = tau), "positive integer")
})

test_that("endpoint algebra matches the manual parameterisations structurally", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  # w = 0 (non-centred) and w = 1 (centred) must both yield a J-vector latent
  # and an n-vector effect; the prior scale differs but the shapes do not.
  J <- 6L
  g <- rep(seq_len(J), length.out = 30L)
  tau <- half_normal(2)

  re0 <- random_effect(g, J, sd = tau, centring = 0)
  re1 <- random_effect(g, J, sd = tau, centring = 1)

  expect_equal(dim(re0$latent), c(J, 1L))
  expect_equal(dim(re1$latent), c(J, 1L))
  expect_true(is.finite(
    get_node(re0$effect)$compute()$sum()$item()
  ))
})

# =============================================================================
# Posterior invariance + recovery (sampling)
# =============================================================================

# A small grouped data set reused across the sampling tests.
.re_test_data <- function(n = 240L, J = 6L, seed = 123L) {
  set.seed(seed)
  x <- rnorm(n)
  g <- rep(seq_len(J), length.out = n)
  u_true <- rnorm(J, 0, 0.7)
  yv <- 1.2 - 0.6 * x + u_true[g] + rnorm(n)
  list(x = x, g = g, yv = yv, J = J, n = n)
}

.fit_re <- function(d, w, seed = 7L) {
  reset_gretaR_env()
  b0 <- normal(0, 5)
  b1 <- normal(0, 5)
  s <- half_normal(5)
  tau <- half_normal(2)
  re <- random_effect(d$g, n_groups = d$J, sd = tau, centring = w)
  eta <- b0 + b1 * as_data(d$x) + re$effect
  yd <- as_data(d$yv)
  distribution(yd) <- normal(eta, s)
  m <- model_from_arrays(
    list(b0, b1, s, tau, re$latent),
    likelihood = yd,
    names = list("b0", "b1", "s", "tau", paste0("u[", seq_len(d$J), "]"))
  )
  fit <- mcmc(m,
    n_samples = 350L, warmup = 350L, chains = 2L,
    sampler = "nuts", seed = seed, verbose = FALSE
  )
  s_tab <- posterior::summarise_draws(fit$draws)
  shared <- s_tab[s_tab$variable %in% c("b0", "b1", "s", "tau"), ]
  stats::setNames(shared$mean, shared$variable)
}

test_that("posterior is invariant to the centring weight (same target)", {
  skip_if_not_installed("torch")
  skip_on_cran()

  d <- .re_test_data()
  m_nc <- .fit_re(d, w = 0) # non-centred
  m_pc <- .fit_re(d, w = 0.9) # near-centred
  m_cp <- .fit_re(d, w = 1) # centred

  # The three parameterisations target the identical posterior, so the
  # shared-coordinate means agree up to Monte Carlo error.
  for (v in c("b0", "b1", "s", "tau")) {
    expect_equal(unname(m_nc[v]), unname(m_pc[v]), tolerance = 0.2)
    expect_equal(unname(m_nc[v]), unname(m_cp[v]), tolerance = 0.25)
  }
})

test_that("random_effect recovers known fixed effects and group scale", {
  skip_if_not_installed("torch")
  skip_on_cran()

  d <- .re_test_data()
  est <- .fit_re(d, w = 0.7, seed = 11L)

  expect_equal(unname(est["b0"]), 1.2, tolerance = 0.3)
  expect_equal(unname(est["b1"]), -0.6, tolerance = 0.2)
  # group scale near the simulated 0.7 (wide tolerance: J = 6 is a small sample
  # of groups, so tau is weakly identified).
  expect_gt(unname(est["tau"]), 0.2)
  expect_lt(unname(est["tau"]), 1.6)
})
