# Regression tests for the 2026-05-22 correctness audit.
#
# Each `test_that` block maps to one finding (A1..A7). Together these guarantee
# the v0.2.1 release does not regress on the issues flagged in
# audit_2026-05-22.md.

# ---------------------------------------------------------------------------
# A1 — model() variable discovery is reachability-scoped, not global.
# ---------------------------------------------------------------------------

test_that("A1: model() only includes the requested target variable", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  a <- normal(0, 1)
  b <- normal(0, 1)   # unrelated; must NOT be pulled into model(a)

  m <- model(a)
  expect_equal(m$total_dim, 1L)
  expect_true("a" %in% vapply(m$param_info, function(p) p$name, character(1)))
  expect_false("b" %in% vapply(m$param_info, function(p) p$name, character(1)))
})

test_that("A1: independent models built in one session do not leak variables", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  a <- normal(0, 1)
  m1 <- model(a)

  b <- normal(0, 1)
  y <- as_data(rnorm(5))
  distribution(y) <- normal(b, 1)
  m2 <- model(b)

  expect_equal(m1$total_dim, 1L)
  expect_equal(m2$total_dim, 1L)
  # m2 must include b but not a — a is unreachable from (b, y, b's likelihood).
  expect_true("b" %in% vapply(m2$param_info, function(p) p$name, character(1)))
  expect_false("a" %in% vapply(m2$param_info, function(p) p$name, character(1)))
})

test_that("A1: reachability pulls in unnamed auxiliary likelihood parameters", {
  # mu is the explicit target; sigma is reachable via the likelihood RHS and
  # must therefore be included as a free variable (otherwise the joint
  # log-density would be missing a parameter).
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 1)
  sigma <- half_cauchy(1)
  y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, sigma)

  m <- model(mu)
  expect_equal(m$total_dim, 2L)
  param_names <- vapply(m$param_info, function(p) p$name, character(1))
  expect_true("mu" %in% param_names)
})

# ---------------------------------------------------------------------------
# A2 — discrete latent variables are rejected.
# ---------------------------------------------------------------------------

test_that("A2: model() rejects a Bernoulli latent target", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  z <- bernoulli(0.5)
  expect_error(model(z), regexp = "[Dd]iscrete latent")
})

test_that("A2: model() rejects a Poisson latent target", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  k <- poisson_dist(3)
  expect_error(model(k), regexp = "[Dd]iscrete latent")
})

test_that("A2: discrete distributions as likelihood RHS remain legal", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  p <- beta_dist(1, 1)
  y <- as_data(c(0, 1, 1, 0))
  distribution(y) <- bernoulli(p)

  expect_s3_class(model(p), "gretaR_model")
})

# ---------------------------------------------------------------------------
# A3 — Dirichlet / LKJ / Wishart are gated as non-samplable latents.
# ---------------------------------------------------------------------------

test_that("A3: Dirichlet latent target is refused with a helpful error", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  theta <- dirichlet(c(1, 1, 1))
  expect_error(model(theta), regexp = "dirichlet|samp")
})

test_that("A3: LKJ latent target is refused", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  R <- lkj_correlation(eta = 2, dim = 3)
  expect_error(model(R), regexp = "lkj|samp")
})

test_that("A3: Wishart latent target is refused", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  S <- wishart(df = 5, scale_matrix = diag(3))
  expect_error(model(S), regexp = "wishart|samp")
})

# ---------------------------------------------------------------------------
# A4 — sum() / mean() S3 methods are registered and dispatch correctly.
# ---------------------------------------------------------------------------

test_that("A4: sum() dispatches for gretaR_array", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- normal(0, 1, dim = c(3, 1))
  s <- sum(x)
  expect_s3_class(s, "gretaR_array")
  node <- get_node(s)
  expect_equal(node$op_type, "sum")
  expect_equal(dim(s), c(1L, 1L))
})

test_that("A4: mean() dispatches for gretaR_array", {
  skip_if_not_installed("torch")
  reset_gretaR_env()
  x <- normal(0, 1, dim = c(3, 1))
  m <- mean(x)
  expect_s3_class(m, "gretaR_array")
  expect_equal(get_node(m)$op_type, "mean")
})

test_that("A4: NAMESPACE registers sum.gretaR_array and mean.gretaR_array", {
  ns_path <- system.file("NAMESPACE", package = "gretaR")
  if (!nzchar(ns_path)) {
    # Fallback to source tree when running via devtools::load_all
    src <- normalizePath(
      file.path(testthat::test_path(".."), "..", "NAMESPACE"),
      mustWork = FALSE
    )
    if (file.exists(src)) ns_path <- src
  }
  if (!file.exists(ns_path)) skip("NAMESPACE not locatable in this test mode")
  ns <- readLines(ns_path)
  expect_true(any(grepl("S3method\\(sum,gretaR_array\\)", ns, fixed = FALSE)))
  expect_true(any(grepl("S3method\\(mean,gretaR_array\\)", ns, fixed = FALSE)))
})

# ---------------------------------------------------------------------------
# A5 — correlated random slopes give an informative, workaround-pointing error.
# ---------------------------------------------------------------------------

test_that("A5: correlated random slopes raise an informative error", {
  skip_if_not_installed("torch")
  dat <- data.frame(
    y = rnorm(20),
    x = rnorm(20),
    group = rep(letters[1:4], each = 5)
  )

  expect_error(
    gretaR_glm(y ~ x + (x | group), dat, sampler = "map", verbose = FALSE),
    regexp = "not yet supported|0 \\+"
  )
})

# ---------------------------------------------------------------------------
# A6 — truncated distributions return -Inf for out-of-support observations.
# ---------------------------------------------------------------------------

test_that("A6: truncated normal returns -Inf below the lower bound", {
  skip_if_not_installed("torch")
  dist <- NormalDistribution$new(0, 1, truncation = c(0, Inf))
  lp <- dist$log_prob(torch::torch_tensor(-1, dtype = torch::torch_float32()))
  expect_true(is.infinite(lp$item()))
  expect_true(lp$item() < 0)
})

test_that("A6: truncated normal evaluates inside support normally", {
  skip_if_not_installed("torch")
  dist <- NormalDistribution$new(0, 1, truncation = c(0, Inf))
  lp <- dist$log_prob(torch::torch_tensor(1, dtype = torch::torch_float32()))
  expect_true(is.finite(lp$item()))
})

# ---------------------------------------------------------------------------
# A7 — MCMC reports real elapsed time and post-warmup divergences.
# ---------------------------------------------------------------------------

test_that("A7: mcmc()$run_time is measured elapsed seconds, not a placeholder", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 1)
  y <- as_data(rnorm(5))
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- mcmc(m, n_samples = 2L, warmup = 2L, chains = 1L, verbose = FALSE)

  expect_true(is.numeric(fit$run_time))
  expect_true(fit$run_time >= 0)
  # Placeholder would have been raw$n_samples == 2 exactly. Real timing is
  # essentially never an exact integer match.
  expect_false(identical(fit$run_time, fit$call_info$n_samples))
})

test_that("A7: draws carry post-warmup and warmup divergences as separate attrs", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  mu <- normal(0, 1)
  y <- as_data(rnorm(5))
  distribution(y) <- normal(mu, 1)
  m <- model(mu)

  fit <- mcmc(m, n_samples = 3L, warmup = 2L, chains = 1L, verbose = FALSE)

  post <- attr(fit$draws, "divergences")
  warm <- attr(fit$draws, "warmup_divergences")
  expect_equal(nrow(post), 3L)
  expect_equal(nrow(warm), 2L)
  expect_equal(fit$convergence$n_divergences, sum(post, na.rm = TRUE))
})
