# Phase 1: model_from_arrays() — the re-entrant, deparse-free builder. It must
# (a) produce a model byte-equivalent to model() on the same DAG, (b) take
# names explicitly (the canonical-name hook), (c) scope the likelihood so
# independent models can be built in one session without cross-contamination,
# and (d) be safe under do.call()/programmatic construction where model()'s
# deparse breaks.

test_that("model_from_arrays matches model() in structure and log-density", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  y <- as_data(rnorm(40, 3, 2))
  distribution(y) <- normal(mu, sigma)

  m_ref <- model(mu, sigma)
  m_arr <- model_from_arrays(
    targets = list(mu, sigma),
    likelihood = y,
    names = c("mu", "sigma")
  )

  # Structural parity.
  expect_s3_class(m_arr, "gretaR_model")
  expect_identical(m_arr$total_dim, m_ref$total_dim)
  expect_identical(m_arr$var_order, m_ref$var_order)
  expect_identical(m_arr$target_names, m_ref$target_names)
  expect_identical(
    vapply(m_arr$param_info, `[[`, character(1), "name"),
    vapply(m_ref$param_info, `[[`, character(1), "name")
  )

  # Log-density parity at a shared point in unconstrained space.
  theta <- torch::torch_randn(m_ref$total_dim, dtype = m_ref$dtype)
  lp_ref <- as.numeric(joint_density(m_ref)(theta$clone())$item())
  lp_arr <- as.numeric(joint_density(m_arr)(theta$clone())$item())
  expect_equal(lp_arr, lp_ref, tolerance = 1e-5)
})

test_that("names are taken explicitly, not deparsed", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  a <- normal(0, 1)
  b <- normal(0, 1)
  y <- as_data(rnorm(20))
  distribution(y) <- normal(a + b, 1)

  m <- model_from_arrays(
    targets = list(a, b),
    likelihood = y,
    names = c("intercept", "slope")
  )
  nm <- vapply(m$param_info, `[[`, character(1), "name")
  expect_setequal(nm, c("intercept", "slope"))

  # A named targets list supplies names when `names` is omitted.
  reset_gretaR_env()
  a <- normal(0, 1)
  y <- as_data(rnorm(20))
  distribution(y) <- normal(a, 1)
  m2 <- model_from_arrays(targets = list(theta = a), likelihood = y)
  expect_equal(m2$param_info[[1]]$name, "theta")
})

test_that("likelihood is scoped — independent models do not leak (re-entrancy)", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()

  # Model A: a Gaussian with its own mu/sigma and data yA.
  mu <- normal(0, 10)
  sigma <- half_cauchy(1)
  yA <- as_data(rnorm(30, 3, 2))
  distribution(yA) <- normal(mu, sigma)
  mA <- model_from_arrays(list(mu, sigma), likelihood = yA, names = c("mu", "sigma"))

  # Model B, built in the SAME session without a reset: a separate parameter
  # theta with its own data zB.
  theta <- normal(0, 1)
  zB <- as_data(rnorm(25))
  distribution(zB) <- normal(theta, 1)
  mB <- model_from_arrays(list(theta), likelihood = zB, names = "theta")

  # B must see only theta — not mu/sigma — and only zB's likelihood.
  expect_equal(length(mB$var_order), 1L)
  expect_equal(unname(vapply(mB$param_info, `[[`, character(1), "name")), "theta")
  expect_equal(names(mB$likelihood_terms), get_node(zB)$id)

  # Contrast: the global-likelihood path (model()'s behaviour, reproduced by
  # passing likelihood = NULL) DOES pull in both likelihoods, hence mu + sigma.
  m_global <- model_from_arrays(list(theta), likelihood = NULL, names = "theta")
  expect_gt(length(m_global$var_order), length(mB$var_order))
})

test_that("model_from_arrays is safe under do.call() where model() deparse breaks", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  mu <- normal(0, 10)
  y <- as_data(rnorm(30, 2))
  distribution(y) <- normal(mu, 1)

  args <- list(targets = list(mu), likelihood = y, names = "mu")
  m <- do.call(model_from_arrays, args)
  expect_equal(m$param_info[[1]]$name, "mu")
  expect_equal(m$total_dim, 1L)
})

test_that("a single gretaR_array target is accepted without wrapping in a list", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  mu <- normal(0, 5)
  y <- as_data(rnorm(20, 1))
  distribution(y) <- normal(mu, 1)

  m <- model_from_arrays(mu, likelihood = y, names = "mu")
  expect_equal(m$total_dim, 1L)
  expect_equal(m$param_info[[1]]$name, "mu")
})

test_that("a model_from_arrays model samples and recovers the posterior", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(11)
  torch::torch_manual_seed(11)
  reset_gretaR_env()
  n <- 40L
  sig <- 1.5
  yv <- rnorm(n, 2, sig)
  pv <- 1 / (1 / 100 + n / sig^2)
  pm <- pv * sum(yv) / sig^2

  mu <- normal(0, 10)
  y <- as_data(yv)
  distribution(y) <- normal(mu, sig)
  m <- model_from_arrays(mu, likelihood = y, names = "mu")

  fit <- mcmc(m, n_samples = 1000L, warmup = 800L, chains = 4L,
              seed = 11L, verbose = FALSE)
  s <- posterior::summarise_draws(fit$draws)
  s <- s[s$variable == "mu", ]
  expect_lt(abs(s$mean - pm), 0.2 * sqrt(pv))
  expect_lt(s$rhat, 1.05)
})

test_that("invalid inputs are rejected with actionable errors", {
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  mu <- normal(0, 1)
  y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, 1)

  # Empty / non-list targets.
  expect_error(model_from_arrays(list()), "non-empty list")
  expect_error(model_from_arrays(42), "non-empty list|gretaR_array")

  # Name count mismatch.
  expect_error(
    model_from_arrays(list(mu), likelihood = y, names = c("a", "b")),
    "one name per target"
  )

  # A data node with no attached likelihood.
  z <- as_data(rnorm(10))
  expect_error(
    model_from_arrays(list(mu), likelihood = z, names = "mu"),
    "No likelihood is attached"
  )

  # A non-variable target (data node) is refused by the shared core.
  expect_error(
    model_from_arrays(list(y), likelihood = y, names = "y"),
    "not a variable node"
  )
})
