# Regression test for the multinomial-NUTS fix (GB1) + the metric/U-turn
# fixes (GS3/GS4). The previous count-based selection sampled uniformly along
# the trajectory and did NOT target the posterior (it produced rhat ~1.5 and a
# wrong posterior SD on a trivial model). Here we check NUTS against a
# conjugate-Normal model whose posterior is known in closed form, asserting
# recovery of BOTH the posterior mean and SD (not just the mean), plus
# convergence. Skipped on CRAN (fits a sampler; runs locally / in CI).

test_that("multinomial NUTS recovers a conjugate-Normal posterior (mean + sd)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  reset_gretaR_env()
  set.seed(42)
  torch::torch_manual_seed(42)

  # Data: y ~ N(mu, sigma^2) with sigma KNOWN; conjugate prior mu ~ N(0, tau).
  n <- 8L
  sigma_known <- 2
  yv <- rnorm(n, mean = 3, sd = sigma_known)

  prior_var <- 100 # tau^2 (prior sd = 10)
  post_var <- 1 / (1 / prior_var + n / sigma_known^2)
  post_mean <- post_var * (sum(yv) / sigma_known^2)
  post_sd <- sqrt(post_var)

  y <- as_data(yv)
  mu <- normal(0, 10)
  distribution(y) <- normal(mu, sigma_known)

  fit <- mcmc(
    model(mu),
    n_samples = 1000L, warmup = 800L, chains = 2L,
    seed = 42L, verbose = FALSE
  )

  s <- posterior::summarise_draws(fit$draws)
  r <- s[s$variable == "mu", ]

  # Posterior mean within 0.2 posterior-SD (generous vs MC error; the broken
  # sampler missed this) and posterior SD within 12% (the broken sampler got
  # the SD badly wrong); rhat confirms convergence.
  expect_lt(abs(r$mean - post_mean), 0.2 * post_sd)
  expect_lt(abs(r$sd - post_sd) / post_sd, 0.12)
  expect_lt(r$rhat, 1.05)
})
