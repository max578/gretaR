# Sampler oracle + reproducibility coverage (closes GS10 / GS8 from the
# 2026-06-01 R99). The recovery tests assert the posterior SD and R-hat
# against a closed-form conjugate-Normal posterior, not just the mean, so a
# regression in sampler correctness is caught -- not silently absorbed. The
# NUTS counterpart lives in test-nuts-multinomial-recovery.R; this adds the
# HMC path and a seed-reproducibility lock. Skipped on CRAN (fits a sampler);
# runs locally / in CI.

# Closed-form posterior for y ~ N(mu, sigma^2) with sigma known and a conjugate
# prior mu ~ N(0, tau).
.conjugate_normal <- function(yv, sigma_known, prior_sd) {
  prior_var <- prior_sd^2
  n <- length(yv)
  post_var <- 1 / (1 / prior_var + n / sigma_known^2)
  list(
    mean = post_var * (sum(yv) / sigma_known^2),
    sd = sqrt(post_var)
  )
}

test_that("HMC recovers a conjugate-Normal posterior (mean + sd + rhat) [GS10]", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  # HB1 (fixed 2026-06-01): fixed-length HMC did not converge here (rhat ~1.5,
  # ess ~10-50) because the trajectory length resonated with the target's
  # periodic flow and dual averaging drove the step size above the leapfrog
  # stability limit. Fixed via accept_stat-based step adaptation + integration-
  # time trajectories (T ~ U(0, 2*pi], L = round(T/eps)). This oracle now
  # guards the recovery; do NOT loosen the assertions.

  reset_gretaR_env()
  set.seed(42)
  torch::torch_manual_seed(42)

  n <- 8L
  sigma_known <- 2
  yv <- rnorm(n, mean = 3, sd = sigma_known)
  post <- .conjugate_normal(yv, sigma_known, prior_sd = 10)

  y <- as_data(yv)
  mu <- normal(0, 10)
  distribution(y) <- normal(mu, sigma_known)

  fit <- mcmc(
    model(mu),
    n_samples = 1000L, warmup = 800L, chains = 2L,
    sampler = "hmc", seed = 42L, verbose = FALSE
  )

  s <- posterior::summarise_draws(fit$draws)
  r <- s[s$variable == "mu", ]

  # Mean within 0.2 posterior-SD; SD within 15% (HMC carries a touch more MC
  # error than NUTS at this budget); rhat confirms convergence.
  expect_lt(abs(r$mean - post$mean), 0.2 * post$sd)
  expect_lt(abs(r$sd - post$sd) / post$sd, 0.15)
  expect_lt(r$rhat, 1.05)
})

test_that("mcmc() is reproducible given seed= [GS8/GS10]", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  run_once <- function() {
    reset_gretaR_env()
    y <- as_data(c(1.0, 2.0, 1.5, 0.5, 2.5))
    mu <- normal(0, 5)
    distribution(y) <- normal(mu, 1)
    mcmc(
      model(mu),
      n_samples = 200L, warmup = 200L, chains = 1L,
      sampler = "nuts", seed = 123L, verbose = FALSE
    )
  }

  d1 <- as.numeric(run_once()$draws)
  d2 <- as.numeric(run_once()$draws)

  # Same seed -> identical draws. This is what actually makes MCMC
  # reproducible (the R RNG, set by seed=), independent of the torch seed.
  expect_equal(length(d1), length(d2))
  expect_equal(d1, d2)
})
