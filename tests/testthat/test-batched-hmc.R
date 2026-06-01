# M2: the batched integration-time HMC sampler must recover a conjugate-Normal
# posterior (mean + sd + rhat, not just the mean) and agree with the verified
# single-chain HMC. Build-beside; the single-chain samplers are unchanged.

.draws_of <- function(raw) {
  s <- raw$samples
  dimnames(s) <- list(NULL, NULL, raw$param_names)
  posterior::as_draws_array(s)
}

test_that("batched HMC recovers a conjugate-Normal posterior (mean + sd + rhat)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(42)
  torch::torch_manual_seed(42)
  n <- 8L
  sig <- 2
  yv <- rnorm(n, 3, sig)
  pv <- 1 / (1 / 100 + n / sig^2)
  pm <- pv * sum(yv) / sig^2
  psd <- sqrt(pv)

  reset_gretaR_env()
  y <- as_data(yv)
  mu <- normal(0, 10)
  distribution(y) <- normal(mu, sig)
  raw <- batched_hmc_sampler(model(mu), n_samples = 1500L, warmup = 1000L,
                             chains = 4L, seed = 42L)
  r <- posterior::summarise_draws(.draws_of(raw))
  r <- r[r$variable == "mu", ]

  expect_lt(abs(r$mean - pm), 0.2 * psd)
  expect_lt(abs(r$sd - psd) / psd, 0.12)
  expect_lt(r$rhat, 1.05)
})

test_that("mcmc(batched = TRUE) routes to the batched sampler and returns a fit", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(42)
  torch::torch_manual_seed(42)
  reset_gretaR_env()
  yv <- rnorm(8, 3, 2)
  y <- as_data(yv)
  mu <- normal(0, 10)
  distribution(y) <- normal(mu, 2)
  fit <- mcmc(model(mu), n_samples = 1000L, warmup = 800L, chains = 4L,
              sampler = "hmc", batched = TRUE, seed = 42L, verbose = FALSE)
  expect_s3_class(fit, "gretaR_fit")
  s <- posterior::summarise_draws(fit$draws)
  expect_lt(s[s$variable == "mu", ]$rhat, 1.05)

  # batched NUTS is not supported -> clear error
  expect_error(
    mcmc(model(mu), sampler = "nuts", batched = TRUE, verbose = FALSE),
    "only.*hmc"
  )
})

test_that("batched HMC posterior agrees with single-chain HMC (A/B)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(7)
  torch::torch_manual_seed(7)
  nn <- 50
  xx <- rnorm(nn)
  yy <- 1.5 - 0.8 * xx + rnorm(nn)
  build <- function() {
    reset_gretaR_env()
    X <- as_data(cbind(1, xx))
    b <- normal(0, 5, dim = 2)
    s <- half_cauchy(2)
    eta <- X %*% b
    yd <- as_data(yy)
    distribution(yd) <- normal(eta, s)
    model(b, s)
  }
  rb <- batched_hmc_sampler(build(), n_samples = 1200L, warmup = 1000L, chains = 4L, seed = 7L)
  sb <- posterior::summarise_draws(.draws_of(rb))
  fs <- mcmc(build(), n_samples = 1200L, warmup = 1000L, chains = 4L,
             sampler = "hmc", seed = 7L, verbose = FALSE)
  ss <- posterior::summarise_draws(fs$draws)

  for (v in c("b[1]", "b[2]", "s")) {
    a <- sb[sb$variable == v, ]
    b2 <- ss[ss$variable == v, ]
    expect_lt(abs(a$mean - b2$mean), 0.1, label = paste0("dmean[", v, "]"))
    expect_lt(abs(a$sd - b2$sd) / b2$sd, 0.2, label = paste0("rel-dsd[", v, "]"))
  }
})
