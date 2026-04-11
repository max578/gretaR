# Integration tests — end-to-end model fitting
# These tests run actual sampling and take longer

test_that("HMC recovers known parameters (normal model)", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  true_mu <- 3
  true_sigma <- 1.5
  y_obs <- rnorm(100, true_mu, true_sigma)

  mu <- normal(0, 10)
  sigma <- half_cauchy(5)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, sigma)

  m <- model(mu, sigma)
  draws <- mcmc(m, n_samples = 500, warmup = 500, chains = 2,
                sampler = "hmc", n_leapfrog = 20, verbose = FALSE)

  expect_s3_class(draws, "gretaR_fit")

  # Check posterior means are in the right ballpark
  summ <- draws$summary
  mu_mean <- summ$mean[summ$variable == "mu"]
  sigma_mean <- summ$mean[summ$variable == "sigma"]

  expect_true(abs(mu_mean - true_mu) < 1,
              label = sprintf("mu posterior mean %f not near true %f", mu_mean, true_mu))
  expect_true(abs(sigma_mean - true_sigma) < 1,
              label = sprintf("sigma posterior mean %f not near true %f", sigma_mean, true_sigma))
})

test_that("NUTS recovers known parameters (normal model)", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(42)
  torch::torch_manual_seed(42)
  true_mu <- 5
  y_obs <- rnorm(50, true_mu, 1)

  mu <- normal(0, 10)
  y <- as_data(y_obs)
  distribution(y) <- normal(mu, 1)

  m <- model(mu)
  draws <- mcmc(m, n_samples = 1000, warmup = 1000, chains = 2,
                sampler = "nuts", verbose = FALSE)

  expect_s3_class(draws, "gretaR_fit")

  summ <- draws$summary
  mu_mean <- summ$mean[summ$variable == "mu"]
  expect_true(abs(mu_mean - true_mu) < 2)
})

test_that("Linear regression model works", {
  skip_if_not_installed("torch")
  skip_on_cran()
  reset_gretaR_env()

  set.seed(123)
  n <- 50
  x_obs <- rnorm(n)
  y_obs <- 2 + 3 * x_obs + rnorm(n, 0, 0.5)

  alpha <- normal(0, 10)
  beta <- normal(0, 10)
  sigma <- half_cauchy(2)

  x <- as_data(x_obs)
  y <- as_data(y_obs)
  mu <- alpha + beta * x

  distribution(y) <- normal(mu, sigma)

  m <- model(alpha, beta, sigma)
  draws <- mcmc(m, n_samples = 500, warmup = 500, chains = 2,
                sampler = "hmc", n_leapfrog = 20, verbose = FALSE)

  summ <- draws$summary

  alpha_mean <- summ$mean[summ$variable == "alpha"]
  beta_mean <- summ$mean[summ$variable == "beta"]

  # Should recover intercept ~ 2 and slope ~ 3
  expect_true(abs(alpha_mean - 2) < 1.5)
  expect_true(abs(beta_mean - 3) < 1.5)
})
