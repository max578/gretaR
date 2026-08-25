# GLMs with gretaR

## Overview

gretaR supports generalised linear models (GLMs) through two interfaces:

1.  **Formula interface** (`gretaR_glm`): quick specification for
    standard models
2.  **DSL interface**: full control for custom models

## Formula Interface

### Gaussian Linear Regression

``` r

library(gretaR)

# Using the iris dataset
fit <- gretaR_glm(
  Sepal.Length ~ Sepal.Width + Petal.Length,
  data = iris,
  family = "gaussian",
  sampler = "nuts",
  iter = 2000
)
print(fit)
summary(fit$draws)
```

### Logistic Regression

``` r

# Simulate binary outcome
set.seed(42)
dat <- data.frame(
  x1 = rnorm(200),
  x2 = rnorm(200)
)
dat$y <- rbinom(200, 1, plogis(0.5 + 1.2 * dat$x1 - 0.8 * dat$x2))

fit <- gretaR_glm(y ~ x1 + x2, data = dat, family = "binomial")
summary(fit$draws)
```

### Poisson Regression

``` r

dat <- data.frame(x = rnorm(150))
dat$y <- rpois(150, exp(1 + 0.5 * dat$x))

fit <- gretaR_glm(y ~ x, data = dat, family = "poisson")
summary(fit$draws)
```

## DSL Interface

For maximum flexibility, use the gretaR DSL directly.

### Linear Regression (DSL)

``` r

library(gretaR)

set.seed(123)
n <- 100
x_obs <- rnorm(n)
y_obs <- 2 + 3 * x_obs + rnorm(n, 0, 0.5)

reset_gretaR_env()

alpha <- normal(0, 10)
beta <- normal(0, 10)
sigma <- half_cauchy(2)

x <- as_data(x_obs)
y <- as_data(y_obs)
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)

m <- model(alpha, beta, sigma)
print(m)
```

### Custom Priors

``` r

reset_gretaR_env()

# Student-t prior for robust regression
alpha <- student_t(df = 3, mu = 0, sigma = 10)
beta <- normal(0, 5)
sigma <- exponential(rate = 1)

x <- as_data(x_obs)
y <- as_data(y_obs)
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)

m <- model(alpha, beta, sigma)
print(m)
```

## Inference Methods

gretaR offers five inference methods:

| Method | Function | Speed | Accuracy | Best for |
|----|----|----|----|----|
| **NUTS** | `mcmc(sampler="nuts")` | Slow | Asymptotically exact | Final analysis |
| **HMC** | `mcmc(sampler="hmc")` | Slow | Asymptotically exact | Debugging |
| **ADVI** | [`variational()`](https://max578.github.io/gretaR/reference/variational.md) | Fast | Approximate | Quick checks, large data |
| **MAP** | [`opt()`](https://max578.github.io/gretaR/reference/opt.md) | Very fast | Point estimate | Model verification |
| **Laplace** | [`laplace()`](https://max578.github.io/gretaR/reference/laplace.md) | Very fast | Gaussian approximation | Quick uncertainty around a mode |

``` r

# MAP (seconds)
map_fit <- opt(m)
coef(map_fit)

# Laplace approximation around the MAP (seconds)
laplace_fit <- laplace(m)

# Variational inference (seconds)
vi_fit <- variational(m, method = "meanfield")

# NUTS (seconds on this small model)
mcmc_fit <- mcmc(m, n_samples = 500, warmup = 500, chains = 2)
```

``` r

plot(mcmc_fit, type = "density")
```

NUTS gives the reference posterior here: MAP and the Laplace
approximation report a point estimate and a local Gaussian curvature
around it, and ADVI reports a factorised approximate posterior, so the
three faster methods are used to sanity-check a model quickly before
committing to the full NUTS run.

## Model Comparison with loo

``` r

# Requires the loo package
library(loo)

# loo::loo() takes a pointwise log-likelihood matrix, which gretaR does
# not currently compute internally -- it must be supplied by the caller
# from the fitted model's likelihood terms. There is no gretaR-native
# LOO-CV verb yet.
```
