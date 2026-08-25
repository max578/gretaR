# Getting Started with gretaR

## Overview

**gretaR** is a Bayesian statistical modelling package for R built on
the `torch` backend. It provides an intuitive, R-native syntax for
defining probabilistic models, which are then compiled to differentiable
torch computations and sampled using Hamiltonian Monte Carlo (HMC) or
the No-U-Turn Sampler (NUTS).

**Key features:**

- Pure R – no Python or reticulate dependency
- Native torch (libtorch) backend – CPU today, GPU/MPS on the roadmap
- Familiar R syntax: standard operators, functions, and assignment
- Output compatible with `posterior`, `bayesplot`, and `loo`

## Installation

``` r

# Install from GitHub (development version)
# remotes::install_github("max578/gretaR")

# Ensure torch is installed
torch::install_torch()
```

## Example 1: Estimating a Normal Mean

Suppose we observe data from a normal distribution and want to estimate
the mean and standard deviation.

``` r

library(gretaR)

# Simulate data
set.seed(42)
y_obs <- rnorm(100, mean = 3, sd = 1.5)

# Define priors
mu <- normal(0, 10)
sigma <- half_cauchy(5)

# Wrap observed data
y <- as_data(y_obs)

# Define likelihood
distribution(y) <- normal(mu, sigma)

# Compile model
m <- model(mu, sigma)
print(m)
```

``` r

# Draw posterior samples
fit1 <- mcmc(m, n_samples = 500, warmup = 500, chains = 2)

# Summarise
summary(fit1)
```

``` r

plot(fit1, type = "trace")
```

The two chains mix over the same range of `mu` and `sigma` from the
first sampled draw, with no visible trend or one chain sitting apart
from the other – consistent with the R-hat close to 1 and the effective
sample size reported by `summary(fit1)`.

## Example 2: Bayesian Linear Regression

``` r

# Simulate data
set.seed(123)
n <- 100
x_obs <- rnorm(n)
y_obs <- 2 + 3 * x_obs + rnorm(n, 0, 0.5)

# Reset model state for new model
reset_gretaR_env()

# Priors
alpha <- normal(0, 10)
beta <- normal(0, 10)
sigma <- half_cauchy(2)

# Data
x <- as_data(x_obs)
y <- as_data(y_obs)

# Linear predictor
mu <- alpha + beta * x

# Likelihood
distribution(y) <- normal(mu, sigma)

# Compile
m <- model(alpha, beta, sigma)
print(m)
```

``` r

# Sample
fit2 <- mcmc(m, n_samples = 500, warmup = 500, chains = 2)
summary(fit2)
```

``` r

plot(fit2, type = "density")
```

The three densities are unimodal, chain-to-chain agreement is close, and
the posterior mass for `alpha`, `beta` and `sigma` brackets the values
used to simulate `y_obs`, confirming the model recovers the generating
parameters on this synthetic dataset.

## Workflow Summary

1.  **Wrap data** with
    [`as_data()`](https://max578.github.io/gretaR/reference/as_data.md)
2.  **Define priors** using distribution functions
    ([`normal()`](https://max578.github.io/gretaR/reference/normal.md),
    [`half_cauchy()`](https://max578.github.io/gretaR/reference/half_cauchy.md),
    etc.)
3.  **Build the model** using standard R operations (`+`, `*`,
    [`log()`](https://rdrr.io/r/base/Log.html), etc.)
4.  **Assign the likelihood** with `distribution(y) <- ...`
5.  **Compile** with
    [`model()`](https://max578.github.io/gretaR/reference/model.md)
6.  **Sample** with
    [`mcmc()`](https://max578.github.io/gretaR/reference/mcmc.md)
7.  **Analyse** using `posterior`, `bayesplot`, or `loo`

## Available Distributions

| Function | Distribution | Support |
|----|----|----|
| `normal(mean, sd)` | Normal | $`(-\infty, \infty)`$ |
| `half_normal(sd)` | Half-Normal | $`(0, \infty)`$ |
| `half_cauchy(scale)` | Half-Cauchy | $`(0, \infty)`$ |
| `student_t(df, mu, sigma)` | Student-t | $`(-\infty, \infty)`$ |
| `uniform(lower, upper)` | Uniform | $`(a, b)`$ |
| `exponential(rate)` | Exponential | $`(0, \infty)`$ |
| `gamma_dist(shape, rate)` | Gamma | $`(0, \infty)`$ |
| `beta_dist(alpha, beta)` | Beta | $`(0, 1)`$ |
| `bernoulli(prob)` | Bernoulli | $`\{0, 1\}`$ |
| `binomial_dist(size, prob)` | Binomial | $`\{0, \ldots, n\}`$ |
| `poisson_dist(rate)` | Poisson | $`\{0, 1, 2, \ldots\}`$ |
| `multivariate_normal(mean, cov)` | Multivariate Normal | $`\mathbb{R}^k`$ |

## Choosing a Sampler

- **NUTS** (default): Adaptive, no tuning of leapfrog steps needed.
  Recommended for most models.
- **HMC**: Fixed number of leapfrog steps. Useful for debugging or when
  NUTS tree depth is a concern.

``` r

# NUTS (default)
draws <- mcmc(m, sampler = "nuts")

# HMC with 25 leapfrog steps
draws <- mcmc(m, sampler = "hmc", n_leapfrog = 25)
```

## Next Steps

- Explore hierarchical models in
  [`vignette("hierarchical-models", package = "gretaR")`](https://max578.github.io/gretaR/articles/hierarchical-models.md)
- Use
  [`bayesplot::mcmc_trace()`](https://mc-stan.org/bayesplot/reference/MCMC-traces.html)
  and
  [`bayesplot::mcmc_dens_overlay()`](https://mc-stan.org/bayesplot/reference/MCMC-distributions.html)
  for diagnostics
- Compare models with
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) on the
  posterior draws
