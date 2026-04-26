# Getting Started with gretaR

## Overview

**gretaR** is a Bayesian statistical modelling package for R built on
the `torch` backend. It provides an intuitive, R-native syntax for
defining probabilistic models, which are then compiled to differentiable
torch computations and sampled using Hamiltonian Monte Carlo (HMC) or
the No-U-Turn Sampler (NUTS).

**Key features:**

- Pure R — no Python or reticulate dependency
- GPU-accelerated inference via torch
- Familiar R syntax: standard operators, functions, and assignment
- Output compatible with `posterior`, `bayesplot`, and `loo`

## Installation

``` r
# Install from GitHub (development version)
# remotes::install_github("maxmoldovan/gretaR")

# Ensure torch is installed
torch::install_torch()
```

## Example 1: Estimating a Normal Mean

Suppose we observe data from a normal distribution and want to estimate
the mean and standard deviation.

``` r
library(gretaR)
#> 
#> Attaching package: 'gretaR'
#> The following object is masked from 'package:base':
#> 
#>     %*%

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
#> gretaR model
#>   Free parameters: 2 (2 total elements)
#>   Variables:
#>     mu ~ normal [1 x 1]
#>     sigma ~ half_cauchy [1 x 1]
#>   Likelihood terms: 1
```

``` r
# Draw posterior samples
draws <- mcmc(m, n_samples = 1000, warmup = 1000, chains = 4)

# Summarise
summary(draws)
```

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
#> gretaR model
#>   Free parameters: 3 (3 total elements)
#>   Variables:
#>     alpha ~ normal [1 x 1]
#>     beta ~ normal [1 x 1]
#>     sigma ~ half_cauchy [1 x 1]
#>   Likelihood terms: 1
```

``` r
# Sample
draws <- mcmc(m, n_samples = 1000, warmup = 1000, chains = 4)
summary(draws)

# Visualise traces (requires bayesplot)
# plot(draws, type = "trace")
```

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

| Function                         | Distribution        | Support              |
|----------------------------------|---------------------|----------------------|
| `normal(mean, sd)`               | Normal              | $( - \infty,\infty)$ |
| `half_normal(sd)`                | Half-Normal         | $(0,\infty)$         |
| `half_cauchy(scale)`             | Half-Cauchy         | $(0,\infty)$         |
| `student_t(df, mu, sigma)`       | Student-t           | $( - \infty,\infty)$ |
| `uniform(lower, upper)`          | Uniform             | $(a,b)$              |
| `exponential(rate)`              | Exponential         | $(0,\infty)$         |
| `gamma_dist(shape, rate)`        | Gamma               | $(0,\infty)$         |
| `beta_dist(alpha, beta)`         | Beta                | $(0,1)$              |
| `bernoulli(prob)`                | Bernoulli           | $\{ 0,1\}$           |
| `binomial_dist(size, prob)`      | Binomial            | $\{ 0,\ldots,n\}$    |
| `poisson_dist(rate)`             | Poisson             | $\{ 0,1,2,\ldots\}$  |
| `multivariate_normal(mean, cov)` | Multivariate Normal | ${\mathbb{R}}^{k}$   |

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

- Explore hierarchical models (coming in v0.2)
- Use
  [`bayesplot::mcmc_trace()`](https://mc-stan.org/bayesplot/reference/MCMC-traces.html)
  and
  [`bayesplot::mcmc_dens_overlay()`](https://mc-stan.org/bayesplot/reference/MCMC-distributions.html)
  for diagnostics
- Compare models with
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) on the
  posterior draws
