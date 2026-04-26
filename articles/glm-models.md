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
#> 
#> Attaching package: 'gretaR'
#> The following object is masked from 'package:base':
#> 
#>     %*%

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
#> gretaR model
#>   Free parameters: 3 (3 total elements)
#>   Variables:
#>     alpha ~ normal [1 x 1]
#>     beta ~ normal [1 x 1]
#>     sigma ~ half_cauchy [1 x 1]
#>   Likelihood terms: 1
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
#> gretaR model
#>   Free parameters: 3 (3 total elements)
#>   Variables:
#>     alpha ~ student_t [1 x 1]
#>     beta ~ normal [1 x 1]
#>     sigma ~ exponential [1 x 1]
#>   Likelihood terms: 1
```

## Inference Methods

gretaR offers four inference methods:

| Method   | Function                                                                    | Speed     | Accuracy       | Best for                 |
|----------|-----------------------------------------------------------------------------|-----------|----------------|--------------------------|
| **NUTS** | `mcmc(sampler="nuts")`                                                      | Slow      | Exact          | Final analysis           |
| **HMC**  | `mcmc(sampler="hmc")`                                                       | Slow      | Exact          | Debugging                |
| **ADVI** | [`variational()`](https://max578.github.io/gretaR/reference/variational.md) | Fast      | Approximate    | Quick checks, large data |
| **MAP**  | [`opt()`](https://max578.github.io/gretaR/reference/opt.md)                 | Very fast | Point estimate | Model verification       |

``` r
# MAP (seconds)
map_fit <- opt(m)

# Variational inference (seconds to minutes)
vi_fit <- variational(m, method = "meanfield")

# NUTS (minutes)
mcmc_draws <- mcmc(m, n_samples = 1000, warmup = 1000)
```

## Model Comparison with loo

``` r
# Requires the loo package
library(loo)
# Use posterior draws for LOO-CV
# (full implementation coming in Phase 3)
```
