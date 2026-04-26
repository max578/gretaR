# Hierarchical Models with gretaR

## Overview

Hierarchical (multi-level) models are a key strength of Bayesian
inference. gretaR supports hierarchical models through the `[` indexing
operator, which lets you define group-level parameters and index them by
observation.

## Random Intercepts Model

Consider data from `J` groups, each with `n_j` observations:

$$y_{ij} = \alpha_{j} + \epsilon_{ij},\quad\epsilon_{ij} \sim N\left( 0,\sigma^{2} \right)$$

$$\alpha_{j} \sim N\left( \mu,\tau^{2} \right)$$

``` r
library(gretaR)
#> 
#> Attaching package: 'gretaR'
#> The following object is masked from 'package:base':
#> 
#>     %*%

# Simulate grouped data
set.seed(42)
n_groups <- 5
n_per_group <- 20
n <- n_groups * n_per_group
group_id <- rep(1:n_groups, each = n_per_group)

true_mu <- 5
true_tau <- 2
true_sigma <- 1
true_alpha <- rnorm(n_groups, true_mu, true_tau)
y_obs <- rnorm(n, true_alpha[group_id], true_sigma)

# Define gretaR model
mu <- normal(0, 10)
tau <- half_cauchy(5)
alpha <- normal(mu, tau, dim = c(n_groups, 1))
sigma <- half_cauchy(5)

y <- as_data(y_obs)
fitted_vals <- alpha[group_id]  # Index into group-level parameters
distribution(y) <- normal(fitted_vals, sigma)

m <- model(mu, tau, sigma, alpha)
print(m)
#> gretaR model
#>   Free parameters: 4 (8 total elements)
#>   Variables:
#>     mu ~ normal [1 x 1]
#>     tau ~ half_cauchy [1 x 1]
#>     sigma ~ half_cauchy [1 x 1]
#>     alpha ~ normal [5 x 1]
#>   Likelihood terms: 1
```

The `alpha[group_id]` syntax creates an operation node in the DAG that
selects the appropriate group-level parameter for each observation. This
is equivalent to the indexing notation used in Stan and JAGS.

``` r
# Fit with NUTS
draws <- mcmc(m, n_samples = 1000, warmup = 1000, chains = 4)
summary(draws)
```

## Random Intercepts and Slopes

Extend to include a group-level slope:

$$y_{ij} = \alpha_{j} + \beta_{j}x_{ij} + \epsilon_{ij}$$

$$\alpha_{j} \sim N\left( \mu_{\alpha},\tau_{\alpha}^{2} \right),\quad\beta_{j} \sim N\left( \mu_{\beta},\tau_{\beta}^{2} \right)$$

``` r
reset_gretaR_env()
set.seed(123)

n_groups <- 8
n_per_group <- 15
n <- n_groups * n_per_group
group_id <- rep(1:n_groups, each = n_per_group)
x_obs <- rnorm(n)

true_mu_a <- 3; true_tau_a <- 1.5
true_mu_b <- 2; true_tau_b <- 0.5
true_sigma <- 0.8

true_a <- rnorm(n_groups, true_mu_a, true_tau_a)
true_b <- rnorm(n_groups, true_mu_b, true_tau_b)
y_obs <- true_a[group_id] + true_b[group_id] * x_obs + rnorm(n, 0, true_sigma)

# Model
mu_a <- normal(0, 10)
mu_b <- normal(0, 10)
tau_a <- half_cauchy(5)
tau_b <- half_cauchy(5)
alpha <- normal(mu_a, tau_a, dim = c(n_groups, 1))
beta <- normal(mu_b, tau_b, dim = c(n_groups, 1))
sigma <- half_cauchy(5)

x <- as_data(x_obs)
y <- as_data(y_obs)

mu_y <- alpha[group_id] + beta[group_id] * x
distribution(y) <- normal(mu_y, sigma)

m <- model(mu_a, mu_b, tau_a, tau_b, sigma, alpha, beta)
print(m)
#> gretaR model
#>   Free parameters: 7 (21 total elements)
#>   Variables:
#>     mu_a ~ normal [1 x 1]
#>     mu_b ~ normal [1 x 1]
#>     tau_a ~ half_cauchy [1 x 1]
#>     tau_b ~ half_cauchy [1 x 1]
#>     sigma ~ half_cauchy [1 x 1]
#>     alpha ~ normal [8 x 1]
#>     beta ~ normal [8 x 1]
#>   Likelihood terms: 1
```

## Using MAP for Quick Estimates

For quick model checking, use
[`opt()`](https://max578.github.io/gretaR/reference/opt.md) to find the
MAP estimate:

``` r
fit <- opt(m, verbose = TRUE)
fit$par
```

## Using Variational Inference

For faster approximate inference:

``` r
fit <- variational(m, method = "meanfield", max_iter = 3000)
fit$mean
fit$sd
summary(fit$draws)
```

## Formula Interface for Simple Models

For standard GLMs, use the formula interface:

``` r
# Simple linear regression via formula
dat <- data.frame(y = y_obs, x = x_obs, group = factor(group_id))
fit <- gretaR_glm(y ~ x, data = dat, family = "gaussian", sampler = "map")
print(fit)
```

## Tips for Hierarchical Models

1.  **Parameterisation**: Use non-centred parameterisation for better
    sampling when data are sparse within groups.
2.  **Priors on variance components**:
    [`half_cauchy()`](https://max578.github.io/gretaR/reference/half_cauchy.md)
    is a sensible default for group-level standard deviations (Gelman
    2006).
3.  **Convergence**: Check R-hat and ESS. Hierarchical models may need
    longer warmup periods.
4.  **Quick checks**: Use
    [`opt()`](https://max578.github.io/gretaR/reference/opt.md) or
    [`vi()`](https://rdrr.io/r/utils/edit.html) to verify the model
    before running full MCMC.
