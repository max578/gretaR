# Complete Guide to gretaR

This vignette demonstrates every major capability of gretaR through
worked examples. Each section is self-contained and can be run
independently.

### Table of Contents

1.  [Installation and Setup](#installation)
2.  [Core DSL: Data, Variables, and Distributions](#core-dsl)
3.  [Simple Models and MCMC](#simple-models)
4.  [All 18 Distributions](#distributions)
5.  [Parameter Transforms](#transforms)
6.  [MAP Estimation and Laplace Approximation](#map-laplace)
7.  [Variational Inference (ADVI)](#advi)
8.  [Hierarchical Models](#hierarchical)
9.  [Formula Interface (GLMs)](#formula-glm)
10. [Random Effects via Formula](#random-effects)
11. [Smooth Terms (mgcv Integration)](#smooth-terms)
12. [Custom Distributions](#custom-dist)
13. [Mixture Models](#mixtures)
14. [Sparse Matrices](#sparse)
15. [Stan Backend](#stan-backend)
16. [The Unified Output Object](#output)
17. [Diagnostics and Plotting](#diagnostics)
18. [Performance Tips](#performance)

## 1. Installation and Setup

``` r
# Install from GitHub
remotes::install_github("max578/gretaR")

# Install the torch backend (one-time, downloads ~60 MB)
torch::install_torch()

# Load
library(gretaR)
```

## 2. Core DSL: Data, Variables, and Distributions

gretaR models are built from three types of objects:

### Data nodes — fixed observations

``` r
library(gretaR)

# Vectors become n x 1 matrices
y <- as_data(c(3.1, 4.2, 5.0, 3.8))
y

# Matrices keep their shape
X <- as_data(matrix(rnorm(12), nrow = 4, ncol = 3))
dim(X)  # 4 x 3

# NA values are rejected with an informative error
try(as_data(c(1, NA, 3)))
# Error: Missing values (NA) detected in data passed to as_data().
# gretaR requires complete data. Preprocess with mice, missRanger, or
# tidyr::drop_na().
```

### Variable nodes — parameters to estimate

``` r
# Unconstrained variable
v <- variable()

# Positive variable (log transform applied automatically)
v_pos <- variable(lower = 0)

# Bounded variable (logit transform)
v_bound <- variable(lower = 0, upper = 1)

# Vector variable
v_vec <- variable(dim = c(5, 1))
```

### Distribution nodes — priors and likelihoods

``` r
# Priors: create variable nodes with distributions
mu <- normal(0, 10)          # mu ~ N(0, 10)
sigma <- half_cauchy(2)      # sigma ~ HalfCauchy(2), constrained > 0
p <- beta_dist(2, 5)         # p ~ Beta(2, 5), constrained to [0, 1]

# Likelihoods: assign distributions to data
y <- as_data(rnorm(50, 3, 1))
distribution(y) <- normal(mu, sigma)

# The model collects everything
m <- model(mu, sigma)
print(m)
```

### Operations — build the computation graph

``` r
reset_gretaR_env()

# Arithmetic
alpha <- normal(0, 10)
beta <- normal(0, 5)
x <- as_data(rnorm(100))

mu <- alpha + beta * x           # element-wise
mu2 <- alpha + beta * x + beta^2 # compound expressions

# Matrix operations
X <- as_data(matrix(rnorm(300), ncol = 3))
beta_vec <- normal(0, 5, dim = c(3, 1))
eta <- X %*% beta_vec            # matrix multiplication

# Math functions
log_mu <- log(exp(alpha))        # log, exp
abs_val <- abs(beta)             # abs
trig <- sin(alpha) + cos(beta)   # trigonometric
```

## 3. Simple Models and MCMC

### Linear regression

``` r
reset_gretaR_env()
set.seed(42)

# Simulate data
n <- 100
x_obs <- rnorm(n)
y_obs <- 2.5 + 1.8 * x_obs + rnorm(n, 0, 0.7)

# Define model
alpha <- normal(0, 10)
beta <- normal(0, 10)
sigma <- half_cauchy(5)

x <- as_data(x_obs)
y <- as_data(y_obs)
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)

m <- model(alpha, beta, sigma)

# Sample with NUTS (default)
fit <- mcmc(m, n_samples = 1000, warmup = 1000, chains = 2)
print(fit)

# Extract point estimates
coef(fit)
# alpha     beta    sigma
# 2.52      1.79     0.71
```

### Accessing results

``` r
# Posterior summary table
summary(fit)

# Raw posterior draws (posterior::draws_array)
fit$draws

# Convergence diagnostics
fit$convergence$max_rhat   # should be < 1.05
fit$convergence$min_ess    # should be > 400
fit$convergence$n_divergences  # should be 0
```

## 4. All 18 Distributions

Every distribution creates a variable node with an appropriate
constraint and transform for unconstrained sampling.

### Continuous distributions

``` r
reset_gretaR_env()

# Unbounded (identity transform)
x1 <- normal(mean = 0, sd = 1)
x2 <- student_t(df = 5, mu = 0, sigma = 1)
x3 <- cauchy(location = 0, scale = 1)

# Positive (log transform)
x4 <- half_normal(sd = 1)
x5 <- half_cauchy(scale = 2)
x6 <- exponential(rate = 1)
x7 <- gamma_dist(shape = 2, rate = 1)
x8 <- lognormal(meanlog = 0, sdlog = 1)

# Unit interval (logit transform)
x9 <- beta_dist(alpha = 2, beta = 5)
x10 <- uniform(lower = 0, upper = 1)

# Bounded interval (scaled logit transform)
x11 <- uniform(lower = -5, upper = 5)

# Multivariate
x12 <- multivariate_normal(
  mean = c(0, 0),
  covariance = matrix(c(1, 0.5, 0.5, 1), 2, 2)
)

# Simplex (Dirichlet)
x13 <- dirichlet(concentration = c(1, 1, 1))

# Correlation matrix (LKJ)
x14 <- lkj_correlation(eta = 2, dim = 3)

# Positive-definite matrix (Wishart)
x15 <- wishart(df = 5, scale_matrix = diag(3))
```

### Discrete distributions (used as likelihoods)

``` r
reset_gretaR_env()

p <- beta_dist(2, 5)
rate <- gamma_dist(2, 1)

# Binary outcomes
y_bin <- as_data(c(1, 0, 1, 1, 0))
distribution(y_bin) <- bernoulli(p)

# Count data
y_count <- as_data(c(3, 5, 2, 7, 4))
distribution(y_count) <- poisson_dist(rate)

# Binomial (n trials)
y_binom <- as_data(c(7, 5, 8, 6))
distribution(y_binom) <- binomial_dist(size = 10, prob = p)

# Overdispersed counts
y_nb <- as_data(c(3, 12, 5, 0, 8))
distribution(y_nb) <- negative_binomial(size = 5, prob = p)
```

## 5. Parameter Transforms

gretaR automatically applies the correct transform so that HMC/NUTS
samples in unconstrained space. You never need to think about this
unless debugging.

``` r
reset_gretaR_env()

# Positive parameter → LogTransform (samples on log scale)
sigma <- half_cauchy(2)  # constraint: lower = 0
# Internally: sigma = exp(theta_free), Jacobian added to log-joint

# Unit interval → LogitTransform
p <- beta_dist(2, 5)    # constraint: lower = 0, upper = 1
# Internally: p = sigmoid(theta_free)

# Bounded → ScaledLogitTransform
u <- uniform(-5, 5)     # constraint: lower = -5, upper = 5
# Internally: u = -5 + 10 * sigmoid(theta_free)

# Unconstrained → IdentityTransform
mu <- normal(0, 10)     # no constraint
# Internally: mu = theta_free directly
```

## 6. MAP Estimation and Laplace Approximation

### MAP — fast point estimates

``` r
reset_gretaR_env()
set.seed(42)
y_obs <- rnorm(200, 5, 2)

mu <- normal(0, 10)
sigma <- half_cauchy(5)
y <- as_data(y_obs)
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)

# MAP via Adam optimiser (~1 second)
fit_map <- opt(m)
print(fit_map)
coef(fit_map)
# mu = 5.01, sigma = 1.98
```

### Laplace — approximate posterior with covariance

``` r
# Laplace approximation (uses MAP + Hessian)
fit_lap <- laplace(m)
fit_lap$par         # posterior means
fit_lap$sd          # posterior standard deviations
fit_lap$covariance  # full posterior covariance matrix
fit_lap$log_evidence  # approximate log marginal likelihood
```

## 7. Variational Inference (ADVI)

``` r
reset_gretaR_env()
set.seed(42)

mu <- normal(0, 10)
sigma <- half_cauchy(5)
y <- as_data(rnorm(200, 5, 2))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)

# Mean-field ADVI (~seconds)
fit_vi <- variational(m, method = "meanfield", max_iter = 3000)
coef(fit_vi)            # posterior means
fit_vi$sd               # posterior SDs (unconstrained)
fit_vi$elbo             # ELBO convergence history
summary(fit_vi)         # full posterior table from VI draws

# Full-rank ADVI (captures correlations)
fit_vi_fr <- variational(m, method = "fullrank", max_iter = 3000)
fit_vi_fr$covariance    # full posterior covariance
```

## 8. Hierarchical Models

The `[` indexing operator enables multi-level models.

### Random intercepts

``` r
reset_gretaR_env()
set.seed(42)

# Simulate grouped data
n_groups <- 8
n_per_group <- 25
n <- n_groups * n_per_group
group_id <- rep(1:n_groups, each = n_per_group)
true_alpha <- rnorm(n_groups, 5, 2)
y_obs <- rnorm(n, true_alpha[group_id], 1)

# Non-centred parameterisation (recommended)
mu <- normal(0, 10)           # grand mean
tau <- half_cauchy(5)         # group-level SD
z_raw <- normal(0, 1, dim = c(n_groups, 1))  # raw effects
sigma <- half_cauchy(5)       # observation noise

# Group-level intercepts
alpha <- mu + tau * z_raw

# Index into group parameters
y <- as_data(y_obs)
distribution(y) <- normal(alpha[group_id], sigma)

m <- model(mu, tau, sigma, z_raw)
fit <- mcmc(m, n_samples = 500, warmup = 500, chains = 2)
print(fit)
```

### Random intercepts + slopes

``` r
reset_gretaR_env()
set.seed(42)

n_groups <- 6; n_per <- 30; n <- n_groups * n_per
gid <- rep(1:n_groups, each = n_per)
x_obs <- rnorm(n)
true_a <- rnorm(n_groups, 3, 1.5)
true_b <- rnorm(n_groups, 1, 0.3)
y_obs <- true_a[gid] + true_b[gid] * x_obs + rnorm(n, 0, 0.5)

# Model
mu_a <- normal(0, 10); mu_b <- normal(0, 5)
tau_a <- half_cauchy(3); tau_b <- half_cauchy(2)
za <- normal(0, 1, dim = c(n_groups, 1))
zb <- normal(0, 1, dim = c(n_groups, 1))
sigma <- half_cauchy(3)

x <- as_data(x_obs); y <- as_data(y_obs)
fitted <- (mu_a + tau_a * za[gid]) + (mu_b + tau_b * zb[gid]) * x
distribution(y) <- normal(fitted, sigma)

m <- model(mu_a, mu_b, tau_a, tau_b, sigma, za, zb)
fit <- opt(m)  # quick MAP check
coef(fit)
```

## 9. Formula Interface (GLMs)

[`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md)
provides a high-level interface using standard R formulas.

### Gaussian linear model

``` r
fit <- gretaR_glm(
  Sepal.Length ~ Sepal.Width + Petal.Length,
  data = iris,
  family = "gaussian",
  sampler = "map"
)
print(fit)
coef(fit)
```

### Logistic regression

``` r
set.seed(42)
dat <- data.frame(
  x1 = rnorm(200), x2 = rnorm(200)
)
dat$y <- rbinom(200, 1, plogis(0.5 + 1.2 * dat$x1 - 0.8 * dat$x2))

fit <- gretaR_glm(y ~ x1 + x2, data = dat, family = "binomial",
                   sampler = "map")
coef(fit)
```

### Poisson regression

``` r
dat <- data.frame(x = rnorm(150))
dat$y <- rpois(150, exp(1 + 0.5 * dat$x))

fit <- gretaR_glm(y ~ x, data = dat, family = "poisson",
                   sampler = "map")
coef(fit)
```

### Custom priors

``` r
fit <- gretaR_glm(
  Sepal.Length ~ Sepal.Width,
  data = iris,
  family = "gaussian",
  prior = list(
    beta = normal(0, 1, dim = c(2, 1)),  # tight prior on coefficients
    sigma = exponential(rate = 1)         # exponential prior on SD
  ),
  sampler = "map"
)
```

## 10. Random Effects via Formula

lme4-style random effects are parsed via regex (lme4 not required).

``` r
set.seed(42)
dat <- data.frame(
  y = rnorm(120),
  x = rnorm(120),
  group = factor(rep(1:6, each = 20))
)
dat$y <- 2 + 0.5 * dat$x + rnorm(6, 0, 1.5)[dat$group] + rnorm(120, 0, 0.5)

# Random intercepts
fit <- gretaR_glm(y ~ x + (1 | group), data = dat, sampler = "map")
print(fit)  # shows "mixed" with group info

# Multiple grouping factors
dat$site <- factor(sample(1:4, 120, replace = TRUE))
fit2 <- gretaR_glm(y ~ x + (1 | group) + (1 | site),
                    data = dat, sampler = "map")
print(fit2)
```

## 11. Smooth Terms (mgcv Integration)

Full mgcv spline syntax is supported via the `smooth2random`
decomposition.

``` r
set.seed(42)
n <- 200
dat <- data.frame(x = runif(n, 0, 2 * pi))
dat$y <- sin(dat$x) + rnorm(n, 0, 0.3)

# Thin plate regression spline
fit <- gretaR_glm(y ~ s(x, k = 10), data = dat, sampler = "map")
coef(fit)

# Cubic regression spline
fit_cr <- gretaR_glm(y ~ s(x, bs = "cr", k = 12), data = dat,
                      sampler = "map")

# Multiple smooths
dat$x2 <- rnorm(n)
dat$y2 <- sin(dat$x) + 0.5 * dat$x2 + rnorm(n, 0, 0.3)
fit_multi <- gretaR_glm(y2 ~ s(x, k = 8) + s(x2, k = 5),
                         data = dat, sampler = "map")

# Smooth + linear + random effects
dat$group <- factor(sample(1:4, n, replace = TRUE))
fit_gam_re <- gretaR_glm(y ~ s(x, k = 8) + (1 | group),
                          data = dat, sampler = "map")
```

#### Supported smooth types

All 21 mgcv basis types work, including:

- `s(x, bs = "tp")` — thin plate regression spline (default)
- `s(x, bs = "cr")` — cubic regression spline
- `s(x, bs = "ps")` — P-spline
- `s(x, bs = "cc")` — cyclic cubic
- `s(x, bs = "re")` — random effect
- `te(x1, x2)` — tensor product
- `ti(x1, x2)` — tensor interaction (ANOVA decomposition)
- `s(x, by = fac)` — factor-by smooth

## 12. Custom Distributions

Define distributions with any torch-differentiable log-probability.

``` r
reset_gretaR_env()

# Laplace distribution (not built in)
x <- custom_distribution(
  log_prob_fn = function(x) {
    -torch::torch_sum(torch::torch_abs(x - 2))
  },
  name = "laplace_at_2"
)

# Truncated normal (positive only)
x_pos <- custom_distribution(
  log_prob_fn = function(x) {
    torch::torch_sum(-0.5 * x^2)
  },
  constraint = list(lower = 0, upper = Inf),
  name = "truncated_normal"
)

# Use in a model
y <- as_data(rnorm(50, 2, 1))
distribution(y) <- normal(x, 1)
m <- model(x)
fit <- opt(m)
coef(fit)  # should be near 2
```

## 13. Mixture Models

Finite mixtures use log-sum-exp marginalisation over discrete
components.

``` r
reset_gretaR_env()
set.seed(42)

# Simulate two-component Gaussian mixture
n <- 200
z <- sample(1:2, n, replace = TRUE, prob = c(0.4, 0.6))
y_obs <- ifelse(z == 1, rnorm(n, -2, 0.5), rnorm(n, 3, 1))

# Model
w <- dirichlet(c(1, 1))               # mixture weights
mu1 <- normal(-5, 5); mu2 <- normal(5, 5)  # component means
sigma1 <- half_cauchy(2); sigma2 <- half_cauchy(2)

mix <- mixture(
  distributions = list(
    normal(mu1, sigma1),
    normal(mu2, sigma2)
  ),
  weights = w
)

y <- as_data(y_obs)
distribution(y) <- mix
m <- model(w, mu1, mu2, sigma1, sigma2)

# MAP for quick check
fit <- opt(m)
coef(fit)
# Should recover mu1 ~ -2, mu2 ~ 3
```

## 14. Sparse Matrices

Large, sparse design matrices are handled efficiently.

``` r
library(Matrix)
reset_gretaR_env()

# Create sparse design matrix (e.g., one-hot encoding)
n <- 1000; p <- 50
i_idx <- seq_len(n)
j_idx <- sample(1:p, n, replace = TRUE)
X_sparse <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, p))

# Use directly in gretaR
X <- as_data(X_sparse)  # stored as torch sparse COO tensor
beta <- normal(0, 1, dim = c(p, 1))
eta <- X %*% beta       # sparse-aware matrix multiplication

y <- as_data(rnorm(n))
distribution(y) <- normal(eta, 1)
m <- model(beta)
fit <- opt(m, verbose = FALSE)
```

## 15. Stan Backend

Use Stan’s compiled C++ sampler for 30-150x faster inference on standard
models.

``` r
reset_gretaR_env()
set.seed(42)

mu <- normal(0, 10)
sigma <- half_cauchy(5)
y <- as_data(rnorm(100, 3, 1.5))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)

# Torch backend (default)
fit_torch <- mcmc(m, n_samples = 500, warmup = 500,
                   chains = 2, backend = "torch")

# Stan backend (requires cmdstanr)
fit_stan <- mcmc(m, n_samples = 500, warmup = 500,
                  chains = 2, backend = "stan")

# Same output structure — same API
coef(fit_torch)
coef(fit_stan)

# Inspect generated Stan code
cat(fit_stan$stan_code)

# Stan MAP
fit_stan_map <- opt(m, backend = "stan")
coef(fit_stan_map)
```

#### When to use which backend

| Scenario             | Backend                                                                 | Why                                         |
|----------------------|-------------------------------------------------------------------------|---------------------------------------------|
| Production inference | `"stan"`                                                                | 30-150x faster for standard models          |
| Custom distributions | `"torch"`                                                               | Stan can’t express arbitrary torch log_prob |
| GPU models           | `"torch"`                                                               | Stan has no GPU support                     |
| Quick prototyping    | `"torch"` + [`opt()`](https://max578.github.io/gretaR/reference/opt.md) | No compilation wait                         |
| Hierarchical models  | `"stan"`                                                                | Massive speedup (100x+)                     |

## 16. The Unified Output Object

All inference functions return a `gretaR_fit` object.

``` r
# Same structure regardless of method
fit <- mcmc(m, backend = "torch")  # or "stan", opt(), variational(), laplace()

fit$draws         # posterior::draws_array (NULL for MAP)
fit$model         # compiled gretaR_model
fit$summary       # posterior summary table
fit$convergence   # list: n_eff, rhat, max_rhat, min_ess, n_divergences
fit$call_info     # original arguments
fit$run_time      # elapsed seconds
fit$method        # "nuts", "hmc", "vi", "map", "laplace", "stan"
fit$par           # point estimates (all methods)

# Consistent S3 methods
print(fit)        # concise summary
summary(fit)      # full posterior table
coef(fit)         # named point estimates
plot(fit, "trace")    # trace plots (requires bayesplot)
plot(fit, "density")  # density overlays
plot(fit, "rhat")     # R-hat diagnostic
plot(fit, "neff")     # ESS ratio plot
```

## 17. Diagnostics and Plotting

``` r
# Check convergence
fit$convergence$max_rhat       # < 1.05 is good
fit$convergence$min_ess        # > 400 is adequate
fit$convergence$n_divergences  # 0 is ideal

# Visual diagnostics (requires bayesplot)
library(bayesplot)
plot(fit, type = "trace")     # should show mixing chains
plot(fit, type = "density")   # posterior density overlays
plot(fit, type = "pairs")     # bivariate posterior
plot(fit, type = "rhat")      # R-hat bar chart

# Use posterior package directly
library(posterior)
summary(fit)  # delegates to summarise_draws()

# Use loo for model comparison
library(loo)
# (future: loo integration coming in next release)
```

## 18. Performance Tips

### Choose the right inference method

``` r
# Fastest to slowest:
# 1. MAP: opt(m)                    ~seconds
# 2. Laplace: laplace(m)            ~seconds
# 3. ADVI: variational(m)           ~seconds to minutes
# 4. MCMC (Stan): mcmc(m, backend="stan")  ~seconds to minutes
# 5. MCMC (torch): mcmc(m, backend="torch") ~minutes to hours

# Recommended workflow:
# 1. Start with MAP to verify the model compiles and estimates are sensible
# 2. Try ADVI for approximate posteriors
# 3. Run MCMC (Stan) for final inference
```

### Non-centred parameterisation for hierarchical models

``` r
# BAD: centred parameterisation (funnel geometry)
# alpha <- normal(mu, tau, dim = c(J, 1))

# GOOD: non-centred (better HMC geometry)
z_raw <- normal(0, 1, dim = c(J, 1))
alpha <- mu + tau * z_raw
```

### Use Stan backend for hierarchical models

``` r
# The Stan backend is 30-150x faster for standard models.
# Always prefer backend = "stan" for production runs on
# models that use standard distributions.
fit <- mcmc(m, backend = "stan", n_samples = 2000, warmup = 1000)
```

### Reset the environment between models

``` r
# Always call reset_gretaR_env() before defining a new model.
# This clears the global DAG and prevents node ID collisions.
reset_gretaR_env()
```
