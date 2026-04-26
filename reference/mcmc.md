# Draw MCMC Samples from a gretaR Model

Run Hamiltonian Monte Carlo or the No-U-Turn Sampler on a compiled
`gretaR_model` and return posterior draws in a format compatible with
the `posterior` and `bayesplot` packages.

## Usage

``` r
mcmc(
  model,
  n_samples = 1000L,
  warmup = 1000L,
  chains = 4L,
  sampler = c("nuts", "hmc"),
  backend = c("torch", "stan"),
  step_size = NULL,
  max_treedepth = 10L,
  n_leapfrog = 25L,
  target_accept = NULL,
  init_values = NULL,
  seed = NULL,
  verbose = TRUE
)
```

## Arguments

- model:

  A `gretaR_model` object created by
  [`model`](https://max578.github.io/gretaR/reference/model.md).

- n_samples:

  Number of post-warmup samples per chain (default 1000).

- warmup:

  Number of warmup (adaptation) iterations per chain (default 1000).

- chains:

  Number of independent chains (default 4).

- sampler:

  Sampler to use: `"nuts"` (default) or `"hmc"`.

- backend:

  Inference backend: `"torch"` (default, native R + torch) or `"stan"`
  (generates Stan code, compiles and runs via cmdstanr).

- step_size:

  Initial step size for the leapfrog integrator. If `NULL` (default),
  automatically tuned during warmup.

- max_treedepth:

  Maximum tree depth for NUTS (default 10).

- n_leapfrog:

  Number of leapfrog steps for static HMC (default 25).

- target_accept:

  Target average acceptance probability (default 0.8 for NUTS, 0.65 for
  HMC).

- init_values:

  Optional list of initial parameter vectors (one per chain).

- seed:

  Optional integer seed for reproducibility. Sets both R and torch
  random number generators.

- verbose:

  Logical; print progress information (default `TRUE`).

## Value

A `gretaR_fit` object with components:

- draws:

  Posterior draws as
  [`posterior::draws_array`](https://mc-stan.org/posterior/reference/draws_array.html).

- model:

  The compiled `gretaR_model`.

- summary:

  Posterior summary table (mean, sd, quantiles, R-hat, ESS).

- convergence:

  List: `n_eff`, `rhat`, `max_rhat`, `min_ess`, `n_divergences`.

- call_info:

  List of sampling arguments for reproducibility.

- run_time:

  Elapsed seconds.

- method:

  `"nuts"` or `"hmc"`.

Use [`coef()`](https://rdrr.io/r/stats/coef.html) for point estimates,
[`summary()`](https://rdrr.io/r/base/summary.html) for full table,
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) for
diagnostics.

## Examples

``` r
if (FALSE) { # \dontrun{
# Simple normal model
mu <- normal(0, 10)
sigma <- half_cauchy(2)
y <- as_data(rnorm(50, 3, 1.5))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)
draws <- mcmc(m, n_samples = 500, warmup = 500)
summary(draws)
} # }
```
