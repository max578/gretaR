# Draw MCMC Samples from a gretaR Model

**\[experimental\]**

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
  metric = c("diag", "dense"),
  init_values = NULL,
  seed = NULL,
  batched = FALSE,
  trajectory = c("fixed", "chees"),
  device = "cpu",
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

  Safety cap on leapfrog steps per static-HMC iteration (default 25).
  HMC integrates for a random time \\T \sim U(0, 2\pi\]\\ and takes
  \\\mathrm{round}(T/\epsilon)\\ steps, capped at `10 * n_leapfrog`, to
  avoid the resonance that cripples fixed-length HMC.

- target_accept:

  Target average acceptance probability (default 0.8 for NUTS, 0.65 for
  HMC).

- metric:

  Mass-matrix metric for the single-chain samplers, estimated during
  warmup. `"diag"` (default) is a diagonal metric (inverse posterior
  variance) – the robust, Stan-default choice. `"dense"` is a dense
  metric (inverse posterior covariance) that captures linear parameter
  correlations a diagonal metric is blind to; it can substantially
  improve mixing on correlated posteriors (e.g. regression with
  correlated predictors), but it is opt-in because it does not help –
  and can hurt – funnel-shaped posteriors such as hierarchical latent
  blocks, and it adds \\O(P^2)\\ cost. Falls back to diagonal (with a
  message) when the dimension is too large or warmup draws are too few
  to estimate a covariance. Ignored on the batched path.

- init_values:

  Optional list of initial parameter vectors (one per chain).

- seed:

  Optional integer seed for reproducibility. Sets both R and torch
  random number generators.

- batched:

  Logical (default `FALSE`). When `TRUE` (and `sampler = "hmc"`), all
  chains advance together as one set of batched `torch` tensor
  operations rather than chain-by-chain. Wall-clock is then roughly flat
  in the number of chains, so many-chain runs are much faster (e.g. ~2x
  at 8 chains, ~4x at 16 on CPU). Statistically equivalent to the
  single-chain HMC. Batched NUTS is not yet supported.

- trajectory:

  Trajectory-length rule for the batched path (`batched = TRUE`).
  `"fixed"` (default) is integration-time HMC – a random time \\T \sim
  U(0, 2\pi\]\\ per iteration. `"chees"` is ChEES-HMC (Hoffman, Radul &
  Sountsov 2021), which adapts the trajectory length during warmup using
  a criterion computed across the chain ensemble, so it batches where
  NUTS's per-chain tree recursion cannot. ChEES is opt-in and most
  useful in a specific regime: it needs a reasonably large ensemble (use
  **at least ~8 chains**; below that its criterion is noisy and it can
  mix worse than NUTS) and a well-conditioned posterior. On hierarchical
  models, pair it with a near-centred
  [`random_effect`](https://max578.github.io/gretaR/reference/random_effect.md)
  – in our tests ChEES on the non-centred funnel was no better than
  NUTS, but ChEES on a near-centred parameterisation gave several times
  the effective sample size per second of the NUTS default, the two
  acting together (the centring conditions the geometry, the adaptive
  trajectory then traverses it efficiently). Single- chain NUTS remains
  the robust default; `"chees"` is ignored unless `batched = TRUE`.
  Adaptive trajectory length means wall-clock is not flat in chain count
  as it is for `"fixed"`.

- device:

  Character device for the batched path: `"cpu"` (default), `"mps"`, or
  `"cuda"`. The batched code is device-generic, but on typical models
  CPU is fastest – gretaR's log-density is many small ops, so GPU
  kernel-launch overhead dominates; GPU is for future large-model use.

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
