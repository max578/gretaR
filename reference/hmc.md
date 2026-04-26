# Run HMC Sampling

Convenience wrapper around
[`mcmc`](https://max578.github.io/gretaR/reference/mcmc.md) that selects
the static Hamiltonian Monte Carlo sampler.

## Usage

``` r
hmc(model, n_samples = 1000L, warmup = 1000L, chains = 4L, ...)
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

- ...:

  Additional arguments passed to
  [`mcmc`](https://max578.github.io/gretaR/reference/mcmc.md).

## Value

A `gretaR_fit` object.

## Examples

``` r
if (FALSE) { # \dontrun{
m <- model(normal(0, 1))
fit <- hmc(m, n_samples = 500, warmup = 500)
coef(fit)
} # }
```
