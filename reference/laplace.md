# Laplace Approximation

Approximate the posterior distribution using a multivariate normal
centred at the MAP estimate with covariance equal to the inverse of the
negative Hessian of the log-joint density.

## Usage

``` r
laplace(model, map_fit = NULL, ...)
```

## Arguments

- model:

  A `gretaR_model` object.

- map_fit:

  Optional output from
  [`opt()`](https://max578.github.io/gretaR/reference/opt.md). If NULL,
  MAP is computed first.

- ...:

  Additional arguments passed to
  [`opt()`](https://max578.github.io/gretaR/reference/opt.md) if
  `map_fit` is NULL.

## Value

A list with components:

- mean:

  Named numeric vector of posterior means (constrained).

- mean_unconstrained:

  Posterior means in unconstrained space.

- covariance:

  Posterior covariance matrix (unconstrained space).

- sd:

  Named numeric vector of posterior standard deviations (unconstrained).

- log_evidence:

  Approximate log marginal likelihood.

- map:

  The MAP fit used.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
y <- as_data(rnorm(50, 3, 1))
distribution(y) <- normal(mu, 1)
m <- model(mu)
la <- laplace(m)
la$mean
la$sd
} # }
```
