# Dirichlet distribution

Creates a variable distributed according to the Dirichlet distribution,
the multivariate generalisation of the Beta distribution. Values lie on
the probability simplex (non-negative, sum to one).

## Usage

``` r
dirichlet(concentration, dim = NULL)
```

## Arguments

- concentration:

  Concentration parameter vector (positive numeric or gretaR_array).
  Length determines the dimensionality of the simplex.

- dim:

  Dimensions of the variable (inferred from concentration if NULL).

## Value

A `gretaR_array` with support on the simplex.

## Note

Latent (sampled) Dirichlet variables are not yet supported because a
correct simplex transform (stick-breaking or centered softmax with the
matching Jacobian) is not yet implemented.
[`model()`](https://max578.github.io/gretaR/reference/model.md) will
error if a Dirichlet variable is reachable as a free parameter. The
distribution can still be evaluated (`log_prob`) and is planned to be
sampler-ready in v0.3.

## Examples

``` r
if (FALSE) { # \dontrun{
theta <- dirichlet(c(1, 1, 1))
theta <- dirichlet(c(2, 5, 1))
} # }
```
