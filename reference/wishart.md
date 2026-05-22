# Wishart Distribution

The Wishart distribution over positive-definite matrices. Used as a
prior for covariance or precision matrices.

## Usage

``` r
wishart(df, scale_matrix)
```

## Arguments

- df:

  Degrees of freedom (must be \>= dimension of scale matrix).

- scale_matrix:

  Scale matrix (positive definite, p x p).

## Value

A `gretaR_array` representing a positive-definite matrix.

## Note

Latent (sampled) Wishart variables are not yet supported because a
correct positive-definite-matrix transform (Cholesky factor with
positive diagonal plus the matching Jacobian) is not yet implemented.
[`model()`](https://max578.github.io/gretaR/reference/model.md) will
error if a Wishart variable is reachable as a free parameter. The
distribution can still be evaluated (`log_prob`); sampler-ready support
is planned for v0.3.

## Examples

``` r
if (FALSE) { # \dontrun{
Sigma <- wishart(df = 5, scale_matrix = diag(3))
} # }
```
