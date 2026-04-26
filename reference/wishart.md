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

Full sampling via Bartlett decomposition is deferred to Phase 3.

## Examples

``` r
if (FALSE) { # \dontrun{
Sigma <- wishart(df = 5, scale_matrix = diag(3))
} # }
```
