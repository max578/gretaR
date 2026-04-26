# Multivariate Normal Distribution

Create a multivariate-normal-distributed variable with the specified
mean vector and covariance matrix. If `dim` is `NULL`, it is inferred
from the length of `mean`.

## Usage

``` r
multivariate_normal(mean, covariance, dim = NULL)
```

## Arguments

- mean:

  Numeric mean vector or `gretaR_array`.

- covariance:

  Covariance matrix (positive definite numeric matrix or
  `gretaR_array`).

- dim:

  Integer vector of dimensions (inferred from `mean` if `NULL`).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- c(0, 0)
Sigma <- diag(2)
x <- multivariate_normal(mean = mu, covariance = Sigma)
} # }
```
