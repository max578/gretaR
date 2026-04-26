# Half-Cauchy Distribution

Create a half-Cauchy-distributed variable. A popular weakly informative
prior for scale parameters (Gelman, 2006), with support on the positive
reals.

## Usage

``` r
half_cauchy(scale = 1, dim = NULL, truncation = NULL)
```

## Arguments

- scale:

  Scale parameter (positive numeric or `gretaR_array`).

- dim:

  Integer vector of dimensions (default scalar).

- truncation:

  Optional length-2 numeric vector `c(lower, upper)` specifying
  truncation bounds. Default `NULL` (no truncation).

## Value

A `gretaR_array` with support on the positive reals.

## Examples

``` r
if (FALSE) { # \dontrun{
sigma <- half_cauchy(1)
tau <- half_cauchy(5)
} # }
```
