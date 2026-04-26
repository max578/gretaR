# Beta Distribution

Create a beta-distributed variable with the specified shape parameters.
Support is on the interval `(0, 1)`. Named `beta_dist` to avoid conflict
with [`base::beta`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
beta_dist(alpha, beta, dim = NULL, truncation = NULL)
```

## Arguments

- alpha:

  First shape parameter (positive numeric or `gretaR_array`).

- beta:

  Second shape parameter (positive numeric or `gretaR_array`).

- dim:

  Integer vector of dimensions (default scalar).

- truncation:

  Optional length-2 numeric vector `c(lower, upper)` specifying
  truncation bounds. Default `NULL` (no truncation).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
p <- beta_dist(alpha = 2, beta = 5)
} # }
```
