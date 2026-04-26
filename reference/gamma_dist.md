# Gamma Distribution

Create a gamma-distributed variable with the specified shape and rate.
Support is on the positive reals. Named `gamma_dist` to avoid conflict
with [`base::gamma`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
gamma_dist(shape, rate, dim = NULL, truncation = NULL)
```

## Arguments

- shape:

  Shape parameter (positive numeric or `gretaR_array`).

- rate:

  Rate parameter (positive numeric or `gretaR_array`).

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
tau <- gamma_dist(shape = 2, rate = 1)
} # }
```
