# Exponential Distribution

Create an exponentially-distributed variable with the specified rate.
Support is on the positive reals.

## Usage

``` r
exponential(rate = 1, dim = NULL, truncation = NULL)
```

## Arguments

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
lambda <- exponential(rate = 1)
} # }
```
