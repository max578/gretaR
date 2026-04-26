# Half-Normal Distribution

Create a half-normal-distributed variable. The half-normal is the
absolute value of a normal distribution, with support on the positive
reals.

## Usage

``` r
half_normal(sd = 1, dim = NULL, truncation = NULL)
```

## Arguments

- sd:

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
sigma <- half_normal(1)
} # }
```
