# Log-Normal Distribution

A variable whose logarithm is normally distributed.

## Usage

``` r
lognormal(meanlog = 0, sdlog = 1, dim = NULL, truncation = NULL)
```

## Arguments

- meanlog:

  Mean of the log-scale distribution.

- sdlog:

  Standard deviation on the log scale (positive).

- dim:

  Dimensions.

- truncation:

  Optional length-2 numeric vector `c(lower, upper)` specifying
  truncation bounds. Default `NULL` (no truncation).

## Value

A `gretaR_array` with support on the positive reals.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- lognormal(0, 1)
} # }
```
