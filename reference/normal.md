# Normal Distribution

Create a normally-distributed variable with the specified mean and
standard deviation. Support is on the entire real line.

## Usage

``` r
normal(mean = 0, sd = 1, dim = NULL, truncation = NULL)
```

## Arguments

- mean:

  Mean of the distribution (numeric or `gretaR_array`).

- sd:

  Standard deviation (numeric or `gretaR_array`, positive).

- dim:

  Integer vector of dimensions (default scalar).

- truncation:

  Optional length-2 numeric vector `c(lower, upper)` specifying
  truncation bounds. Default `NULL` (no truncation). Compatible with
  greta's truncation syntax.

## Value

A `gretaR_array` representing a normally-distributed variable.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- normal(0, 1)
beta <- normal(0, 5, dim = c(3, 1))
x_pos <- normal(0, 1, truncation = c(0, Inf))
} # }
```
