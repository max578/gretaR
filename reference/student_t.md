# Student-t Distribution

Create a Student-t-distributed variable with the specified degrees of
freedom, location, and scale. Useful as a robust alternative to the
normal distribution.

## Usage

``` r
student_t(df = 3, mu = 0, sigma = 1, dim = NULL, truncation = NULL)
```

## Arguments

- df:

  Degrees of freedom (positive numeric or `gretaR_array`).

- mu:

  Location parameter (numeric or `gretaR_array`).

- sigma:

  Scale parameter (positive numeric or `gretaR_array`).

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
x <- student_t(df = 3, mu = 0, sigma = 1)
} # }
```
