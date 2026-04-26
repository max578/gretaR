# Cauchy Distribution

The Cauchy distribution (Student-t with 1 degree of freedom).
Heavy-tailed; useful as a weakly informative prior.

## Usage

``` r
cauchy(location = 0, scale = 1, dim = NULL, truncation = NULL)
```

## Arguments

- location:

  Location parameter.

- scale:

  Scale parameter (positive).

- dim:

  Dimensions.

- truncation:

  Optional length-2 numeric vector `c(lower, upper)` specifying
  truncation bounds. Default `NULL` (no truncation).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- cauchy(0, 1)
} # }
```
