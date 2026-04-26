# Get the Distribution of a gretaR Array

Retrieve the distribution object associated with a `gretaR_array`
variable node, or `NULL` if none is set.

## Usage

``` r
distribution(x)
```

## Arguments

- x:

  A `gretaR_array`.

## Value

The `GretaRDistribution` object, or `NULL`.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 1)
distribution(mu)
} # }
```
