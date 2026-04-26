# Poisson Distribution

Create a Poisson-distributed variable with the specified rate. Named
`poisson_dist` to avoid conflict with
[`stats::poisson`](https://rdrr.io/r/stats/family.html).

## Usage

``` r
poisson_dist(rate, dim = NULL)
```

## Arguments

- rate:

  Rate parameter (positive numeric or `gretaR_array`).

- dim:

  Integer vector of dimensions (default scalar).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
y <- poisson_dist(rate = 5)
} # }
```
