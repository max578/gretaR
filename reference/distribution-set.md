# Assign a Distribution (Likelihood) to Observed Data

Define the likelihood by assigning a distribution to a data
`gretaR_array`. This registers the distribution as a likelihood term in
the model's log-joint density.

## Usage

``` r
distribution(x) <- value
```

## Arguments

- x:

  A data `gretaR_array` (created with
  [`as_data`](https://max578.github.io/gretaR/reference/as_data.md)).

- value:

  A distribution `gretaR_array` (e.g., from
  [`normal`](https://max578.github.io/gretaR/reference/normal.md)).

## Value

The data `gretaR_array` `x`, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
y <- as_data(rnorm(100))
mu <- normal(0, 10)
sigma <- half_cauchy(1)
distribution(y) <- normal(mu, sigma)
} # }
```
