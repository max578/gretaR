# Binomial Distribution

Create a binomially-distributed variable with the specified number of
trials and probability of success per trial.

Named `binomial_dist` to avoid masking
[`stats::binomial`](https://rdrr.io/r/stats/family.html) (the GLM family
constructor).

## Usage

``` r
binomial_dist(size, prob, dim = NULL)
```

## Arguments

- size:

  Number of trials (positive integer or `gretaR_array`).

- prob:

  Probability of success per trial (numeric or `gretaR_array`, 0 to 1).

- dim:

  Integer vector of dimensions (default scalar).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
y <- binomial_dist(size = 10, prob = 0.3)
} # }
```
