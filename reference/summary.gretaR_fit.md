# Summarise a gretaR Fit Object

Compute or display detailed posterior summary statistics and convergence
diagnostics.

## Usage

``` r
# S3 method for class 'gretaR_fit'
summary(object, ...)
```

## Arguments

- object:

  A `gretaR_fit` object.

- ...:

  Additional arguments passed to
  [`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html).

## Value

A data frame of posterior summaries (from
[`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)),
or a list for MAP/Laplace fits.
