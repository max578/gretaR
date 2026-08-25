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

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
y <- as_data(rnorm(50, 3, 1))
distribution(y) <- normal(mu, 1)
m <- model(mu)
fit <- mcmc(m, n_samples = 200, warmup = 200, chains = 2)
summary(fit)
} # }
```
