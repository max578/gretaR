# Plot Diagnostics for a gretaR Fit

Generate diagnostic plots for MCMC or VI posterior draws. Requires the
`bayesplot` package.

## Usage

``` r
# S3 method for class 'gretaR_fit'
plot(x, type = c("trace", "density", "pairs", "rhat", "neff"), ...)
```

## Arguments

- x:

  A `gretaR_fit` object.

- type:

  Plot type: `"trace"` (default), `"density"`, `"pairs"`, `"rhat"`,
  `"neff"`.

- ...:

  Additional arguments passed to the bayesplot function.

## Value

A ggplot object.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
y <- as_data(rnorm(50, 3, 1))
distribution(y) <- normal(mu, 1)
m <- model(mu)
fit <- mcmc(m, n_samples = 200, warmup = 200, chains = 2)
plot(fit, type = "trace")
} # }
```
