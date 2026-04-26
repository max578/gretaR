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
