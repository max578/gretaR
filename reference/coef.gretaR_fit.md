# Extract Coefficients from a gretaR Fit

Extract posterior means (for MCMC/VI) or MAP estimates as a named
numeric vector.

## Usage

``` r
# S3 method for class 'gretaR_fit'
coef(object, ...)
```

## Arguments

- object:

  A `gretaR_fit` object.

- ...:

  Ignored.

## Value

A named numeric vector of point estimates.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
y <- as_data(rnorm(50, 3, 1))
distribution(y) <- normal(mu, 1)
m <- model(mu)
fit <- opt(m)
coef(fit)
} # }
```
