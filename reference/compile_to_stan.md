# Generate Stan Code from a gretaR Model

Translate a compiled `gretaR_model` into Stan code suitable for use with
`cmdstanr`. The generated code preserves the model structure defined via
the gretaR DSL.

## Usage

``` r
compile_to_stan(model)
```

## Arguments

- model:

  A `gretaR_model` object created by
  [`model()`](https://max578.github.io/gretaR/reference/model.md).

## Value

A character string containing valid Stan code.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
sigma <- half_cauchy(2)
y <- as_data(rnorm(50, 3, 1.5))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)
cat(compile_to_stan(m))
} # }
```
