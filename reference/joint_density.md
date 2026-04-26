# Get the Log Joint Density Function

Extract a torch-compatible function that evaluates the log joint density
of a compiled `gretaR_model` at a given parameter vector in
unconstrained space.

## Usage

``` r
joint_density(model)
```

## Arguments

- model:

  A `gretaR_model` object created by
  [`model`](https://max578.github.io/gretaR/reference/model.md).

## Value

A function `f(theta)` that takes a 1-D torch tensor of unconstrained
parameter values and returns a scalar torch tensor (the log joint
density).

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
y <- as_data(rnorm(50, 3))
distribution(y) <- normal(mu, 1)
m <- model(mu)
ld <- joint_density(m)
ld(torch::torch_zeros(1))
} # }
```
