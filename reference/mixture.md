# Mixture Distribution

Define a finite mixture of distributions. The mixture is marginalised
over the discrete component indicator using the log-sum-exp trick,
enabling gradient-based inference (HMC/NUTS).

## Usage

``` r
mixture(distributions, weights)
```

## Arguments

- distributions:

  A list of gretaR distribution objects (e.g.,
  `list(normal(mu1, sigma1), normal(mu2, sigma2))`). Each must be a
  `gretaR_array` with a distribution attached.

- weights:

  A gretaR_array or numeric vector of mixture weights (must sum to 1).
  Typically from
  [`dirichlet()`](https://max578.github.io/gretaR/reference/dirichlet.md)
  or `softmax()`.

## Value

A distribution object suitable for use with
[`distribution()`](https://max578.github.io/gretaR/reference/distribution.md).

## Examples

``` r
if (FALSE) { # \dontrun{
# Two-component Gaussian mixture
w <- dirichlet(c(1, 1))
mu1 <- normal(-2, 1); mu2 <- normal(2, 1)
sigma <- half_cauchy(1)

mix <- mixture(
  distributions = list(normal(mu1, sigma), normal(mu2, sigma)),
  weights = w
)
y <- as_data(rnorm(100))
distribution(y) <- mix
m <- model(w, mu1, mu2, sigma)
} # }
```
