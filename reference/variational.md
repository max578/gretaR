# Variational Inference (ADVI)

Fit a model using Automatic Differentiation Variational Inference.
Approximates the posterior with a multivariate Gaussian in unconstrained
parameter space.

## Usage

``` r
variational(
  model,
  n_samples = 1L,
  max_iter = 5000L,
  learning_rate = 0.01,
  tolerance = 1e-04,
  method = c("meanfield", "fullrank"),
  init_from_map = TRUE,
  seed = NULL,
  verbose = TRUE
)
```

## Arguments

- model:

  A `gretaR_model` object created by
  [`model()`](https://max578.github.io/gretaR/reference/model.md).

- n_samples:

  Number of Monte Carlo samples per ELBO gradient estimate (default 1).

- max_iter:

  Maximum number of optimisation iterations (default 5000).

- learning_rate:

  Adam learning rate (default 0.01).

- tolerance:

  Convergence tolerance on relative ELBO change (default 1e-4).

- method:

  Variational family: `"meanfield"` (default) or `"fullrank"`.

- init_from_map:

  Logical; initialise from MAP estimate (default TRUE).

- seed:

  Optional integer seed for reproducibility. Sets both R and torch
  random number generators.

- verbose:

  Logical; print progress (default TRUE).

## Value

A `gretaR_fit` object (method = "vi") with standard fields plus:

- par:

  Named vector of posterior means (constrained space).

- sd:

  Named vector of posterior SDs (unconstrained space).

- covariance:

  Posterior covariance matrix (unconstrained, fullrank only).

- elbo:

  Vector of ELBO values per iteration.

- draws:

  A
  [`posterior::draws_array`](https://mc-stan.org/posterior/reference/draws_array.html)
  of samples from the variational posterior.

- converged:

  Logical; did the optimiser converge?

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
sigma <- half_cauchy(2)
y <- as_data(rnorm(50, 3, 1.5))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)
fit <- variational(m)
coef(fit)
} # }
```
