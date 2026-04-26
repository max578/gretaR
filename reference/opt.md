# Find the Maximum A Posteriori (MAP) Estimate

Optimise the log-joint density using gradient-based methods to find the
posterior mode. Uses the Adam optimiser via torch.

## Usage

``` r
opt(
  model,
  max_iter = 2000L,
  learning_rate = 0.01,
  tolerance = 1e-06,
  init = NULL,
  verbose = TRUE,
  backend = c("torch", "stan"),
  seed = NULL
)
```

## Arguments

- model:

  A `gretaR_model` object created by
  [`model()`](https://max578.github.io/gretaR/reference/model.md).

- max_iter:

  Maximum number of optimisation iterations (default 2000).

- learning_rate:

  Adam learning rate (default 0.01).

- tolerance:

  Convergence tolerance on relative change in log-prob (default 1e-6).

- init:

  Optional initial values (numeric vector in unconstrained space).

- verbose:

  Logical; print progress (default TRUE).

- backend:

  Inference backend: `"torch"` (default) or `"stan"`.

- seed:

  Optional integer seed for reproducibility.

## Value

A `gretaR_fit` object (method = "map") with components:

- par:

  Named numeric vector of MAP estimates (constrained space).

- par_unconstrained:

  Numeric vector of MAP in unconstrained space.

- log_prob:

  Log-joint density at the MAP.

- convergence:

  List with convergence info.

- iterations:

  Number of iterations used.

## Examples

``` r
if (FALSE) { # \dontrun{
mu <- normal(0, 10)
sigma <- half_cauchy(2)
y <- as_data(rnorm(50, 3, 1.5))
distribution(y) <- normal(mu, sigma)
m <- model(mu, sigma)
fit <- opt(m)
fit$par
} # }
```
