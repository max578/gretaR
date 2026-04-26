# Process Smooth Terms from an mgcv-Style Formula

Extract and construct smooth terms from a formula containing `s()`,
`te()`, `ti()`, or `t2()` terms. Requires the `mgcv` package.

Uses the `smooth2random` decomposition (same approach as brms) to
convert penalised smooth bases into fixed + random effect components
suitable for HMC/NUTS sampling.

## Usage

``` r
process_smooths(formula, data, knots = NULL)
```

## Arguments

- formula:

  A formula potentially containing smooth terms.

- data:

  A data frame with the covariates.

- knots:

  Optional named list of knot locations.

## Value

A list with components:

- fixed_formula:

  The formula with smooth terms removed (parametric part).

- smooth_Xf:

  Combined fixed-effects design matrix from all smooths (n x p_fixed).

- smooth_Zs:

  List of random-effects design matrices (each n x p_j).

- smooth_info:

  List of metadata for each smooth (label, type, rank, etc.).

- n_smooth_fixed:

  Number of smooth fixed-effect columns.

- n_smooth_random:

  List of dimensions per random-effect block.

## Examples

``` r
if (FALSE) { # \dontrun{
library(mgcv)
dat <- data.frame(y = rnorm(100), x = rnorm(100))
sm <- process_smooths(y ~ s(x, k = 10), data = dat)
str(sm)
} # }
```
