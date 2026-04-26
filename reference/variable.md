# Create a Free Variable

Create an unconstrained or constrained variable without a distributional
prior. Useful for deterministic transformations or parameters that do
not require a prior.

## Usage

``` r
variable(lower = -Inf, upper = Inf, dim = NULL)
```

## Arguments

- lower:

  Lower bound (default `-Inf`, unconstrained).

- upper:

  Upper bound (default `Inf`, unconstrained).

- dim:

  Integer vector of dimensions (default `c(1, 1)`).

## Value

A `gretaR_array` variable node.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- variable()
x_pos <- variable(lower = 0)
x_bounded <- variable(lower = 0, upper = 1)
} # }
```
