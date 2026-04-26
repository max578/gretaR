# Custom Distribution

Define a distribution with a user-supplied log-probability density
function. The function must accept a torch tensor and return a scalar
torch tensor. It must be differentiable via torch autograd.

## Usage

``` r
custom_distribution(
  log_prob_fn,
  constraint = NULL,
  dim = NULL,
  name = "custom"
)
```

## Arguments

- log_prob_fn:

  A function `f(x) -> scalar torch tensor` computing the log-probability
  density at `x`.

- constraint:

  Optional list with `lower` and `upper` bounds for the parameter space.
  Determines the transform for HMC sampling.

- dim:

  Dimensions of the variable.

- name:

  Optional name for display purposes.

## Value

A `gretaR_array` representing a variable with the custom distribution.

## Examples

``` r
if (FALSE) { # \dontrun{
# Laplace distribution (not built in)
x <- custom_distribution(
  log_prob_fn = function(x) -torch_sum(torch_abs(x)),
  name = "laplace"
)

# Truncated normal (positive only)
x <- custom_distribution(
  log_prob_fn = function(x) {
    torch_sum(-0.5 * x^2)  # kernel of N(0,1)
  },
  constraint = list(lower = 0, upper = Inf),
  name = "truncated_normal"
)
} # }
```
