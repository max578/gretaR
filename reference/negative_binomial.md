# Negative Binomial distribution

Parameterised by the number of successes `size` (r) and the probability
of success `prob` (p). Models the number of failures before `size`
successes occur.

## Usage

``` r
negative_binomial(size, prob, dim = NULL)
```

## Arguments

- size:

  Target number of successes (positive numeric or gretaR_array).

- prob:

  Probability of success per trial (0 to 1).

- dim:

  Dimensions.

## Value

A `gretaR_array` (discrete, non-negative integers).

## Examples

``` r
if (FALSE) { # \dontrun{
y <- negative_binomial(size = 5, prob = 0.5)
} # }
```
