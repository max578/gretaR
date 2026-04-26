# Bernoulli Distribution

Create a Bernoulli-distributed variable with the specified probability
of success. Support is on `{0, 1}`.

## Usage

``` r
bernoulli(prob, dim = NULL)
```

## Arguments

- prob:

  Probability of success (numeric or `gretaR_array`, 0 to 1).

- dim:

  Integer vector of dimensions (default scalar).

## Value

A `gretaR_array`.

## Examples

``` r
if (FALSE) { # \dontrun{
z <- bernoulli(0.5)
} # }
```
