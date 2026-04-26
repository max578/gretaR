# Matrix multiplication with gretaR_array support

Overrides base `%*%` to dispatch to sparse-aware matrix multiplication
when either argument is a `gretaR_array`.

## Usage

``` r
x %*% y
```

## Arguments

- x:

  A matrix, gretaR_array, or numeric.

- y:

  A matrix, gretaR_array, or numeric.

## Value

A gretaR_array (if either input is gretaR_array) or the base result.
