# Sum of a gretaR_array

Build an operation node that reduces a `gretaR_array` to a scalar by
summing over all elements.

## Usage

``` r
# S3 method for class 'gretaR_array'
sum(x, ..., na.rm = FALSE)
```

## Arguments

- x:

  A `gretaR_array`.

- ...:

  Ignored.

- na.rm:

  Ignored (no NAs in torch).

## Value

A scalar `gretaR_array`.
