# Wrap Observed Data as a gretaR Array

Convert numeric vectors, matrices, or arrays into a `gretaR_array` data
node for use in model definitions. Data nodes are fixed (not sampled)
during inference.

## Usage

``` r
as_data(x)
```

## Arguments

- x:

  A numeric vector, matrix, or array of observed data.

## Value

A `gretaR_array` representing fixed (observed) data.

## Examples

``` r
if (FALSE) { # \dontrun{
y <- as_data(iris$Sepal.Length)
X <- as_data(matrix(rnorm(30), ncol = 3))
} # }
```
