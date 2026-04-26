# Extract elements from a gretaR_array

Enables indexing into group-level parameters for hierarchical models.
Supports integer vector indexing (e.g., `alpha[group_id]`).

## Usage

``` r
# S3 method for class 'gretaR_array'
x[i, j, ..., drop = TRUE]
```

## Arguments

- x:

  A gretaR_array.

- i:

  Index: an integer vector or a gretaR_array of integer indices.

- j:

  Optional second index (for 2D arrays).

- ...:

  Additional arguments (ignored).

- drop:

  Logical (ignored; always returns a gretaR_array).

## Value

A new gretaR_array with elements selected by the index.
