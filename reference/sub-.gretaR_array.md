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

  Not supported. Supplying a column index raises an error; only
  single-index row/element selection is currently implemented.

- ...:

  Additional arguments (ignored).

- drop:

  Logical (ignored; always returns a gretaR_array).

## Value

A new gretaR_array with elements selected by the index.
