# LKJ Correlation distribution

The LKJ distribution over correlation matrices (Lewandowski, Kurowicka,
and Joe, 2009). The density is proportional to \\\det(R)^{\eta - 1}\\
where \\R\\ is a correlation matrix. When \\\eta = 1\\, the distribution
is uniform over valid correlation matrices.

## Usage

``` r
lkj_correlation(eta = 1, dim = 2L)
```

## Arguments

- eta:

  Shape parameter (positive). Values \> 1 concentrate mass around the
  identity matrix; values \< 1 favour extreme correlations.

- dim:

  Dimension of the correlation matrix (integer \>= 2).

## Value

A `gretaR_array` representing a correlation matrix.

## Note

Simplex/correlation transforms and efficient sampling are deferred to
Phase 3. The current implementation uses an identity transform and stub
sampling (returns identity matrices).

## Examples

``` r
if (FALSE) { # \dontrun{
R <- lkj_correlation(eta = 2, dim = 3)
} # }
```
