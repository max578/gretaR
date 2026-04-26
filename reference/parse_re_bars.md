# Parse random effects bar terms from an lme4-style formula

Extracts all `(expr | group)` terms and classifies them as
`"intercept"`, `"slope_only"`, or `"intercept_slope"`.

## Usage

``` r
parse_re_bars(formula)
```

## Arguments

- formula:

  A formula potentially containing bar terms.

## Value

A list of parsed random effect specifications. Each element is a list
with components: `raw` (original text), `lhs` (left of bar), `group`
(grouping variable), `type` (one of `"intercept"`, `"slope_only"`,
`"intercept_slope"`), and `slope_vars` (character vector of slope
variable names, may be empty).

## Examples

``` r
parse_re_bars(y ~ x + (1 | group))
#> [[1]]
#> [[1]]$raw
#> [1] "(1 | group)"
#> 
#> [[1]]$lhs
#> [1] "1"
#> 
#> [[1]]$group
#> [1] "group"
#> 
#> [[1]]$type
#> [1] "intercept"
#> 
#> [[1]]$slope_vars
#> character(0)
#> 
#> 
parse_re_bars(y ~ x + (x | group))
#> [[1]]
#> [[1]]$raw
#> [1] "(x | group)"
#> 
#> [[1]]$lhs
#> [1] "x"
#> 
#> [[1]]$group
#> [1] "group"
#> 
#> [[1]]$type
#> [1] "intercept_slope"
#> 
#> [[1]]$slope_vars
#> [1] "x"
#> 
#> 
parse_re_bars(y ~ x + (0 + x | group))
#> [[1]]
#> [[1]]$raw
#> [1] "(0 + x | group)"
#> 
#> [[1]]$lhs
#> [1] "0 + x"
#> 
#> [[1]]$group
#> [1] "group"
#> 
#> [[1]]$type
#> [1] "slope_only"
#> 
#> [[1]]$slope_vars
#> [1] "x"
#> 
#> 
parse_re_bars(y ~ x + (1 | site) + (1 | year))
#> [[1]]
#> [[1]]$raw
#> [1] "(1 | site)"
#> 
#> [[1]]$lhs
#> [1] "1"
#> 
#> [[1]]$group
#> [1] "site"
#> 
#> [[1]]$type
#> [1] "intercept"
#> 
#> [[1]]$slope_vars
#> character(0)
#> 
#> 
#> [[2]]
#> [[2]]$raw
#> [1] "(1 | year)"
#> 
#> [[2]]$lhs
#> [1] "1"
#> 
#> [[2]]$group
#> [1] "year"
#> 
#> [[2]]$type
#> [1] "intercept"
#> 
#> [[2]]$slope_vars
#> character(0)
#> 
#> 
```
