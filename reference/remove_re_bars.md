# Remove random effects bar terms from a formula

Returns the fixed-effects-only formula by stripping all `(expr|group)`
terms. If lme4 is installed, delegates to `lme4::nobars()`; otherwise
uses regex substitution.

## Usage

``` r
remove_re_bars(formula)
```

## Arguments

- formula:

  A formula potentially containing bar terms.

## Value

A formula with bar terms removed.

## Examples

``` r
remove_re_bars(y ~ x + (1 | group))
#> Warning: the ‘nobars’ function has moved to the reformulas package. Please update your imports, or ask an upstream package maintainer to do so.
#> This warning is displayed once per session.
#> y ~ x
#> <environment: 0x555ac92955c8>
# y ~ x
```
