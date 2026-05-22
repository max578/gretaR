# Remove random effects bar terms from a formula

Returns the fixed-effects-only formula by stripping all `(expr|group)`
terms. If reformulas is installed, delegates to
[`reformulas::nobars()`](https://rdrr.io/pkg/reformulas/man/nobars.html);
otherwise falls back to `lme4::nobars()`, and finally to regex
substitution.

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
#> y ~ x
#> <environment: 0x55d2c74209a8>
# y ~ x
```
