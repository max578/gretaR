# Build a gretaR Model from Explicit Arrays (Programmatic Interface)

Compile a computation graph into a `gretaR_model` the same way
[`model`](https://max578.github.io/gretaR/reference/model.md) does, but
with target parameters, their names, and the likelihood passed
explicitly rather than recovered from the call. This makes model
construction safe to perform programmatically – from
[`do.call()`](https://rdrr.io/r/base/do.call.html), a code generator, or
another package building gretaR models from its own representation – and
re-entrant, since the likelihood is scoped to the supplied data nodes
instead of the whole session.

## Usage

``` r
model_from_arrays(
  targets,
  likelihood = NULL,
  names = NULL,
  precision = c("float32", "float64")
)
```

## Arguments

- targets:

  A list of `gretaR_array` variable nodes – the free parameters of
  interest. These come first in the parameter vector, in the order
  given. A single `gretaR_array` is accepted and wrapped.

- likelihood:

  Optional. The data `gretaR_array`(s) whose assigned likelihood should
  enter this model, as a single array or a list of arrays. Each must
  already have a distribution attached via `distribution(y) <- ...`.
  When `NULL` (default), the session-global likelihood registry is used,
  reproducing
  [`model`](https://max578.github.io/gretaR/reference/model.md)'s
  behaviour.

- names:

  Optional parameter names, one entry per element of `targets`. A
  character vector gives one label per target (a vector target expands
  to `label[1]`, `label[2]`, ...). A list gives finer control: each
  entry is either a single label or a character vector of per-element
  names whose length matches that target's number of elements – so a
  length-`p` coefficient vector can be labelled
  `c("(Intercept)", "x1", ...)` directly, and those labels flow straight
  through to the posterior draws. When `NULL`, the names of `targets`
  are used if it is a named list; otherwise each target falls back to
  its internal node id.

- precision:

  Torch dtype: `"float32"` (default) or `"float64"`.

## Value

A `gretaR_model` object, identical in structure to one produced by
[`model`](https://max578.github.io/gretaR/reference/model.md).

## Details

`model(alpha, beta, sigma)` reads the names `"alpha"`, `"beta"`,
`"sigma"` by deparsing its call. A programmatic caller has no stable way
to reproduce that, and an interactive one cannot override it.
`model_from_arrays()` takes the targets as a list and their names as a
character vector (or as the names of the list), so the parameter labels
are whatever the caller chooses – the hook a host package needs to map
gretaR's internal names onto its own canonical scheme.

[`model()`](https://max578.github.io/gretaR/reference/model.md)
additionally folds in *every* `distribution(y) <- ...` assignment
registered in the session, so two models built in the same session share
each other's likelihood terms unless the environment is reset between
them. `model_from_arrays()` instead includes only the likelihood of the
data nodes named in `likelihood`, leaving the global registry untouched.
Independent models can therefore be compiled in sequence without
interference.

The two front-ends delegate to the same internal compiler, so a model
built with `model_from_arrays()` is indistinguishable from the
equivalent
[`model()`](https://max578.github.io/gretaR/reference/model.md) call and
is accepted everywhere a `gretaR_model` is –
[`mcmc`](https://max578.github.io/gretaR/reference/mcmc.md),
[`opt`](https://max578.github.io/gretaR/reference/opt.md),
[`joint_density`](https://max578.github.io/gretaR/reference/joint_density.md).

## See also

[`model`](https://max578.github.io/gretaR/reference/model.md) for the
interactive front-end.

## Examples

``` r
if (FALSE) { # \dontrun{
# Programmatic construction — names supplied, not deparsed.
mu <- normal(0, 10)
sigma <- half_cauchy(1)
y <- as_data(rnorm(100, 3))
distribution(y) <- normal(mu, sigma)
m <- model_from_arrays(
  targets = list(mu, sigma),
  likelihood = y,
  names = c("mu", "sigma")
)

# Re-entrant: a second, independent model in the same session.
theta <- normal(0, 1)
z <- as_data(rnorm(50))
distribution(z) <- normal(theta, 1)
m2 <- model_from_arrays(targets = list(theta), likelihood = z, names = "theta")
} # }
```
