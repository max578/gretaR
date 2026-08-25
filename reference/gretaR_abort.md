# Raise a classed gretaR refusal

Thin wrapper around
[`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
that stamps every gretaR error with a class vector an orchestra leader
can route on: a reason-specific class (`gretaR_<reason_code>_refusal`),
the package-wide class (`gretaR_refusal`), and the federation-shared
marker (`orchestra_refusal`), ahead of the
`rlang_error`/`error`/`condition` classes
[`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
already supplies. Every `cli_abort()` call site in `R/` that represents
gretaR declining to proceed calls this instead.

## Usage

``` r
gretaR_abort(
  message,
  reason_code,
  ...,
  call = rlang::caller_env(),
  .envir = parent.frame()
)
```

## Arguments

- message:

  A cli-formatted message (character vector or bullet list), passed
  straight to
  [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html).

- reason_code:

  One of `.gretaR_reason_codes`, identifying *why* the call was
  declined: `"unsupported_distribution"` (a distribution or operation
  gretaR does not implement), `"untransformable_constraint"` (a
  parameter, index, or dimension that cannot be resolved or reconciled),
  `"backend_unavailable"` (a required optional dependency, e.g.
  `cmdstanr`/`bayesplot`/`Matrix`, is not installed), `"nonconvergence"`
  (a diagnostic the caller asked for is not computable because sampling
  did not produce it), or `"invalid_input"` (the call's own arguments
  fail validation at the boundary).

- ...:

  Further arguments forwarded to
  [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
  (e.g. `call`).

- call:

  The calling environment, forwarded to
  [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
  so the error still reports the user-facing call site.

- .envir:

  Environment for cli glue interpolation (`{arg}` markers in `message`),
  forwarded to
  [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html).
  Defaults to the frame that called `gretaR_abort()`, matching the
  resolution scope a direct
  [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
  call at that site would have had – glue markers referring to a
  caller-local variable (e.g. a `for`-loop's `arg`) resolve there rather
  than inside this wrapper's own frame.

## Value

Does not return; raises a condition of class
`c("gretaR_<reason_code>_refusal", "gretaR_refusal", "orchestra_refusal", "rlang_error", "error", "condition")`.
