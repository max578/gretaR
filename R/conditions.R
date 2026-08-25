# conditions.R -- classed refusal conditions for orchestra routing
#
# Every gretaR verb that declines to produce a result (an unresolvable
# parameter, an untranslatable Stan target, a backend that is not installed,
# a fit whose diagnostics never converged, or any other input the package
# will not proceed on) raises through `gretaR_abort()` rather than a bare
# `cli::cli_abort()`. This gives the condition a class the orchestra leader
# can route on without inspecting the message text: `is_orchestra_decline()`
# (`ORCHESTRA_dev/integration/refusal_contract.R`) returns TRUE for any class
# vector containing the shared `"orchestra_refusal"` marker, or any class
# element matching `"_(refusal|abstention)$"`. gretaR mints refusals only
# (it never produces a typed "assumptions not met" object in place of a
# result), so every classed condition here ends in `_refusal`.

#' Reason codes for a gretaR refusal
#'
#' @keywords internal
.gretaR_reason_codes <- c(
  "unsupported_distribution",
  "untransformable_constraint",
  "backend_unavailable",
  "nonconvergence",
  "invalid_input"
)

#' Raise a classed gretaR refusal
#'
#' Thin wrapper around [cli::cli_abort()] that stamps every gretaR error with
#' a class vector an orchestra leader can route on: a reason-specific class
#' (`gretaR_<reason_code>_refusal`), the package-wide class (`gretaR_refusal`),
#' and the federation-shared marker (`orchestra_refusal`), ahead of the
#' `rlang_error`/`error`/`condition` classes `cli::cli_abort()` already
#' supplies. Every `cli_abort()` call site in `R/` that represents gretaR
#' declining to proceed calls this instead.
#'
#' @param message A cli-formatted message (character vector or bullet list),
#'   passed straight to [cli::cli_abort()].
#' @param reason_code One of `.gretaR_reason_codes`, identifying *why* the
#'   call was declined: `"unsupported_distribution"` (a distribution or
#'   operation gretaR does not implement), `"untransformable_constraint"` (a
#'   parameter, index, or dimension that cannot be resolved or reconciled),
#'   `"backend_unavailable"` (a required optional dependency, e.g.
#'   `cmdstanr`/`bayesplot`/`Matrix`, is not installed), `"nonconvergence"`
#'   (a diagnostic the caller asked for is not computable because sampling
#'   did not produce it), or `"invalid_input"` (the call's own arguments
#'   fail validation at the boundary).
#' @param ... Further arguments forwarded to [cli::cli_abort()] (e.g.
#'   `call`).
#' @param call The calling environment, forwarded to [cli::cli_abort()] so
#'   the error still reports the user-facing call site.
#' @param .envir Environment for cli glue interpolation (`{arg}` markers in
#'   `message`), forwarded to [cli::cli_abort()]. Defaults to the frame that
#'   called `gretaR_abort()`, matching the resolution scope a direct
#'   `cli::cli_abort()` call at that site would have had -- glue markers
#'   referring to a caller-local variable (e.g. a `for`-loop's `arg`) resolve
#'   there rather than inside this wrapper's own frame.
#'
#' @return Does not return; raises a condition of class
#'   `c("gretaR_<reason_code>_refusal", "gretaR_refusal", "orchestra_refusal",
#'   "rlang_error", "error", "condition")`.
#' @keywords internal
gretaR_abort <- function(message, reason_code, ..., call = rlang::caller_env(),
                         .envir = parent.frame()) {
  if (!isTRUE(reason_code %in% .gretaR_reason_codes)) {
    cli::cli_abort(
      "Internal error: unknown {.arg reason_code} {.val {reason_code}}.",
      call = call
    )
  }
  cls <- c(
    paste0("gretaR_", reason_code, "_refusal"),
    "gretaR_refusal",
    "orchestra_refusal"
  )
  cli::cli_abort(message, class = cls, ..., call = call, .envir = .envir)
}
