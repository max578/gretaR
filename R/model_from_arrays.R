# model_from_arrays.R — re-entrant, deparse-free model construction.
#
# `model()` recovers its parameter names by deparsing the unevaluated `...`
# expressions of its own call. That is convenient interactively but makes the
# function impossible to drive programmatically: under `do.call()`, inside a
# code generator, or from another package's intermediate representation, the
# deparsed text is an expression rather than the name the caller intends, and
# there is no way to supply names out of band. `model_from_arrays()` is the
# explicit counterpart — targets and their names are passed as ordinary
# arguments, and the likelihood is scoped to the data nodes the caller
# declares, so independent models can be built back to back in one session
# without `reset_gretaR_env()` between them. Both front-ends share the
# `.compile_gretaR_model()` core, so the resulting `gretaR_model` objects are
# identical.

#' @title Build a gretaR Model from Explicit Arrays (Programmatic Interface)
#'
#' @description Compile a computation graph into a \code{gretaR_model} the same
#'   way \code{\link{model}} does, but with target parameters, their names, and
#'   the likelihood passed explicitly rather than recovered from the call. This
#'   makes model construction safe to perform programmatically -- from
#'   \code{do.call()}, a code generator, or another package building gretaR
#'   models from its own representation -- and re-entrant, since the likelihood
#'   is scoped to the supplied data nodes instead of the whole session.
#'
#' @details \code{model(alpha, beta, sigma)} reads the names
#'   \code{"alpha"}, \code{"beta"}, \code{"sigma"} by deparsing its call. A
#'   programmatic caller has no stable way to reproduce that, and an interactive
#'   one cannot override it. \code{model_from_arrays()} takes the targets as a
#'   list and their names as a character vector (or as the names of the list),
#'   so the parameter labels are whatever the caller chooses -- the hook a host
#'   package needs to map gretaR's internal names onto its own canonical scheme.
#'
#'   \code{model()} additionally folds in \emph{every} \code{distribution(y) <-
#'   ...} assignment registered in the session, so two models built in the same
#'   session share each other's likelihood terms unless the environment is reset
#'   between them. \code{model_from_arrays()} instead includes only the
#'   likelihood of the data nodes named in \code{likelihood}, leaving the global
#'   registry untouched. Independent models can therefore be compiled in
#'   sequence without interference.
#'
#'   The two front-ends delegate to the same internal compiler, so a model built
#'   with \code{model_from_arrays()} is indistinguishable from the equivalent
#'   \code{model()} call and is accepted everywhere a \code{gretaR_model} is --
#'   \code{\link{mcmc}}, \code{\link{opt}}, \code{\link{joint_density}}.
#'
#' @param targets A list of \code{gretaR_array} variable nodes -- the free
#'   parameters of interest. These come first in the parameter vector, in the
#'   order given. A single \code{gretaR_array} is accepted and wrapped.
#' @param likelihood Optional. The data \code{gretaR_array}(s) whose assigned
#'   likelihood should enter this model, as a single array or a list of arrays.
#'   Each must already have a distribution attached via \code{distribution(y) <-
#'   ...}. When \code{NULL} (default), the session-global likelihood registry is
#'   used, reproducing \code{\link{model}}'s behaviour.
#' @param names Optional parameter names, one entry per element of
#'   \code{targets}. A character vector gives one label per target (a vector
#'   target expands to \code{label[1]}, \code{label[2]}, ...). A list gives finer
#'   control: each entry is either a single label or a character vector of
#'   per-element names whose length matches that target's number of elements --
#'   so a length-\code{p} coefficient vector can be labelled
#'   \code{c("(Intercept)", "x1", ...)} directly, and those labels flow straight
#'   through to the posterior draws. When \code{NULL}, the names of
#'   \code{targets} are used if it is a named list; otherwise each target falls
#'   back to its internal node id.
#' @param precision Torch dtype: \code{"float32"} (default) or \code{"float64"}.
#'
#' @return A \code{gretaR_model} object, identical in structure to one produced
#'   by \code{\link{model}}.
#'
#' @seealso \code{\link{model}} for the interactive front-end.
#' @export
#' @examples
#' \dontrun{
#' # Programmatic construction — names supplied, not deparsed.
#' mu <- normal(0, 10)
#' sigma <- half_cauchy(1)
#' y <- as_data(rnorm(100, 3))
#' distribution(y) <- normal(mu, sigma)
#' m <- model_from_arrays(
#'   targets = list(mu, sigma),
#'   likelihood = y,
#'   names = c("mu", "sigma")
#' )
#'
#' # Re-entrant: a second, independent model in the same session.
#' theta <- normal(0, 1)
#' z <- as_data(rnorm(50))
#' distribution(z) <- normal(theta, 1)
#' m2 <- model_from_arrays(targets = list(theta), likelihood = z, names = "theta")
#' }
model_from_arrays <- function(targets, likelihood = NULL, names = NULL,
                              precision = c("float32", "float64")) {
  precision <- rlang::arg_match(precision)
  dtype <- if (precision == "float64") torch_float64() else torch_float32()

  # Accept a bare array for the common single-target case.
  if (inherits(targets, "gretaR_array")) {
    targets <- list(targets)
  }
  if (!is.list(targets) || length(targets) == 0L) {
    cli_abort(c(
      "{.arg targets} must be a non-empty list of {.cls gretaR_array} variables.",
      "i" = "Pass e.g. {.code targets = list(mu, sigma)} or a single array."
    ))
  }

  # Resolve names with no call introspection. `display` is one node label per
  # target (for `model()` parity and printing); `elements` optionally carries a
  # per-element name vector per target (the full canonical-name contract).
  resolved <- .resolve_names(targets, names)

  # Scope the likelihood. NULL keeps the global registry (model() parity);
  # otherwise include only the declared data nodes' assigned distributions.
  likelihood_terms <- .scope_likelihood_terms(likelihood)

  m <- .compile_gretaR_model(
    targets = targets,
    target_names = resolved$display,
    likelihood_terms = likelihood_terms,
    dtype = dtype
  )

  # Attach caller-supplied per-element names so they flow straight through to
  # the draws (via make_param_names()), eliminating any downstream relabel.
  for (i in seq_along(targets)) {
    el <- resolved$elements[[i]]
    if (is.null(el)) next
    vid <- get_node(targets[[i]])$id
    n_elem <- m$param_info[[vid]]$n_elem
    if (length(el) != n_elem) {
      cli_abort(paste(
        "Target {i} has {n_elem} element{?s} but {length(el)} element name{?s}",
        "were supplied. Provide one name per element, or a single node name."
      ))
    }
    m$param_info[[vid]]$element_names <- el
  }

  m
}

#' Resolve names without deparsing the call.
#'
#' Returns `list(display, elements)`: `display` is a length-n character vector
#' of node labels (one per target); `elements[[i]]` is either `NULL` or a
#' character vector of per-element names for target `i`. `names` may be a
#' character vector (one node label each) or a list (each entry a single label
#' or a per-element vector).
#' @noRd
.resolve_names <- function(targets, names) {
  n <- length(targets)
  node_id <- function(a) {
    node <- get_node(a)
    if (is.null(node)) {
      cli_abort("Every element of {.arg targets} must be a {.cls gretaR_array}.")
    }
    node$id
  }
  display <- character(n)
  elements <- vector("list", n)

  check_str <- function(v, what) {
    if (anyNA(v) || any(!nzchar(v))) {
      cli_abort("{.arg names} must contain no missing or empty strings ({what}).")
    }
  }

  if (is.null(names)) {
    list_names <- base::names(targets)
    if (!is.null(list_names) && all(nzchar(list_names))) {
      display <- list_names
    } else {
      display <- vapply(targets, node_id, character(1))
    }
    return(list(display = display, elements = elements))
  }

  if (length(names) != n) {
    cli_abort(paste(
      "{.arg names} has length {length(names)} but there are {n} target{?s}.",
      "Supply exactly one entry per target."
    ))
  }

  # List form allows a per-element name vector per target.
  if (is.list(names)) {
    slot_names <- base::names(names)
    for (i in seq_len(n)) {
      entry <- as.character(names[[i]])
      check_str(entry, paste("target", i))
      if (length(entry) <= 1L) {
        display[i] <- if (length(entry) == 1L) entry else node_id(targets[[i]])
      } else {
        elements[[i]] <- entry
        display[i] <- if (!is.null(slot_names) && nzchar(slot_names[i])) {
          slot_names[i]
        } else {
          node_id(targets[[i]])
        }
      }
    }
    return(list(display = display, elements = elements))
  }

  # Character vector: one node label per target.
  display <- as.character(names)
  check_str(display, "node labels")
  list(display = display, elements = elements)
}

#' Build a likelihood-term list scoped to the declared data nodes.
#' @noRd
.scope_likelihood_terms <- function(likelihood) {
  if (is.null(likelihood)) {
    return(.gretaR_env$distributions)
  }
  if (inherits(likelihood, "gretaR_array")) {
    likelihood <- list(likelihood)
  }
  if (!is.list(likelihood)) {
    cli_abort(paste(
      "{.arg likelihood} must be a data {.cls gretaR_array} or a list of them."
    ))
  }

  terms <- list()
  for (d in likelihood) {
    dn <- get_node(d)
    if (is.null(dn)) {
      cli_abort("Each {.arg likelihood} element must be a {.cls gretaR_array}.")
    }
    reg <- .gretaR_env$distributions[[dn$id]]
    if (is.null(reg)) {
      cli_abort(c(
        "No likelihood is attached to one of the supplied data nodes.",
        "x" = "Data node {.val {dn$id}} has no {.code distribution(y) <- ...}.",
        "i" = "Assign a distribution to it before building the model."
      ))
    }
    terms[[dn$id]] <- reg
  }
  terms
}
