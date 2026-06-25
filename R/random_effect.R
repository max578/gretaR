# random_effect.R — partially-centred random-effect blocks.
#
# A grouped (random-intercept) effect can be written in two geometries that
# describe the *same* posterior but sample very differently. The centred
# parameterisation draws the group effects directly, u_j ~ N(mean, sd^2); the
# non-centred parameterisation draws standardised effects eta_j ~ N(0, 1) and
# sets u_j = mean + sd * eta_j. Neither is uniformly better: when a group is
# richly informed by data the centred form mixes well and the non-centred form
# induces a posterior funnel, and when a group is data-poor the reverse holds
# (Papaspiliopoulos, Roberts & Skold 2007; Betancourt & Girolami 2015).
#
# `random_effect()` exposes the continuum between the two through a single
# centring weight w in [0, 1] (w = 0 non-centred, w = 1 centred), applied
# globally or per group. The partial parameterisation samples
# xi_j ~ N(0, sd^{w_j}) and returns u_j = mean + sd^{1 - w_j} * xi_j, so that
# u_j ~ N(mean, sd^2) marginally for every w_j -- only the sampling geometry
# changes, never the target. This is the construction-time primitive a
# data-adaptive scheme drives by estimating each w_j from the data; on its own
# it lets a caller pin the parameterisation that suits their design.

#' @title Partially-Centred Random-Effect Block
#'
#' @description Construct a grouped random-effect term with an explicit
#'   centring weight that interpolates between the non-centred and centred
#'   parameterisations. The weight controls only the sampling geometry seen by
#'   the HMC/NUTS sampler -- the implied prior on the group effects,
#'   \eqn{u_j \sim N(\mathrm{mean}, \mathrm{sd}^2)}, is identical for every
#'   weight -- so it is a pure efficiency control for hierarchical models whose
#'   mixing is bottlenecked by the funnel coupling between a group-scale
#'   parameter and its latent effects.
#'
#' @details A centring weight \code{w = 0} gives the non-centred form
#'   \eqn{u_j = \mathrm{mean} + \mathrm{sd}\,\xi_j} with
#'   \eqn{\xi_j \sim N(0, 1)}, which mixes well when groups are weakly informed.
#'   A weight \code{w = 1} gives the centred form
#'   \eqn{u_j = \mathrm{mean} + \xi_j} with
#'   \eqn{\xi_j \sim N(0, \mathrm{sd}^2)}, which mixes well when groups are
#'   strongly informed. Intermediate and per-group weights interpolate:
#'   \deqn{\xi_j \sim N(0, \mathrm{sd}^{w_j}), \qquad
#'         u_j = \mathrm{mean} + \mathrm{sd}^{\,1 - w_j}\,\xi_j,}
#'   which leaves \eqn{u_j \sim N(\mathrm{mean}, \mathrm{sd}^2)} unchanged. The
#'   informativeness-optimal weight for a balanced group of \eqn{n_j}
#'   observations with residual scale \eqn{\sigma} is
#'   \eqn{w_j^\star = (n_j/\sigma^2) / (n_j/\sigma^2 + 1/\mathrm{sd}^2)}; a
#'   data-adaptive sampler estimates it during warmup, but a fixed value may be
#'   supplied directly here.
#'
#'   The returned latent block \code{$latent} is the free parameter the sampler
#'   moves and must be passed to \code{\link{model_from_arrays}} (or
#'   \code{\link{model}}); the returned \code{$effect} is the per-observation
#'   effect to add into the linear predictor. The latent block carries the
#'   group-level names, not the observation-level ones.
#'
#' @param group Integer vector of length \eqn{n} mapping each observation to a
#'   group in \code{seq_len(n_groups)}, or the equivalent data
#'   \code{gretaR_array}. Used to gather the group effects to the observations.
#' @param n_groups Number of groups \eqn{J} (a positive integer).
#' @param sd A scalar \code{gretaR_array} -- the random-effect standard
#'   deviation (commonly a \code{\link{half_normal}} or \code{\link{half_cauchy}}
#'   group-scale parameter). The funnel partner of the latent block.
#' @param centring The centring weight \eqn{w}. Either a single number in
#'   \eqn{[0, 1]} applied to every group, or a numeric vector of length
#'   \code{n_groups} giving a per-group weight. Defaults to \code{0}
#'   (non-centred), gretaR's historical hierarchical parameterisation.
#' @param mean The random-effect mean, a numeric scalar or scalar
#'   \code{gretaR_array}. Defaults to \code{0}: the standard zero-mean
#'   deviation block whose global level lives in a separate intercept term.
#'
#' @return A list of class \code{gretaR_random_effect} with elements
#'   \describe{
#'     \item{\code{latent}}{The \eqn{J}-element latent \code{gretaR_array} to
#'       include in the model's target list.}
#'     \item{\code{effect}}{The \eqn{n}-element per-observation effect
#'       \code{gretaR_array} for the linear predictor.}
#'     \item{\code{centring}}{The resolved per-group weight vector.}
#'     \item{\code{n_groups}}{The number of groups.}
#'   }
#'
#' @seealso \code{\link{model_from_arrays}} for the programmatic model
#'   front-end the latent block is designed for.
#' @export
#' @examples
#' \dontrun{
#' # Random-intercept model, near-centred (informative groups).
#' n <- 1000L
#' J <- 20L
#' g <- rep(seq_len(J), length.out = n)
#' x <- rnorm(n)
#' y <- as_data(1.5 - 0.8 * x + rnorm(n))
#'
#' b0 <- normal(0, 5)
#' b1 <- normal(0, 5)
#' s <- half_normal(5)
#' tau <- half_normal(2)
#'
#' re <- random_effect(g, n_groups = J, sd = tau, centring = 0.9)
#' eta <- b0 + b1 * as_data(x) + re$effect
#' distribution(y) <- normal(eta, s)
#'
#' m <- model_from_arrays(
#'   list(b0, b1, s, tau, re$latent),
#'   likelihood = y,
#'   names = list("b0", "b1", "s", "tau", paste0("u[", seq_len(J), "]"))
#' )
#' }
random_effect <- function(group, n_groups, sd, centring = 0, mean = 0) {
  if (!inherits(sd, "gretaR_array")) {
    cli_abort(c(
      "{.arg sd} must be a scalar {.cls gretaR_array} (the group-scale parameter).",
      "i" = "Pass e.g. {.code sd = half_normal(2)}, not a plain number."
    ))
  }

  n_groups <- .as_count(n_groups, "n_groups")
  w <- .resolve_centring(centring, n_groups)
  group_idx <- .resolve_group_index(group, n_groups)

  # Per-group prior and transform scales. sd^0 = 1 (non-centred end) and
  # sd^1 = sd (centred end) fall out of the same two expressions, so the
  # endpoints need no special-casing.
  w_node <- as_data(matrix(w, ncol = 1L))
  sd_prior <- sd^w_node # N(0, sd^{w_j}) prior scale
  sd_transform <- sd^(1 - w_node) # sd^{1 - w_j} transform scale

  xi <- normal(0, sd_prior, dim = c(n_groups, 1L))
  u <- mean + sd_transform * xi
  effect <- u[group_idx]

  structure(
    list(
      latent = xi,
      effect = effect,
      centring = w,
      n_groups = n_groups
    ),
    class = "gretaR_random_effect"
  )
}

#' @noRd
#' @export
print.gretaR_random_effect <- function(x, ...) {
  w <- x$centring
  lab <- if (length(unique(w)) == 1L) {
    sprintf("w = %.3g (%s)", w[1], .centring_label(w[1]))
  } else {
    sprintf(
      "per-group w in [%.3g, %.3g]", min(w), max(w)
    )
  }
  cli::cli_text("{.cls gretaR_random_effect}: {x$n_groups} group{?s}, centring {lab}")
  cli::cli_text(
    "Pass {.code $latent} to the model targets; add {.code $effect} to the linear predictor."
  )
  invisible(x)
}

# Internal helpers: centring resolution and validation -----------------------

#' Human label for a centring weight.
#' @noRd
.centring_label <- function(w) {
  if (isTRUE(all.equal(w, 0))) {
    "non-centred"
  } else if (isTRUE(all.equal(w, 1))) {
    "centred"
  } else {
    "partial"
  }
}

#' Coerce and validate a positive count.
#' @noRd
.as_count <- function(x, arg) {
  if (length(x) != 1L || !is.finite(x) || x < 1 ||
    abs(x - round(x)) > .Machine$double.eps^0.5) {
    cli_abort("{.arg {arg}} must be a single positive integer.")
  }
  as.integer(round(x))
}

#' Resolve a centring weight to a per-group vector in [0, 1].
#' @noRd
.resolve_centring <- function(centring, n_groups) {
  if (!is.numeric(centring) || anyNA(centring)) {
    cli_abort("{.arg centring} must be a numeric weight, with no missing values.")
  }
  if (any(centring < 0 | centring > 1)) {
    cli_abort(c(
      "{.arg centring} must lie in {.val [0, 1]}.",
      "x" = "Got value{?s} outside the unit interval: {.val {centring[centring < 0 | centring > 1]}}.",
      "i" = "0 is non-centred, 1 is centred; intermediate values partially centre."
    ))
  }
  if (length(centring) == 1L) {
    return(rep(centring, n_groups))
  }
  if (length(centring) != n_groups) {
    cli_abort(paste(
      "{.arg centring} must have length 1 or {n_groups} (one weight per group),",
      "but has length {length(centring)}."
    ))
  }
  centring
}

#' Resolve the group index to a data node, validating the range.
#' @noRd
.resolve_group_index <- function(group, n_groups) {
  if (inherits(group, "gretaR_array")) {
    return(group)
  }
  if (!is.numeric(group) || anyNA(group)) {
    cli_abort("{.arg group} must be an integer vector with no missing values.")
  }
  bad <- group[group < 1 | group > n_groups |
    abs(group - round(group)) > .Machine$double.eps^0.5]
  if (length(bad) > 0L) {
    cli_abort(c(
      "{.arg group} must hold integer group ids in the range 1 to {n_groups}.",
      "x" = "Found {length(bad)} id{?s} outside that range: {.val {unique(bad)}}."
    ))
  }
  as_data(matrix(as.numeric(round(group)), ncol = 1L))
}
