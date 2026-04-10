# smooth.R — mgcv-style smooth terms for gretaR
#
# Uses mgcv's smooth construction machinery (smoothCon + smooth2random)
# to build basis matrices and penalty structures, then injects them into
# the gretaR DAG as fixed + random effect components.
#
# Strategy: brms-style smooth2random decomposition
#   1. mgcv::smoothCon() constructs the basis and penalty
#   2. mgcv::smooth2random() decomposes into:
#      - Xf: fixed-effects basis (null-space, unpenalised)
#      - Zs: list of random-effects bases (penalised)
#   3. Each Zs block gets bs ~ N(0, sigma^2 * I) — identity penalty
#   4. sigma gets a half-Student-t or half-Cauchy hyperprior
#
# This avoids rank-deficient precision matrices entirely.

#' @title Process Smooth Terms from an mgcv-Style Formula
#'
#' @description Extract and construct smooth terms from a formula containing
#'   `s()`, `te()`, `ti()`, or `t2()` terms. Requires the `mgcv` package.
#'
#'   Uses the `smooth2random` decomposition (same approach as brms) to
#'   convert penalised smooth bases into fixed + random effect components
#'   suitable for HMC/NUTS sampling.
#'
#' @param formula A formula potentially containing smooth terms.
#' @param data A data frame with the covariates.
#' @param knots Optional named list of knot locations.
#'
#' @return A list with components:
#'   \describe{
#'     \item{fixed_formula}{The formula with smooth terms removed (parametric part).}
#'     \item{smooth_Xf}{Combined fixed-effects design matrix from all smooths (n x p_fixed).}
#'     \item{smooth_Zs}{List of random-effects design matrices (each n x p_j).}
#'     \item{smooth_info}{List of metadata for each smooth (label, type, rank, etc.).}
#'     \item{n_smooth_fixed}{Number of smooth fixed-effect columns.}
#'     \item{n_smooth_random}{List of dimensions per random-effect block.}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' library(mgcv)
#' dat <- data.frame(y = rnorm(100), x = rnorm(100))
#' sm <- process_smooths(y ~ s(x, k = 10), data = dat)
#' str(sm)
#' }
process_smooths <- function(formula, data, knots = NULL) {

  if (!requireNamespace("mgcv", quietly = TRUE)) {
    cli_abort(c(
      "Package {.pkg mgcv} is required for smooth terms (s, te, ti, t2).",
      "i" = "Install with {.code install.packages('mgcv')}."
    ))
  }

  # --- Parse the formula to extract smooth specs ---
  # Use mgcv's own formula interpretation
  gp <- mgcv::interpret.gam(formula)

  if (length(gp$smooth.spec) == 0) {
    cli_abort("No smooth terms found in formula. Use s(), te(), ti(), or t2().")
  }

  # --- Build the parametric (fixed) part of the formula ---
  # gp$pf is the parametric formula
  fixed_formula <- gp$pf

  # --- Construct smooth bases via mgcv ---
  n <- nrow(data)
  all_Xf <- list()       # fixed-effect columns per smooth
  all_Zs <- list()       # random-effect blocks per smooth
  smooth_info <- list()   # metadata

  for (i in seq_along(gp$smooth.spec)) {
    spec <- gp$smooth.spec[[i]]

    # Construct the smooth basis
    sm_list <- mgcv::smoothCon(
      spec, data = data, knots = knots,
      absorb.cons = TRUE,
      scale.penalty = TRUE,
      null.space.penalty = FALSE
    )

    for (j in seq_along(sm_list)) {
      sm <- sm_list[[j]]

      # Decompose into fixed + random via smooth2random
      # type=2 gives the most convenient decomposition
      re <- mgcv::smooth2random(sm, names(data), type = 2)

      # re$Xf: n x null.space.dim matrix (unpenalised fixed effects)
      # re$rand: list of random-effect matrices, each n x p_j
      # re$trans.U, re$trans.D: back-transformation matrices

      label <- sm$label %||% paste0("s_", i, "_", j)

      # Collect fixed-effect columns
      if (!is.null(re$Xf) && ncol(re$Xf) > 0) {
        colnames(re$Xf) <- paste0(label, "_f", seq_len(ncol(re$Xf)))
        all_Xf[[length(all_Xf) + 1]] <- re$Xf
      }

      # Collect random-effect blocks
      for (k in seq_along(re$rand)) {
        Z <- re$rand[[k]]
        if (is.list(Z)) Z <- Z[[1]]  # some smooth types nest further
        block_name <- paste0(label, "_z", k)
        colnames(Z) <- paste0(block_name, "_", seq_len(ncol(Z)))
        all_Zs[[length(all_Zs) + 1]] <- Z

        smooth_info[[length(smooth_info) + 1]] <- list(
          label = label,
          block = block_name,
          type = class(sm)[1],
          bs = sm$bs.dim,
          rank = sm$rank,
          null_space_dim = sm$null.space.dim,
          n_coef = ncol(Z),
          term = sm$term,
          by_level = sm$by.level
        )
      }
    }
  }

  # Combine all fixed-effect smooth columns into one matrix
  smooth_Xf <- if (length(all_Xf) > 0) do.call(cbind, all_Xf) else NULL

  list(
    fixed_formula = fixed_formula,
    smooth_Xf = smooth_Xf,
    smooth_Zs = all_Zs,
    smooth_info = smooth_info,
    n_smooth_fixed = if (!is.null(smooth_Xf)) ncol(smooth_Xf) else 0L,
    n_smooth_random = vapply(all_Zs, ncol, integer(1))
  )
}

#' @title Build gretaR Model Components from Smooth Terms
#'
#' @description Convert processed smooth terms into gretaR_array objects
#'   ready for inclusion in a model. Returns fixed-effect and random-effect
#'   contributions to the linear predictor.
#'
#' @param smooth_result Output from [process_smooths()].
#' @param prior_smooth_sd Prior on the smooth random-effect standard deviation.
#'   Default: `half_cauchy(2)`.
#'
#' @return A list with:
#'   \describe{
#'     \item{eta_smooth}{A gretaR_array: the smooth contribution to the linear predictor.}
#'     \item{target_vars}{Named list of gretaR_array variables to include in model().}
#'   }
#'
#' @noRd
build_smooth_gretaR <- function(smooth_result, prior_smooth_sd = NULL) {

  target_vars <- list()
  eta_terms <- list()

  # --- Fixed-effect smooth component ---
  if (!is.null(smooth_result$smooth_Xf) && smooth_result$n_smooth_fixed > 0) {
    Xf <- smooth_result$smooth_Xf
    p_f <- ncol(Xf)
    Xf_data <- as_data(Xf)
    beta_smooth <- normal(0, 5, dim = c(p_f, 1L))
    eta_terms[[length(eta_terms) + 1]] <- Xf_data %*% beta_smooth
    target_vars[["beta_smooth"]] <- beta_smooth
  }

  # --- Random-effect smooth components ---
  for (k in seq_along(smooth_result$smooth_Zs)) {
    Z <- smooth_result$smooth_Zs[[k]]
    p_k <- ncol(Z)
    info <- smooth_result$smooth_info[[k]]

    Z_data <- as_data(Z)

    # Variance component for this penalty block
    sd_name <- paste0("sd_", info$block)
    sd_k <- prior_smooth_sd %||% half_cauchy(2)
    target_vars[[sd_name]] <- sd_k

    # Non-centred random effects: z_raw ~ N(0, 1), alpha = sd * z_raw
    z_name <- paste0("z_", info$block)
    z_raw <- normal(0, 1, dim = c(p_k, 1L))
    target_vars[[z_name]] <- z_raw

    # Smooth contribution: Z %*% (sd * z_raw)
    alpha_k <- sd_k * z_raw
    eta_terms[[length(eta_terms) + 1]] <- Z_data %*% alpha_k
  }

  # Sum all smooth contributions
  eta_smooth <- NULL
  if (length(eta_terms) > 0) {
    eta_smooth <- eta_terms[[1]]
    for (i in seq_along(eta_terms)[-1]) {
      eta_smooth <- eta_smooth + eta_terms[[i]]
    }
  }

  list(
    eta_smooth = eta_smooth,
    target_vars = target_vars
  )
}
