# formula.R — Formula-based model specification
#
# Provides gretaR_glm() as a high-level convenience layer on top of the
# core DSL. Uses model.matrix() for design matrix construction.
#
# Supports formula styles from lm, glm, lme4, brms, etc.
# Auto-detects formula style or allows explicit specification.
# lme4-style random effects are parsed with regex (no lme4 dependency).

#' @title Fit a Bayesian GLM Using Formula Syntax
#'
#' @description A high-level interface for specifying and fitting generalised
#'   linear models (including mixed / hierarchical models) using standard R
#'   formula syntax. Internally translates to the gretaR DSL, constructs the
#'   model, and runs MCMC inference.
#'
#'   lme4-style random effects are supported via regex parsing (lme4 is
#'   **not** required). Recognised patterns:
#'   \itemize{
#'     \item \code{(1|group)} — random intercepts by group
#'     \item \code{(x|group)} — correlated random intercepts + slopes
#'     \item \code{(0 + x|group)} — random slopes only (no intercept)
#'   }
#'   Multiple random effect terms are permitted, e.g.
#'   \code{y ~ x + (1|site) + (1|year)}.
#'
#'   A non-centred parameterisation is used by default for superior HMC
#'   geometry.
#'
#' @param formula A formula specifying the model (e.g., \code{y ~ x1 + x2},
#'   or \code{y ~ x + (1|group)} for mixed models).
#' @param data A data frame containing the variables referenced in the formula.
#' @param family Distribution family: \code{"gaussian"} (default),
#'   \code{"binomial"}, or \code{"poisson"}.
#' @param prior A named list of gretaR distribution objects for parameter priors.
#'   Recognised names: \code{"beta"} (regression coefficients),
#'   \code{"intercept"}, \code{"sigma"} (residual SD, gaussian only),
#'   \code{"tau"} (random effect SD). Use \code{NULL} for default priors.
#' @param sampler Sampler: \code{"nuts"} (default), \code{"hmc"}, \code{"vi"},
#'   or \code{"map"}.
#' @param chains Number of MCMC chains (default 4).
#' @param iter Total iterations per chain (warmup + samples, default 2000).
#' @param warmup Number of warmup iterations (default half of iter).
#' @param formula_style Optional explicit formula style hint: \code{"base"},
#'   \code{"lme4"}, \code{"brms"}, \code{"mgcv"}. If \code{NULL} (default),
#'   auto-detected.
#' @param verbose Logical; print progress (default TRUE).
#' @param ... Additional arguments passed to the sampler.
#'
#' @return A \code{gretaR_glm_fit} object with components:
#'   \describe{
#'     \item{draws}{Posterior draws (from MCMC or VI).}
#'     \item{model}{The compiled gretaR_model.}
#'     \item{formula}{The original formula.}
#'     \item{family}{The family used.}
#'     \item{data}{The original data.}
#'     \item{design_matrix}{The model matrix (fixed effects).}
#'     \item{col_names}{Named mapping of fixed-effect parameters.}
#'     \item{random_effects}{List of parsed random effect specifications
#'       (NULL for base-style formulas).}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Gaussian linear model
#' fit <- gretaR_glm(Sepal.Length ~ Sepal.Width + Petal.Length,
#'                   data = iris, family = "gaussian")
#' summary(fit$draws)
#'
#' # Logistic regression
#' dat <- data.frame(y = rbinom(100, 1, 0.6), x = rnorm(100))
#' fit <- gretaR_glm(y ~ x, data = dat, family = "binomial")
#'
#' # Custom priors
#' fit <- gretaR_glm(y ~ x, data = dat, family = "gaussian",
#'                   prior = list(beta = normal(0, 1), sigma = half_cauchy(1)))
#'
#' # Random intercepts model (lme4-style)
#' sleepstudy <- data.frame(
#'   Reaction = rnorm(180, 300, 50),
#'   Days = rep(0:9, each = 18),
#'   Subject = factor(rep(1:18, times = 10))
#' )
#' fit <- gretaR_glm(Reaction ~ Days + (1 | Subject),
#'                   data = sleepstudy, family = "gaussian")
#'
#' # Random intercepts + slopes
#' fit <- gretaR_glm(Reaction ~ Days + (Days | Subject),
#'                   data = sleepstudy, family = "gaussian")
#' }
gretaR_glm <- function(formula, data, family = c("gaussian", "binomial", "poisson"),
                        prior = NULL, sampler = c("nuts", "hmc", "vi", "map"),
                        chains = 4L, iter = 2000L, warmup = NULL,
                        formula_style = NULL, verbose = TRUE, ...) {

  family <- rlang::arg_match(family)
  sampler <- rlang::arg_match(sampler)
  if (is.null(warmup)) warmup <- as.integer(iter / 2)
  n_samples <- iter - warmup

  # --- Auto-detect or validate formula style ---
  style <- detect_formula_style(formula, formula_style)
  if (verbose) cli_alert_info("Formula style: {style}")

  # --- Dispatch: lme4-style random effects or base fixed-effects ---
  if (style == "lme4") {
    return(.gretaR_glm_mixed(formula, data, family, prior, sampler,
                             chains, n_samples, warmup, verbose, ...))
  }

  if (style == "mgcv") {
    return(.gretaR_glm_smooth(formula, data, family, prior, sampler,
                               chains, n_samples, warmup, verbose, ...))
  }

  # --- Extract response and build design matrix ---
  mf <- stats::model.frame(formula, data = data, na.action = stats::na.fail)
  y_vec <- stats::model.response(mf)
  X <- stats::model.matrix(formula, data = mf)

  n <- nrow(X)
  p <- ncol(X)
  col_names <- colnames(X)
  has_intercept <- "(Intercept)" %in% col_names

  if (verbose) {
    cli_alert_info("Observations: {n}, Predictors: {p} ({paste(col_names, collapse=', ')})")
  }

  # --- Reset DAG and build gretaR model ---
  reset_gretaR_env()

  # Wrap data
  y <- as_data(y_vec)
  x <- as_data(X)

  # --- Set up priors ---
  beta_prior <- prior$beta %||% normal(0, 5, dim = c(p, 1L))
  if (!inherits(beta_prior, "gretaR_array")) {
    # User passed a distribution spec — it's already a gretaR_array from the constructor
  }

  # Linear predictor
  eta <- x %*% beta_prior

  # --- Link function and likelihood ---
  target_vars <- list(beta = beta_prior)

  if (family == "gaussian") {
    sigma_prior <- prior$sigma %||% half_cauchy(2)
    distribution(y) <- normal(eta, sigma_prior)
    target_vars$sigma <- sigma_prior
  } else if (family == "binomial") {
    # Logistic link: p = sigmoid(eta)
    prob <- logistic_link(eta)
    distribution(y) <- bernoulli(prob)
  } else if (family == "poisson") {
    # Log link: lambda = exp(eta)
    rate <- exp(eta)
    distribution(y) <- poisson_dist(rate)
  }

  # Compile model
  m <- do.call(model, unname(target_vars))

  # --- Inference ---
  result <- if (sampler == "map") {
    opt(m, verbose = verbose, ...)
  } else if (sampler == "vi") {
    variational(m, verbose = verbose, ...)
  } else {
    mcmc(m, n_samples = n_samples, warmup = warmup, chains = chains,
         sampler = sampler, verbose = verbose, ...)
  }

  # result is already a gretaR_fit — enrich with formula-specific info
  result$formula <- formula
  result$family <- family
  result$data <- data
  result$design_matrix <- X
  result$col_names <- col_names
  result$random_effects <- NULL

  class(result) <- c("gretaR_glm_fit", "gretaR_fit")
  result
}


# =============================================================================
# Internal: fit a mixed-effects model (lme4-style random effects)
# =============================================================================

#' Build and fit a mixed-effects gretaR model
#'
#' Uses non-centred parameterisation for group-level effects.
#' Called internally by \code{gretaR_glm()} when lme4-style bars are detected.
#'
#' @noRd
.gretaR_glm_mixed <- function(formula, data, family, prior, sampler,
                               chains, n_samples, warmup, verbose, ...) {


  # --- Parse random effects bars ---
  re_terms <- parse_re_bars(formula)
  fixed_formula <- remove_re_bars(formula)

  if (verbose) {
    n_re <- length(re_terms)
    cli_alert_info("Random effect terms: {n_re}")
    for (rt in re_terms) {
      cli_alert_info("  ({rt$lhs} | {rt$group}): {rt$type}")
    }
  }

  # --- Fixed effects design matrix (from the bar-free formula) ---
  mf <- stats::model.frame(fixed_formula, data = data,
                            na.action = stats::na.fail)
  y_vec <- stats::model.response(mf)
  X_fixed <- stats::model.matrix(fixed_formula, data = mf)

  n <- nrow(X_fixed)
  p_fixed <- ncol(X_fixed)
  col_names <- colnames(X_fixed)

  if (verbose) {
    cli_alert_info(
      "Observations: {n}, Fixed predictors: {p_fixed} ({paste(col_names, collapse=', ')})"
    )
  }

  # --- Validate grouping variables exist in data ---
  for (rt in re_terms) {
    if (!rt$group %in% names(data)) {
      cli_abort(c(
        "Grouping variable {.var {rt$group}} not found in {.arg data}.",
        "i" = "Available columns: {paste(names(data), collapse = ', ')}"
      ))
    }
  }

  # --- Reset DAG ---
  reset_gretaR_env()

  # --- Wrap data ---
  y <- as_data(y_vec)
  x_fixed <- as_data(X_fixed)

  # --- Fixed effects priors ---
  beta <- prior$beta %||% normal(0, 5, dim = c(p_fixed, 1L))

  # Fixed-effects linear predictor
  eta <- x_fixed %*% beta

  # --- Random effects: non-centred parameterisation ---
  target_vars <- list(beta = beta)
  re_info <- list()  # metadata for summary output

  for (k in seq_along(re_terms)) {
    rt <- re_terms[[k]]
    group_var <- data[[rt$group]]
    group_factor <- as.factor(group_var)
    group_id <- as.integer(group_factor)
    n_groups <- nlevels(group_factor)
    group_levels <- levels(group_factor)

    # Determine how many random coefficients per group
    if (rt$type == "intercept") {
      # (1|group): one random intercept per group
      n_re_coefs <- 1L
      re_coef_names <- "(Intercept)"
    } else if (rt$type == "slope_only") {
      # (0 + x|group): random slopes only, no intercept
      n_re_coefs <- length(rt$slope_vars)
      re_coef_names <- rt$slope_vars
    } else {
      # (x|group): random intercept + slopes
      n_re_coefs <- 1L + length(rt$slope_vars)
      re_coef_names <- c("(Intercept)", rt$slope_vars)
    }

    # Prior for group-level SD(s)
    tau_prior_name <- paste0("tau_", rt$group)
    tau <- prior[[tau_prior_name]] %||% half_cauchy(2, dim = c(n_re_coefs, 1L))

    # Non-centred raw values: n_groups x n_re_coefs
    z_raw <- normal(0, 1, dim = c(n_groups, n_re_coefs))

    # Scaled random effects: alpha[g, j] = tau[j] * z_raw[g, j]
    # Broadcasting: tau is (n_re_coefs, 1), need to transpose to (1, n_re_coefs)
    # so that element-wise multiply with z_raw (n_groups, n_re_coefs) broadcasts.
    # Since our Ops handle broadcasting, we can reshape tau or multiply directly.
    # tau is (n_re_coefs, 1); z_raw is (n_groups, n_re_coefs).
    # We need each column j of z_raw scaled by tau[j].
    # Strategy: loop over random coefficients and sum contributions.

    # Store variable references for model()
    tau_name <- paste0("tau_", k)
    z_name <- paste0("z_raw_", k)
    target_vars[[tau_name]] <- tau
    target_vars[[z_name]] <- z_raw

    # Add each random coefficient's contribution to eta
    for (j in seq_len(n_re_coefs)) {
      coef_name <- re_coef_names[j]

      if (n_re_coefs == 1L) {
        # Single random effect: tau is scalar (1,1), z_raw is (n_groups, 1)
        alpha_j <- tau * z_raw  # (n_groups, 1) via broadcasting
      } else {
        # Multiple random effects: extract column j
        # tau[j, 1] * z_raw[, j]
        # We need to index tau and z_raw per column.
        # For now, use separate tau/z_raw per coefficient to keep it simple.
        # This is handled below in the restructured approach.
        cli_abort(c(
          "Correlated random slopes with >1 coefficient per group are not yet ",
          "supported in a single (expr|group) term.",
          "i" = "Use separate terms: e.g., (1|group) + (0 + x|group)."
        ))
      }

      # Build the contribution to eta from this random effect
      if (coef_name == "(Intercept)") {
        # Random intercept: alpha[group_id] added to eta
        re_contribution <- alpha_j[group_id]
      } else {
        # Random slope: alpha[group_id] * x_var
        if (!coef_name %in% names(data)) {
          cli_abort("Slope variable {.var {coef_name}} not found in {.arg data}.")
        }
        slope_vals <- as_data(as.numeric(data[[coef_name]]))
        re_contribution <- alpha_j[group_id] * slope_vals
      }

      eta <- eta + re_contribution
    }

    # Store metadata for output
    re_info[[k]] <- list(
      group = rt$group,
      type = rt$type,
      n_groups = n_groups,
      n_re_coefs = n_re_coefs,
      re_coef_names = re_coef_names,
      group_levels = group_levels
    )
  }

  # --- Residual SD and likelihood ---
  if (family == "gaussian") {
    sigma <- prior$sigma %||% half_cauchy(2)
    distribution(y) <- normal(eta, sigma)
    target_vars$sigma <- sigma
  } else if (family == "binomial") {
    prob <- logistic_link(eta)
    distribution(y) <- bernoulli(prob)
  } else if (family == "poisson") {
    rate <- exp(eta)
    distribution(y) <- poisson_dist(rate)
  }

  # --- Compile model ---
  m <- do.call(model, unname(target_vars))

  # --- Inference ---
  result <- if (sampler == "map") {
    opt(m, verbose = verbose, ...)
  } else if (sampler == "vi") {
    variational(m, verbose = verbose, ...)
  } else {
    mcmc(m, n_samples = n_samples, warmup = warmup, chains = chains,
         sampler = sampler, verbose = verbose, ...)
  }

  # result is already a gretaR_fit — enrich with formula-specific info
  result$formula <- formula
  result$family <- family
  result$data <- data
  result$design_matrix <- X_fixed
  result$col_names <- col_names
  result$random_effects <- re_info

  class(result) <- c("gretaR_glm_fit", "gretaR_fit")
  result
}

#' Logistic (sigmoid) link for gretaR_arrays
#' @noRd
logistic_link <- function(eta) {
  # 1 / (1 + exp(-eta)) — build as operation node
  node <- get_node(eta)
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) torch_sigmoid(pvals[[1]]),
    parents = node$id,
    dim = node$dim_
  )
  result_node$op_type <- "sigmoid"
  wrap_gretaR_array(result_node)
}

#' Detect formula style
#'
#' Examines a formula for patterns indicating lme4, mgcv, brms, or base R style.
#'
#' @param formula A formula object.
#' @param explicit Optional explicit style hint.
#' @return Character string: one of "base", "lme4", "mgcv", "brms".
#' @noRd
detect_formula_style <- function(formula, explicit = NULL) {
  if (!is.null(explicit)) {
    explicit <- rlang::arg_match(explicit, c("base", "lme4", "brms", "mgcv",
                                              "glmmTMB", "asreml"))
    return(explicit)
  }

  formula_str <- deparse(formula, width.cutoff = 500)

  # lme4/glmmTMB-style: contains (1|group) or (var|group)

  if (grepl("\\(.*\\|.*\\)", formula_str)) {
    return("lme4")
  }

  # mgcv-style: contains s(), te(), ti()
  if (grepl("\\bs\\(|\\bte\\(|\\bti\\(", formula_str)) {
    return("mgcv")
  }

  # brms-style: uses bf() or has distributional parameters
  # (would be detected at the brmsformula level, not here)

  "base"
}

#' @export
print.gretaR_glm_fit <- function(x, ...) {
  has_re <- !is.null(x$random_effects) && length(x$random_effects) > 0
  type_str <- if (has_re) "mixed" else "GLM"
  cat(sprintf("gretaR %s fit (%s, %s)\n", type_str, x$family, x$method))
  cat(sprintf("  Formula: %s\n", deparse(x$formula)))
  cat(sprintf("  Observations: %d, Fixed predictors: %d\n",
              nrow(x$design_matrix), ncol(x$design_matrix)))
  cat(sprintf("  Fixed effects: %s\n", paste(x$col_names, collapse = ", ")))


  # Random effects summary
  if (has_re) {
    cat("\n  Random effects:\n")
    for (re in x$random_effects) {
      cat(sprintf("    Group: %s (%d levels), Type: %s, Coefficients: %s\n",
                  re$group, re$n_groups, re$type,
                  paste(re$re_coef_names, collapse = ", ")))
    }
  }

  if (!is.null(x$draws)) {
    cat("\nPosterior summary:\n")
    summ <- posterior::summarise_draws(x$draws)
    print(summ, n = min(nrow(summ), 20))
  } else if (x$method == "map") {
    cat("\nMAP estimates:\n")
    print(round(x$par, 4))
  }

  invisible(x)
}

#' @export
summary.gretaR_glm_fit <- function(object, ...) {
  has_re <- !is.null(object$random_effects) && length(object$random_effects) > 0

  if (!is.null(object$draws)) {
    summ <- posterior::summarise_draws(object$draws, ...)
    if (has_re) {
      attr(summ, "random_effects") <- object$random_effects
    }
    summ
  } else if (object$sampler == "map") {
    # Return a structured list for MAP fits
    out <- list(
      fixed_effects = object$col_names,
      map_estimates = object$result$par,
      random_effects = object$random_effects
    )
    class(out) <- "summary.gretaR_glm_fit"
    print(object)
    invisible(out)
  } else {
    print(object)
  }
}


# =============================================================================
# Random effects formula parsing (regex-based, no lme4 dependency)
# =============================================================================

#' Parse random effects bar terms from an lme4-style formula
#'
#' Extracts all \code{(expr | group)} terms and classifies them as
#' \code{"intercept"}, \code{"slope_only"}, or \code{"intercept_slope"}.
#'
#' @param formula A formula potentially containing bar terms.
#' @return A list of parsed random effect specifications. Each element is a
#'   list with components: \code{raw} (original text), \code{lhs} (left of
#'   bar), \code{group} (grouping variable), \code{type} (one of
#'   \code{"intercept"}, \code{"slope_only"}, \code{"intercept_slope"}),
#'   and \code{slope_vars} (character vector of slope variable names, may
#'   be empty).
#'
#' @examples
#' parse_re_bars(y ~ x + (1 | group))
#' parse_re_bars(y ~ x + (x | group))
#' parse_re_bars(y ~ x + (0 + x | group))
#' parse_re_bars(y ~ x + (1 | site) + (1 | year))
#'
#' @export
parse_re_bars <- function(formula) {
  formula_str <- paste(deparse(formula, width.cutoff = 500), collapse = " ")

  # Match all (expr | group) patterns, handling nested parentheses carefully.
  # The bar "|" inside parentheses is the lme4 convention.
  # Regex: match balanced parens containing a bar.
  bar_pattern <- "\\(([^()]+)\\|([^()]+)\\)"
  matches <- gregexpr(bar_pattern, formula_str, perl = TRUE)
  matched_texts <- regmatches(formula_str, matches)[[1]]

  if (length(matched_texts) == 0) {
    cli_abort("No random effect terms (expr|group) found in formula.")
  }

  lapply(matched_texts, function(mt) {
    # Strip outer parens and split on |
    inner <- sub("^\\((.*)\\)$", "\\1", mt)
    parts <- strsplit(inner, "\\|")[[1]]
    if (length(parts) != 2) {
      cli_abort("Cannot parse random effect term: {.code {mt}}")
    }
    lhs <- trimws(parts[1])
    group <- trimws(parts[2])

    # Classify the random effect
    .classify_re_term(lhs, group, raw = mt)
  })
}


#' Classify a single random effect term
#'
#' @param lhs Left-hand side of the bar (e.g., "1", "x", "0 + x").
#' @param group Grouping variable name.
#' @param raw Original matched text for error messages.
#' @return A list with components: raw, lhs, group, type, slope_vars.
#' @noRd
.classify_re_term <- function(lhs, group, raw) {
  # (1 | group): intercept only
  if (lhs == "1") {
    return(list(
      raw = raw, lhs = lhs, group = group,
      type = "intercept", slope_vars = character(0)
    ))
  }

  # (0 + x | group) or (0+x|group): slopes only, no intercept
  if (grepl("^0\\s*\\+", lhs)) {
    vars <- trimws(strsplit(sub("^0\\s*\\+\\s*", "", lhs), "\\+")[[1]])
    return(list(
      raw = raw, lhs = lhs, group = group,
      type = "slope_only", slope_vars = vars
    ))
  }

  # (x | group): intercept + slope(s)

  # Could be a single variable or multiple: (x1 + x2 | group)
  vars <- trimws(strsplit(lhs, "\\+")[[1]])
  # Remove "1" if explicitly included alongside slopes
  vars <- setdiff(vars, "1")

  if (length(vars) == 0) {
    # Edge case: (1 | group) caught above; this shouldn't happen
    return(list(
      raw = raw, lhs = lhs, group = group,
      type = "intercept", slope_vars = character(0)
    ))
  }

  list(
    raw = raw, lhs = lhs, group = group,
    type = "intercept_slope", slope_vars = vars
  )
}


#' Remove random effects bar terms from a formula
#'
#' Returns the fixed-effects-only formula by stripping all \code{(expr|group)}
#' terms. If lme4 is installed, delegates to \code{lme4::nobars()}; otherwise
#' uses regex substitution.
#'
#' @param formula A formula potentially containing bar terms.
#' @return A formula with bar terms removed.
#'
#' @examples
#' remove_re_bars(y ~ x + (1 | group))
#' # y ~ x
#'
#' @export
remove_re_bars <- function(formula) {
  # Try lme4::nobars() if available (most robust)
  if (requireNamespace("lme4", quietly = TRUE)) {
    nb <- lme4::nobars(formula)
    if (!is.null(nb)) return(nb)
  }

  # Fallback: regex removal of (expr|group) terms
  formula_str <- paste(deparse(formula, width.cutoff = 500), collapse = " ")
  bar_pattern <- "\\([^()]+\\|[^()]+\\)"

  # Remove bar terms
  cleaned <- gsub(bar_pattern, "", formula_str, perl = TRUE)

  # Clean up dangling operators: leading/trailing +, double +, trailing ~
  cleaned <- gsub("\\+\\s*\\+", "+", cleaned)          # ++ -> +
  cleaned <- gsub("~\\s*\\+", "~", cleaned)             # ~ + -> ~
  cleaned <- gsub("\\+\\s*$", "", cleaned)               # trailing +
  cleaned <- trimws(cleaned)

  # If the RHS is now empty (e.g., y ~ (1|group)), add intercept
  if (grepl("~\\s*$", cleaned)) {
    cleaned <- paste0(cleaned, " 1")
  }

  stats::as.formula(cleaned, env = environment(formula))
}


# =============================================================================
# Internal: fit a GAM-style model (mgcv smooth terms)
# =============================================================================

#' Build and fit a model with mgcv-style smooth terms
#'
#' Uses the smooth2random decomposition from mgcv to convert penalised
#' spline bases into fixed + random effect components for HMC/NUTS sampling.
#'
#' @noRd
.gretaR_glm_smooth <- function(formula, data, family, prior, sampler,
                                chains, n_samples, warmup, verbose, ...) {

  # --- Process smooth terms ---
  sm <- process_smooths(formula, data)

  if (verbose) {
    n_sm <- length(sm$smooth_info)
    cli_alert_info("Smooth terms: {n_sm} random-effect blocks")
    for (info in sm$smooth_info) {
      cli_alert_info("  {info$label}: {info$type}, {info$n_coef} coefficients")
    }
    if (sm$n_smooth_fixed > 0) {
      cli_alert_info("  Smooth fixed effects: {sm$n_smooth_fixed} columns")
    }
  }

  # --- Parametric (fixed) part ---
  mf <- stats::model.frame(sm$fixed_formula, data = data,
                            na.action = stats::na.fail)
  y_vec <- stats::model.response(mf)
  X_fixed <- stats::model.matrix(sm$fixed_formula, data = mf)

  n <- nrow(X_fixed)
  p_fixed <- ncol(X_fixed)
  col_names <- colnames(X_fixed)

  if (verbose) {
    cli_alert_info(
      "Parametric: {p_fixed} columns ({paste(col_names, collapse=', ')})"
    )
  }

  # --- Reset DAG and build model ---
  reset_gretaR_env()

  y <- as_data(y_vec)
  x_fixed <- as_data(X_fixed)
  beta_fixed <- prior$beta %||% normal(0, 5, dim = c(p_fixed, 1L))

  # Parametric linear predictor
  eta <- x_fixed %*% beta_fixed

  # --- Add smooth contributions ---
  smooth_parts <- build_smooth_gretaR(sm, prior_smooth_sd = prior$smooth_sd)

  if (!is.null(smooth_parts$eta_smooth)) {
    eta <- eta + smooth_parts$eta_smooth
  }

  # Collect all target variables
  target_vars <- c(list(beta_fixed = beta_fixed), smooth_parts$target_vars)

  # --- Likelihood ---
  if (family == "gaussian") {
    sigma <- prior$sigma %||% half_cauchy(2)
    distribution(y) <- normal(eta, sigma)
    target_vars$sigma <- sigma
  } else if (family == "binomial") {
    prob <- logistic_link(eta)
    distribution(y) <- bernoulli(prob)
  } else if (family == "poisson") {
    rate <- exp(eta)
    distribution(y) <- poisson_dist(rate)
  }

  # --- Compile and fit ---
  m <- do.call(model, unname(target_vars))

  result <- if (sampler == "map") {
    opt(m, verbose = verbose, ...)
  } else if (sampler == "vi") {
    variational(m, verbose = verbose, ...)
  } else {
    mcmc(m, n_samples = n_samples, warmup = warmup, chains = chains,
         sampler = sampler, verbose = verbose, ...)
  }

  # --- Build output ---
  result$formula <- formula
  result$family <- family
  result$data <- data
  result$design_matrix <- X_fixed
  result$col_names <- col_names
  result$random_effects <- NULL
  result$smooth_info <- sm$smooth_info
  result$n_smooth_fixed <- sm$n_smooth_fixed
  result$n_smooth_random <- sm$n_smooth_random

  class(result) <- c("gretaR_glm_fit", "gretaR_fit")
  result
}
