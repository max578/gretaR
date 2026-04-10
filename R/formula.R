# formula.R — Formula-based model specification
#
# Provides gretaR_glm() as a high-level convenience layer on top of the
# core DSL. Uses model.matrix() for design matrix construction.
#
# Supports formula styles from lm, glm, lme4, brms, etc.
# Auto-detects formula style or allows explicit specification.

#' @title Fit a Bayesian GLM Using Formula Syntax
#'
#' @description A high-level interface for specifying and fitting generalised
#'   linear models using standard R formula syntax. Internally translates to
#'   the gretaR DSL, constructs the model, and runs MCMC inference.
#'
#' @param formula A formula specifying the model (e.g., `y ~ x1 + x2`).
#' @param data A data frame containing the variables referenced in the formula.
#' @param family Distribution family: `"gaussian"` (default), `"binomial"`,
#'   or `"poisson"`.
#' @param prior A named list of gretaR distribution objects for parameter priors.
#'   Recognised names: `"beta"` (regression coefficients), `"intercept"`,
#'   `"sigma"` (residual SD, gaussian only). Use `NULL` for default priors.
#' @param sampler Sampler: `"nuts"` (default), `"hmc"`, `"vi"`, or `"map"`.
#' @param chains Number of MCMC chains (default 4).
#' @param iter Total iterations per chain (warmup + samples, default 2000).
#' @param warmup Number of warmup iterations (default half of iter).
#' @param formula_style Optional explicit formula style hint: `"base"`, `"lme4"`,
#'   `"brms"`, `"mgcv"`. If NULL (default), auto-detected.
#' @param verbose Logical; print progress (default TRUE).
#' @param ... Additional arguments passed to the sampler.
#'
#' @return A `gretaR_glm_fit` object with components:
#'   \describe{
#'     \item{draws}{Posterior draws (from MCMC or VI).}
#'     \item{model}{The compiled gretaR_model.}
#'     \item{formula}{The original formula.}
#'     \item{family}{The family used.}
#'     \item{data}{The original data.}
#'     \item{design_matrix}{The model matrix.}
#'     \item{param_names}{Named mapping of parameters.}
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
#' }
gretaR_glm <- function(formula, data, family = c("gaussian", "binomial", "poisson"),
                        prior = NULL, sampler = c("nuts", "hmc", "vi", "map"),
                        chains = 4L, iter = 2000L, warmup = NULL,
                        formula_style = NULL, verbose = TRUE, ...) {

  family <- match.arg(family)
  sampler <- match.arg(sampler)
  if (is.null(warmup)) warmup <- as.integer(iter / 2)
  n_samples <- iter - warmup

  # --- Auto-detect or validate formula style ---
  style <- detect_formula_style(formula, formula_style)
  if (verbose) cli_alert_info("Formula style: {style}")

  # --- Handle formula based on style ---
  if (style == "lme4") {
    cli_abort(c(
      "Random effects formulas (lme4-style) are not yet supported in {.fn gretaR_glm}.",
      "i" = "Use the gretaR DSL directly for hierarchical models.",
      "i" = "See {.code vignette('getting-started', package = 'gretaR')}."
    ))
  }

  if (style == "mgcv") {
    cli_abort(c(
      "Smooth term formulas (mgcv-style) are not yet supported in {.fn gretaR_glm}.",
      "i" = "Use the gretaR DSL directly for non-linear models."
    ))
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
    vi(m, verbose = verbose, ...)
  } else {
    mcmc(m, n_samples = n_samples, warmup = warmup, chains = chains,
         sampler = sampler, verbose = verbose, ...)
  }

  # --- Build output ---
  draws <- if (sampler == "map") NULL
           else if (sampler == "vi") result$draws
           else result

  fit <- list(
    draws = draws,
    model = m,
    result = result,
    formula = formula,
    family = family,
    data = data,
    design_matrix = X,
    col_names = col_names,
    sampler = sampler
  )

  class(fit) <- "gretaR_glm_fit"
  fit
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
    explicit <- match.arg(explicit, c("base", "lme4", "brms", "mgcv",
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
  cat(sprintf("gretaR GLM fit (%s, %s)\n", x$family, x$sampler))
  cat(sprintf("  Formula: %s\n", deparse(x$formula)))
  cat(sprintf("  Observations: %d, Predictors: %d\n",
              nrow(x$design_matrix), ncol(x$design_matrix)))
  cat(sprintf("  Coefficients: %s\n", paste(x$col_names, collapse = ", ")))

  if (!is.null(x$draws)) {
    cat("\nPosterior summary:\n")
    summ <- posterior::summarise_draws(x$draws)
    print(summ, n = min(nrow(summ), 20))
  } else if (x$sampler == "map") {
    cat("\nMAP estimates:\n")
    print(round(x$result$par, 4))
  }

  invisible(x)
}

#' @export
summary.gretaR_glm_fit <- function(object, ...) {
  if (!is.null(object$draws)) {
    posterior::summarise_draws(object$draws, ...)
  } else {
    print(object)
  }
}
