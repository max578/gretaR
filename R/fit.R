# fit.R — Unified output object for all gretaR inference methods
#
# All inference functions (mcmc, vi, opt, laplace, gretaR_glm) return
# a gretaR_fit object with a consistent structure modelled after
# bayesreml::atg_fit.

#' Create a gretaR_fit object
#'
#' @param draws Posterior draws (posterior::draws_array or NULL for MAP).
#' @param model The compiled gretaR_model object.
#' @param summary Posterior summary data frame.
#' @param convergence List with n_eff, rhat, divergences.
#' @param call_info List of original call arguments.
#' @param run_time Elapsed seconds.
#' @param method Character: "nuts", "hmc", "vi", "map", "laplace".
#' @param extra Named list of method-specific extras (e.g., VI elbo, MAP par).
#' @return A gretaR_fit S3 object.
#' @noRd
new_gretaR_fit <- function(draws = NULL, model = NULL, summary = NULL,
                           convergence = NULL, call_info = NULL,
                           run_time = NULL, method = "nuts",
                           extra = list()) {
  fit <- list(
    draws = draws,
    model = model,
    summary = summary,
    convergence = convergence,
    call_info = call_info,
    run_time = run_time,
    method = method
  )

  # Merge any extra fields

  for (nm in names(extra)) {
    fit[[nm]] <- extra[[nm]]
  }

  class(fit) <- "gretaR_fit"
  fit
}

#' Build convergence diagnostics from draws
#' @noRd
build_convergence <- function(draws, raw_divergences = NULL) {
  if (is.null(draws)) return(NULL)

  summ <- tryCatch(
    posterior::summarise_draws(draws),
    error = function(e) NULL
  )

  n_eff <- if (!is.null(summ) && "ess_bulk" %in% names(summ)) {
    stats::setNames(summ$ess_bulk, summ$variable)
  }

  rhat <- if (!is.null(summ) && "rhat" %in% names(summ)) {
    stats::setNames(summ$rhat, summ$variable)
  }

  n_divergences <- if (!is.null(raw_divergences)) {
    sum(raw_divergences, na.rm = TRUE)
  } else {
    n_div <- attr(draws, "divergences")
    if (!is.null(n_div)) sum(n_div, na.rm = TRUE) else 0L
  }

  list(
    n_eff = n_eff,
    rhat = rhat,
    max_rhat = if (!is.null(rhat)) max(rhat, na.rm = TRUE) else NA_real_,
    min_ess = if (!is.null(n_eff)) min(n_eff, na.rm = TRUE) else NA_real_,
    n_divergences = n_divergences
  )
}

# =============================================================================
# S3 methods for gretaR_fit
# =============================================================================

#' @title Print a gretaR Fit Object
#'
#' @description Display a concise summary of the model fit including method,
#'   parameter count, convergence diagnostics, and timing.
#'
#' @param x A `gretaR_fit` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.gretaR_fit <- function(x, ...) {
  method_label <- toupper(x$method)
  cat(sprintf("gretaR fit (%s)\n", method_label))
  cat(paste(rep("-", 50), collapse = ""), "\n")

  # Model info
  if (!is.null(x$model)) {
    cat(sprintf("  Parameters: %d (%d elements)\n",
                length(x$model$var_order), x$model$total_dim))
  }

  # Timing
  if (!is.null(x$run_time)) {
    cat(sprintf("  Run time: %.1f seconds\n", x$run_time))
  }

  # Convergence (MCMC/VI)
  if (!is.null(x$convergence)) {
    cv <- x$convergence
    if (!is.na(cv$max_rhat)) {
      rhat_ok <- cv$max_rhat < 1.05
      cat(sprintf("  Max R-hat: %.3f %s\n", cv$max_rhat,
                  if (rhat_ok) "" else "[WARNING: > 1.05]"))
    }
    if (!is.na(cv$min_ess)) {
      cat(sprintf("  Min ESS: %.0f\n", cv$min_ess))
    }
    if (cv$n_divergences > 0) {
      cat(sprintf("  Divergences: %d [WARNING]\n", cv$n_divergences))
    }
  }

  # Method-specific info
  if (x$method == "map") {
    cat("\n  MAP estimates:\n")
    if (!is.null(x$par)) {
      print(round(x$par, 4))
    }
  } else if (x$method == "laplace") {
    cat("\n  Posterior means (Laplace):\n")
    if (!is.null(x$par)) {
      print(round(x$par, 4))
    }
    if (!is.null(x$log_evidence)) {
      cat(sprintf("\n  Log marginal likelihood: %.2f\n", x$log_evidence))
    }
  } else if (x$method == "vi") {
    if (!is.null(x$elbo)) {
      cat(sprintf("  Final ELBO: %.2f\n", tail(x$elbo, 1)))
    }
  }

  # Posterior summary table (for MCMC/VI)
  if (!is.null(x$summary)) {
    cat("\nPosterior summary:\n")
    print(x$summary, n = min(nrow(x$summary), 20))
    if (nrow(x$summary) > 20) {
      cat(sprintf("  ... and %d more variables\n", nrow(x$summary) - 20))
    }
  }

  invisible(x)
}

#' @title Summarise a gretaR Fit Object
#'
#' @description Compute or display detailed posterior summary statistics
#'   and convergence diagnostics.
#'
#' @param object A `gretaR_fit` object.
#' @param ... Additional arguments passed to `posterior::summarise_draws()`.
#' @return A data frame of posterior summaries (from `posterior::summarise_draws()`),
#'   or a list for MAP/Laplace fits.
#' @export
summary.gretaR_fit <- function(object, ...) {
  if (!is.null(object$draws)) {
    posterior::summarise_draws(object$draws, ...)
  } else if (!is.null(object$par)) {
    list(
      method = object$method,
      par = object$par,
      convergence = object$convergence,
      log_evidence = object$log_evidence
    )
  } else {
    cat("No posterior draws available.\n")
    invisible(NULL)
  }
}

#' @title Extract Coefficients from a gretaR Fit
#'
#' @description Extract posterior means (for MCMC/VI) or MAP estimates as a
#'   named numeric vector.
#'
#' @param object A `gretaR_fit` object.
#' @param ... Ignored.
#' @return A named numeric vector of point estimates.
#' @export
coef.gretaR_fit <- function(object, ...) {
  if (!is.null(object$par)) {
    return(object$par)
  }
  if (!is.null(object$summary)) {
    stats::setNames(object$summary$mean, object$summary$variable)
  } else if (!is.null(object$draws)) {
    summ <- posterior::summarise_draws(object$draws)
    stats::setNames(summ$mean, summ$variable)
  } else {
    cli_alert_warning("No estimates available.")
    NULL
  }
}

#' @title Plot Diagnostics for a gretaR Fit
#'
#' @description Generate diagnostic plots for MCMC or VI posterior draws.
#'   Requires the `bayesplot` package.
#'
#' @param x A `gretaR_fit` object.
#' @param type Plot type: `"trace"` (default), `"density"`, `"pairs"`,
#'   `"rhat"`, `"neff"`.
#' @param ... Additional arguments passed to the bayesplot function.
#' @return A ggplot object.
#' @export
plot.gretaR_fit <- function(x, type = c("trace", "density", "pairs",
                                         "rhat", "neff"), ...) {
  type <- rlang::arg_match(type)

  if (is.null(x$draws)) {
    cli_abort("No posterior draws available for plotting.")
  }

  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cli_abort("Package {.pkg bayesplot} required for plotting.")
  }

  switch(type,
    trace = bayesplot::mcmc_trace(x$draws, ...),
    density = bayesplot::mcmc_dens_overlay(x$draws, ...),
    pairs = bayesplot::mcmc_pairs(x$draws, ...),
    rhat = {
      if (!is.null(x$convergence$rhat)) {
        bayesplot::mcmc_rhat(x$convergence$rhat, ...)
      } else {
        cli_abort("R-hat not available.")
      }
    },
    neff = {
      if (!is.null(x$convergence$n_eff)) {
        ratios <- x$convergence$n_eff / (dim(x$draws)[1] * dim(x$draws)[2])
        bayesplot::mcmc_neff(ratios, ...)
      } else {
        cli_abort("ESS not available.")
      }
    }
  )
}
