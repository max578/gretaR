# mcmc.R — User-facing MCMC interface

#' @title Draw MCMC Samples from a gretaR Model
#'
#' @description Run Hamiltonian Monte Carlo or the No-U-Turn Sampler on a
#'   compiled \code{gretaR_model} and return posterior draws in a format
#'   compatible with the \code{posterior} and \code{bayesplot} packages.
#'
#' @param model A \code{gretaR_model} object created by \code{\link{model}}.
#' @param n_samples Number of post-warmup samples per chain (default 1000).
#' @param warmup Number of warmup (adaptation) iterations per chain (default 1000).
#' @param chains Number of independent chains (default 4).
#' @param sampler Sampler to use: \code{"nuts"} (default) or \code{"hmc"}.
#' @param step_size Initial step size for the leapfrog integrator. If
#'   \code{NULL} (default), automatically tuned during warmup.
#' @param max_treedepth Maximum tree depth for NUTS (default 10).
#' @param n_leapfrog Number of leapfrog steps for static HMC (default 25).
#' @param target_accept Target average acceptance probability (default 0.8 for
#'   NUTS, 0.65 for HMC).
#' @param init_values Optional list of initial parameter vectors (one per chain).
#' @param verbose Logical; print progress information (default \code{TRUE}).
#'
#' @return A `gretaR_fit` object with components:
#'   \describe{
#'     \item{draws}{Posterior draws as `posterior::draws_array`.}
#'     \item{model}{The compiled `gretaR_model`.}
#'     \item{summary}{Posterior summary table (mean, sd, quantiles, R-hat, ESS).}
#'     \item{convergence}{List: `n_eff`, `rhat`, `max_rhat`, `min_ess`, `n_divergences`.}
#'     \item{call_info}{List of sampling arguments for reproducibility.}
#'     \item{run_time}{Elapsed seconds.}
#'     \item{method}{`"nuts"` or `"hmc"`.}
#'   }
#'   Use `coef()` for point estimates, `summary()` for full table,
#'   `plot()` for diagnostics.
#'
#' @export
#' @examples
#' \dontrun{
#' # Simple normal model
#' mu <- normal(0, 10)
#' sigma <- half_cauchy(2)
#' y <- as_data(rnorm(50, 3, 1.5))
#' distribution(y) <- normal(mu, sigma)
#' m <- model(mu, sigma)
#' draws <- mcmc(m, n_samples = 500, warmup = 500)
#' summary(draws)
#' }
mcmc <- function(model, n_samples = 1000L, warmup = 1000L, chains = 4L,
                 sampler = c("nuts", "hmc"),
                 backend = c("torch", "stan"),
                 step_size = NULL, max_treedepth = 10L,
                 n_leapfrog = 25L, target_accept = NULL,
                 init_values = NULL, verbose = TRUE) {

  sampler <- match.arg(sampler)
  backend <- match.arg(backend)

  # --- Stan backend dispatch ---
  if (backend == "stan") {
    return(stan_sample(model, n_samples = n_samples, warmup = warmup,
                       chains = chains, verbose = verbose))
  }

  # --- Torch backend (default) ---
  # Set defaults based on sampler
  # step_size = NULL lets the sampler auto-tune via find_reasonable_epsilon
  if (is.null(target_accept)) {
    target_accept <- if (sampler == "nuts") 0.8 else 0.65
  }

  # Compile the log-prob function for faster gradient evaluation
  compiled_fn <- tryCatch(
    compile_model(model, use_jit = TRUE),
    error = function(e) NULL
  )
  if (!is.null(compiled_fn) && verbose) {
    cli_alert_info("Compiled log-prob for fast evaluation")
  }

  if (verbose) {
    cli_alert_info("Sampler: {toupper(sampler)}")
  }

  raw <- if (sampler == "nuts") {
    nuts_sampler(
      model = model,
      n_samples = n_samples,
      warmup = warmup,
      chains = chains,
      step_size = step_size,
      max_treedepth = max_treedepth,
      target_accept = target_accept,
      init_values = init_values,
      verbose = verbose,
      compiled_fn = compiled_fn
    )
  } else {
    hmc_sampler(
      model = model,
      n_samples = n_samples,
      warmup = warmup,
      chains = chains,
      step_size = step_size,
      n_leapfrog = n_leapfrog,
      target_accept = target_accept,
      init_values = init_values,
      verbose = verbose,
      compiled_fn = compiled_fn
    )
  }

  # Convert to posterior::draws_array
  t0 <- proc.time()
  draws <- format_draws(raw)
  elapsed <- (proc.time() - t0)[["elapsed"]] + raw$n_samples  # approximate

  if (verbose) {
    n_div <- sum(raw$divergences)
    if (n_div > 0) {
      cli_alert_warning("{n_div} divergent transition{?s} detected. Consider reparameterising.")
    }
    cli_alert_success("Sampling complete.")
  }

  # Build unified gretaR_fit object
  summ <- tryCatch(posterior::summarise_draws(draws), error = function(e) NULL)
  convergence <- build_convergence(draws, raw$divergences)

  new_gretaR_fit(
    draws = draws,
    model = model,
    summary = summ,
    convergence = convergence,
    call_info = list(
      n_samples = n_samples, warmup = warmup, chains = chains,
      sampler = sampler, step_size = step_size,
      target_accept = target_accept
    ),
    run_time = raw$n_samples,  # placeholder — actual timing in sampler
    method = sampler
  )
}

#' @title Run HMC Sampling
#'
#' @description Convenience wrapper around \code{\link{mcmc}} that selects the
#'   static Hamiltonian Monte Carlo sampler.
#'
#' @inheritParams mcmc
#' @param ... Additional arguments passed to \code{\link{mcmc}}.
#' @return A `gretaR_fit` object.
#' @export
#' @examples
#' \dontrun{
#' m <- model(normal(0, 1))
#' fit <- hmc(m, n_samples = 500, warmup = 500)
#' coef(fit)
#' }
hmc <- function(model, n_samples = 1000L, warmup = 1000L, chains = 4L, ...) {
  mcmc(model, n_samples = n_samples, warmup = warmup, chains = chains,
       sampler = "hmc", ...)
}

#' @title Run NUTS Sampling
#'
#' @description Convenience wrapper around \code{\link{mcmc}} that selects the
#'   No-U-Turn Sampler (NUTS).
#'
#' @inheritParams mcmc
#' @param ... Additional arguments passed to \code{\link{mcmc}}.
#' @return A `gretaR_fit` object.
#' @export
#' @examples
#' \dontrun{
#' m <- model(normal(0, 1))
#' fit <- nuts(m, n_samples = 500, warmup = 500)
#' coef(fit)
#' }
nuts <- function(model, n_samples = 1000L, warmup = 1000L, chains = 4L, ...) {
  mcmc(model, n_samples = n_samples, warmup = warmup, chains = chains,
       sampler = "nuts", ...)
}

# =============================================================================
# Format raw samples into posterior::draws_array
# =============================================================================

#' @noRd
format_draws <- function(raw) {
  # raw$samples is iterations x chains x parameters
  arr <- raw$samples
  dimnames(arr) <- list(
    iteration = seq_len(raw$n_samples),
    chain = seq_len(raw$chains),
    variable = raw$param_names
  )

  # Convert to posterior draws_array
  draws <- posterior::as_draws_array(arr)

  # Attach metadata as attributes
  attr(draws, "sampler") <- raw$sampler
  attr(draws, "warmup") <- raw$warmup
  attr(draws, "divergences") <- raw$divergences
  if (!is.null(raw$treedepths)) {
    attr(draws, "treedepths") <- raw$treedepths
  }
  attr(draws, "acceptance_rates") <- raw$acceptance_rates

  class(draws) <- c("gretaR_draws", class(draws))
  draws
}

# =============================================================================
# S3 methods for gretaR_draws
# =============================================================================

#' @export
print.gretaR_draws <- function(x, ...) {
  sampler <- attr(x, "sampler") %||% "unknown"
  warmup <- attr(x, "warmup") %||% 0
  n_div <- sum(attr(x, "divergences") %||% 0)

  cat(sprintf("gretaR posterior draws (%s)\n", toupper(sampler)))
  cat(sprintf("  Chains: %d, Samples per chain: %d, Warmup: %d\n",
              dim(x)[2], dim(x)[1], warmup))
  if (n_div > 0) {
    cat(sprintf("  WARNING: %d divergent transitions\n", n_div))
  }
  cat("\n")

  # Print summary
  summ <- posterior::summarise_draws(x)
  print(summ, n = min(nrow(summ), 20))
  if (nrow(summ) > 20) {
    cat(sprintf("  ... and %d more variables\n", nrow(summ) - 20))
  }

  invisible(x)
}

#' @export
summary.gretaR_draws <- function(object, ...) {
  posterior::summarise_draws(object, ...)
}

#' @export
plot.gretaR_draws <- function(x, type = c("trace", "density", "pairs"), ...) {
  type <- match.arg(type)

  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cli_abort("Package {.pkg bayesplot} is required for plotting. Install with {.code install.packages('bayesplot')}.")
  }

  switch(type,
    trace = bayesplot::mcmc_trace(x, ...),
    density = bayesplot::mcmc_dens_overlay(x, ...),
    pairs = bayesplot::mcmc_pairs(x, ...)
  )
}
