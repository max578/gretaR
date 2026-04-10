# inference_vi.R — Automatic Differentiation Variational Inference (ADVI)
#
# Implements mean-field and full-rank Gaussian variational inference
# following Kucukelbir et al. (2017).
#
# The variational family is a multivariate normal in unconstrained space:
#   q(theta) = N(mu, Sigma)
# where Sigma is diagonal (mean-field) or full (full-rank).
#
# The ELBO is optimised using the reparameterisation trick and stochastic
# gradient ascent via Adam.

#' @title Variational Inference (ADVI)
#'
#' @description Fit a model using Automatic Differentiation Variational
#'   Inference. Approximates the posterior with a multivariate Gaussian
#'   in unconstrained parameter space.
#'
#' @param model A `gretaR_model` object created by [model()].
#' @param n_samples Number of Monte Carlo samples per ELBO gradient estimate
#'   (default 1).
#' @param max_iter Maximum number of optimisation iterations (default 5000).
#' @param learning_rate Adam learning rate (default 0.01).
#' @param tolerance Convergence tolerance on relative ELBO change (default 1e-4).
#' @param method Variational family: `"meanfield"` (default) or `"fullrank"`.
#' @param init_from_map Logical; initialise from MAP estimate (default TRUE).
#' @param verbose Logical; print progress (default TRUE).
#'
#' @return A `gretaR_vi` object with components:
#'   \describe{
#'     \item{mean}{Named vector of posterior means (constrained space).}
#'     \item{mean_unconstrained}{Posterior means in unconstrained space.}
#'     \item{sd}{Named vector of posterior SDs (unconstrained space).}
#'     \item{covariance}{Posterior covariance matrix (unconstrained, fullrank only).}
#'     \item{elbo}{Vector of ELBO values per iteration.}
#'     \item{draws}{A `posterior::draws_array` of samples from the variational posterior.}
#'     \item{converged}{Logical; did the optimiser converge?}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' mu <- normal(0, 10)
#' sigma <- half_cauchy(2)
#' y <- as_data(rnorm(50, 3, 1.5))
#' distribution(y) <- normal(mu, sigma)
#' m <- model(mu, sigma)
#' fit <- variational(m)
#' coef(fit)
#' }
variational <- function(model, n_samples = 1L, max_iter = 5000L,
               learning_rate = 0.01, tolerance = 1e-4,
               method = c("meanfield", "fullrank"),
               init_from_map = TRUE, verbose = TRUE) {

  method <- match.arg(method)
  n_params <- model$total_dim

  if (verbose) {
    cli_alert_info("ADVI ({method}): {n_params} parameters, max {max_iter} iterations")
  }

  # Initialise variational parameters
  if (init_from_map) {
    map_fit <- opt(model, verbose = FALSE)
    mu_init <- map_fit$par_unconstrained
  } else {
    mu_init <- find_initial_values(model, n_params, n_steps = 100)
  }

  # Variational parameters (learnable)
  mu <- torch_tensor(mu_init, dtype = model$dtype, requires_grad = TRUE)
  # Log standard deviations (ensures positivity)
  log_sigma <- torch_tensor(rep(-1, n_params), dtype = model$dtype,
                            requires_grad = TRUE)

  if (method == "fullrank") {
    # Lower-triangular Cholesky factor of covariance
    # Initialise as diagonal
    L_flat_size <- n_params * (n_params + 1) / 2
    L_flat <- torch_zeros(L_flat_size, dtype = model$dtype,
                          requires_grad = TRUE)
    # Set diagonal to log_sigma initial values
    idx <- cumsum(seq_len(n_params))
    L_init <- as.numeric(L_flat$detach()$cpu())
    L_init[idx] <- -1  # log(exp(-1)) = exp(-1) ≈ 0.37
    L_flat <- torch_tensor(L_init, dtype = model$dtype, requires_grad = TRUE)
    opt_params <- list(mu, L_flat)
  } else {
    opt_params <- list(mu, log_sigma)
  }

  optimizer <- optim_adam(opt_params, lr = learning_rate)

  elbo_history <- numeric(0)
  converged <- FALSE

  for (iter in seq_len(max_iter)) {
    optimizer$zero_grad()

    # Compute ELBO via reparameterisation trick
    elbo <- torch_zeros(1, dtype = model$dtype)

    for (s in seq_len(n_samples)) {
      # Sample epsilon ~ N(0, I)
      epsilon <- torch_randn(n_params, dtype = model$dtype)

      if (method == "meanfield") {
        sigma <- torch_exp(log_sigma)
        # Reparameterised sample: theta = mu + sigma * epsilon
        theta <- mu + sigma * epsilon
        # Entropy of q: 0.5 * d * (1 + log(2*pi)) + sum(log_sigma)
        entropy <- 0.5 * n_params * 1.8378771 + torch_sum(log_sigma)
      } else {
        # Full-rank: build lower-triangular L from flat vector
        L <- build_lower_triangular(L_flat, n_params)
        theta <- mu + torch_mv(L, epsilon)
        # Entropy: 0.5 * d * (1 + log(2*pi)) + sum(log(diag(L)))
        diag_L <- torch_diag(L)
        entropy <- 0.5 * n_params * 1.8378771 + torch_sum(torch_log(torch_abs(diag_L) + 1e-10))
      }

      # Evaluate log joint at the sample
      lp <- log_prob(model, theta)
      elbo <- elbo + (lp + entropy)
    }

    elbo <- elbo / n_samples

    # Minimise negative ELBO
    neg_elbo <- -elbo
    neg_elbo$backward()
    optimizer$step()

    elbo_val <- elbo$item()
    elbo_history <- c(elbo_history, elbo_val)

    # Check convergence (using median of recent window)
    if (iter > 100) {
      recent <- tail(elbo_history, 50)
      older <- elbo_history[max(1, iter - 100):(iter - 50)]
      if (length(older) > 0) {
        median_recent <- median(recent)
        median_older <- median(older)
        rel_change <- abs(median_recent - median_older) / (abs(median_older) + 1e-8)
        if (rel_change < tolerance) {
          converged <- TRUE
          if (verbose) cli_alert_success("Converged at iteration {iter} (ELBO: {round(elbo_val, 2)})")
          break
        }
      }
    }

    if (verbose && iter %% 1000 == 0) {
      cli_alert_info("  Iteration {iter}: ELBO = {round(elbo_val, 2)}")
    }
  }

  if (!converged && verbose) {
    cli_alert_warning("Did not converge in {max_iter} iterations (ELBO: {round(tail(elbo_history, 1), 2)})")
  }

  # Extract final variational parameters
  mu_final <- as.numeric(mu$detach()$cpu())

  if (method == "meanfield") {
    sigma_final <- as.numeric(torch_exp(log_sigma)$detach()$cpu())
    cov_mat <- diag(sigma_final^2)
  } else {
    L_final <- build_lower_triangular(L_flat$detach(), n_params)
    L_mat <- as.matrix(L_final$cpu())
    cov_mat <- L_mat %*% t(L_mat)
    sigma_final <- sqrt(diag(cov_mat))
  }

  # Convert means to constrained space
  theta_t <- torch_tensor(mu_final, dtype = model$dtype)
  constrained <- unconstrained_to_constrained(model, theta_t)
  constrained_vec <- as.numeric(constrained$cpu())

  param_names <- make_param_names(model)
  names(constrained_vec) <- param_names
  names(sigma_final) <- param_names

  # Generate posterior draws from the variational approximation
  draws <- generate_vi_draws(model, mu_final, cov_mat, param_names,
                             n_draws = 1000L, n_chains = 4L)

  summ <- tryCatch(posterior::summarise_draws(draws), error = function(e) NULL)
  convergence <- build_convergence(draws)

  new_gretaR_fit(
    draws = draws,
    model = model,
    summary = summ,
    convergence = convergence,
    call_info = list(method = method, max_iter = max_iter,
                     learning_rate = learning_rate),
    run_time = NULL,
    method = "vi",
    extra = list(
      par = constrained_vec,
      par_unconstrained = mu_final,
      sd = sigma_final,
      covariance = cov_mat,
      elbo = elbo_history,
      vi_method = method,
      converged = converged,
      iterations = min(iter, max_iter)
    )
  )
}

#' Build a lower-triangular matrix from a flat vector
#' @noRd
build_lower_triangular <- function(flat, d) {
  L <- torch_zeros(c(d, d), dtype = flat$dtype)
  idx <- 1L
  for (i in seq_len(d)) {
    for (j in seq_len(i)) {
      if (i == j) {
        # Diagonal: use exp to ensure positivity
        L[i, j] <- torch_exp(flat[idx])
      } else {
        L[i, j] <- flat[idx]
      }
      idx <- idx + 1L
    }
  }
  L
}

#' Generate posterior draws from the variational approximation
#' @noRd
generate_vi_draws <- function(model, mu_vec, cov_mat, param_names,
                              n_draws = 1000L, n_chains = 4L) {
  n_params <- length(mu_vec)

  # Cholesky of covariance
  L <- tryCatch(
    t(chol(cov_mat)),
    error = function(e) {
      # Fall back to diagonal
      diag(sqrt(pmax(diag(cov_mat), 1e-6)))
    }
  )

  samples <- array(NA_real_, dim = c(n_draws, n_chains, n_params))

  for (chain in seq_len(n_chains)) {
    eps <- matrix(rnorm(n_draws * n_params), nrow = n_draws)
    # theta = mu + L %*% eps^T → each row is a sample
    theta_unconstrained <- sweep(eps %*% t(L), 2, mu_vec, "+")

    for (i in seq_len(n_draws)) {
      theta_t <- torch_tensor(theta_unconstrained[i, ], dtype = model$dtype)
      constrained <- unconstrained_to_constrained(model, theta_t)
      samples[i, chain, ] <- as.numeric(constrained$cpu())
    }
  }

  dimnames(samples) <- list(
    iteration = seq_len(n_draws),
    chain = seq_len(n_chains),
    variable = param_names
  )

  posterior::as_draws_array(samples)
}

# Legacy print/summary removed — handled by gretaR_fit S3 methods

# Legacy print.gretaR_vi and summary.gretaR_vi removed.
# All output is now handled by gretaR_fit S3 methods in fit.R.
