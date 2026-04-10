# optimise.R — MAP estimation and Laplace approximation

#' @title Find the Maximum A Posteriori (MAP) Estimate
#'
#' @description Optimise the log-joint density using gradient-based methods
#'   to find the posterior mode. Uses the Adam optimiser via torch.
#'
#' @param model A `gretaR_model` object created by [model()].
#' @param max_iter Maximum number of optimisation iterations (default 2000).
#' @param learning_rate Adam learning rate (default 0.01).
#' @param tolerance Convergence tolerance on relative change in log-prob
#'   (default 1e-6).
#' @param init Optional initial values (numeric vector in unconstrained space).
#' @param verbose Logical; print progress (default TRUE).
#' @param backend Inference backend: \code{"torch"} (default) or \code{"stan"}.
#'
#' @return A `gretaR_fit` object (method = "map") with components:
#'   \describe{
#'     \item{par}{Named numeric vector of MAP estimates (constrained space).}
#'     \item{par_unconstrained}{Numeric vector of MAP in unconstrained space.}
#'     \item{log_prob}{Log-joint density at the MAP.}
#'     \item{convergence}{List with convergence info.}
#'     \item{iterations}{Number of iterations used.}
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
#' fit <- opt(m)
#' fit$par
#' }
opt <- function(model, max_iter = 2000L, learning_rate = 0.01,
                tolerance = 1e-6, init = NULL, verbose = TRUE,
                backend = c("torch", "stan")) {

  backend <- match.arg(backend)
  if (backend == "stan") {
    return(stan_optimize(model, verbose = verbose))
  }

  n_params <- model$total_dim

  # Initialise
  if (!is.null(init)) {
    theta <- torch_tensor(init, dtype = model$dtype, requires_grad = TRUE)
  } else {
    # Use gradient ascent to find a good start
    init_vec <- find_initial_values(model, n_params, n_steps = 100,
                                    learning_rate = 0.1)
    theta <- torch_tensor(init_vec, dtype = model$dtype, requires_grad = TRUE)
  }

  # Adam optimiser (maximising log_prob → minimising -log_prob)
  optimizer <- optim_adam(list(theta), lr = learning_rate)

  prev_lp <- -Inf
  converged <- FALSE
  best_lp <- -Inf
  best_theta <- as.numeric(theta$detach()$cpu())

  if (verbose) {
    cli_alert_info("MAP optimisation: {n_params} parameters, max {max_iter} iterations")
  }

  for (iter in seq_len(max_iter)) {
    optimizer$zero_grad()

    # Compute -log_prob (we minimise)
    lp <- log_prob(model, theta)
    neg_lp <- -lp
    neg_lp$backward()

    optimizer$step()

    current_lp <- lp$item()

    # Track best
    if (current_lp > best_lp) {
      best_lp <- current_lp
      best_theta <- as.numeric(theta$detach()$cpu())
    }

    # Check convergence
    if (is.finite(prev_lp) && is.finite(current_lp)) {
      rel_change <- abs(current_lp - prev_lp) / (abs(prev_lp) + 1e-8)
      if (rel_change < tolerance) {
        converged <- TRUE
        if (verbose) cli_alert_success("Converged at iteration {iter} (log-prob: {round(current_lp, 2)})")
        break
      }
    }

    prev_lp <- current_lp

    if (verbose && iter %% 500 == 0) {
      cli_alert_info("  Iteration {iter}: log-prob = {round(current_lp, 2)}")
    }
  }

  if (!converged && verbose) {
    cli_alert_warning("Did not converge in {max_iter} iterations (log-prob: {round(best_lp, 2)})")
  }

  # Convert to constrained space with names
  theta_t <- torch_tensor(best_theta, dtype = model$dtype)
  constrained <- unconstrained_to_constrained(model, theta_t)
  constrained_vec <- as.numeric(constrained$cpu())

  # Build named vector
  param_names <- make_param_names(model)
  names(constrained_vec) <- param_names

  new_gretaR_fit(
    draws = NULL,
    model = model,
    summary = NULL,
    convergence = list(
      n_eff = NULL, rhat = NULL, max_rhat = NA_real_,
      min_ess = NA_real_, n_divergences = 0L,
      converged = converged
    ),
    call_info = list(max_iter = max_iter, learning_rate = learning_rate,
                     tolerance = tolerance),
    run_time = NULL,
    method = "map",
    extra = list(
      par = constrained_vec,
      par_unconstrained = best_theta,
      log_prob = best_lp,
      iterations = min(iter, max_iter)
    )
  )
}

#' @title Laplace Approximation
#'
#' @description Approximate the posterior distribution using a multivariate
#'   normal centred at the MAP estimate with covariance equal to the inverse
#'   of the negative Hessian of the log-joint density.
#'
#' @param model A `gretaR_model` object.
#' @param map_fit Optional output from [opt()]. If NULL, MAP is computed first.
#' @param ... Additional arguments passed to [opt()] if `map_fit` is NULL.
#'
#' @return A list with components:
#'   \describe{
#'     \item{mean}{Named numeric vector of posterior means (constrained).}
#'     \item{mean_unconstrained}{Posterior means in unconstrained space.}
#'     \item{covariance}{Posterior covariance matrix (unconstrained space).}
#'     \item{sd}{Named numeric vector of posterior standard deviations (unconstrained).}
#'     \item{log_evidence}{Approximate log marginal likelihood.}
#'     \item{map}{The MAP fit used.}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' mu <- normal(0, 10)
#' y <- as_data(rnorm(50, 3, 1))
#' distribution(y) <- normal(mu, 1)
#' m <- model(mu)
#' la <- laplace(m)
#' la$mean
#' la$sd
#' }
laplace <- function(model, map_fit = NULL, ...) {

  if (is.null(map_fit)) {
    map_fit <- opt(model, ...)
  }

  n_params <- model$total_dim
  theta_map <- map_fit$par_unconstrained

  # Compute the Hessian at the MAP via finite differences on the gradient
  hessian <- compute_hessian(model, theta_map)

  # Negative Hessian = precision matrix
  neg_hessian <- -hessian

  # Covariance = inverse of negative Hessian
  cov_mat <- tryCatch({
    solve(neg_hessian)
  }, error = function(e) {
    cli_alert_warning("Hessian not invertible; using diagonal approximation.")
    diag(1 / pmax(diag(neg_hessian), 1e-6))
  })

  # Ensure positive definiteness
  eig <- eigen(cov_mat, symmetric = TRUE)
  if (any(eig$values <= 0)) {
    cli_alert_warning("Posterior covariance not positive definite; clamping eigenvalues.")
    eig$values <- pmax(eig$values, 1e-6)
    cov_mat <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  }

  # Posterior SDs in unconstrained space
  sd_vec <- sqrt(diag(cov_mat))
  param_names <- make_param_names(model)
  names(sd_vec) <- param_names

  # Log marginal likelihood approximation (Laplace)
  # log p(y) ≈ log p(y, theta_MAP) + (d/2)*log(2*pi) + 0.5*log|Sigma|
  log_det_cov <- determinant(cov_mat, logarithm = TRUE)$modulus[1]
  log_evidence <- map_fit$log_prob + 0.5 * n_params * log(2 * pi) + 0.5 * log_det_cov

  new_gretaR_fit(
    draws = NULL,
    model = model,
    summary = NULL,
    convergence = NULL,
    call_info = list(method = "laplace"),
    run_time = NULL,
    method = "laplace",
    extra = list(
      par = map_fit$par,
      par_unconstrained = theta_map,
      covariance = cov_mat,
      sd = sd_vec,
      log_evidence = log_evidence,
      map = map_fit
    )
  )
}

#' Compute Hessian via finite differences on the gradient
#' @noRd
compute_hessian <- function(model, theta_vec, eps = 1e-4) {
  n <- length(theta_vec)
  H <- matrix(0, n, n)

  for (i in seq_len(n)) {
    theta_plus <- theta_vec
    theta_minus <- theta_vec
    theta_plus[i] <- theta_plus[i] + eps
    theta_minus[i] <- theta_minus[i] - eps

    grad_plus <- eval_grad(model, theta_plus)$grad
    grad_minus <- eval_grad(model, theta_minus)$grad

    H[, i] <- (grad_plus - grad_minus) / (2 * eps)
  }

  # Symmetrise
  0.5 * (H + t(H))
}
