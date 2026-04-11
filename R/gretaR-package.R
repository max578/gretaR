#' @title gretaR: Bayesian Statistical Modelling with Torch
#'
#' @description
#' A probabilistic programming package for Bayesian statistical modelling in R
#' using native R syntax. Models are defined interactively with `gretaR_array`
#' objects, then compiled to torch tensors for GPU-accelerated HMC and NUTS
#' inference.
#'
#' @section Core workflow:
#' 1. Wrap observed data with [as_data()]
#' 2. Define priors using distribution functions (e.g., [normal()], [gamma_dist()])
#' 3. Define the model structure using standard R operations
#' 4. Assign a likelihood with [distribution()]
#' 5. Create a model with [model()]
#' 6. Draw samples with [mcmc()]
#'
#' @import R6
#' @import torch
#' @importFrom posterior as_draws_array as_draws_df summarise_draws
#' @importFrom cli cli_alert_info cli_alert_success cli_alert_warning cli_abort
#' @importFrom cli cli_progress_bar cli_progress_update cli_progress_done
#' @importFrom stats runif rnorm dnorm dt var model.frame model.matrix
#'   model.response na.fail median
#' @importFrom utils tail
#' @importFrom methods as
#'
#' @keywords internal
"_PACKAGE"

# Package-level environment for storing global state
.gretaR_env <- new.env(parent = emptyenv())
.gretaR_env$dag <- NULL
.gretaR_env$distributions <- list()
.gretaR_env$node_counter <- 0L
