#!/usr/bin/env Rscript
# =============================================================================
# gretaR Validation & Benchmarking Suite
# Compares: gretaR vs cmdstanr vs greta
# 10 benchmark models with parameter recovery and timing
# =============================================================================

library(torch)
library(R6)
library(cli)
library(posterior)

# Source gretaR
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) source(f)

# Check available backends
has_cmdstanr <- requireNamespace("cmdstanr", quietly = TRUE) &&
  tryCatch({ cmdstanr::cmdstan_path(); TRUE }, error = function(e) FALSE)
has_greta <- requireNamespace("greta", quietly = TRUE)

cat("=== gretaR Validation & Benchmarking Suite ===\n")
cat("Backends: gretaR=TRUE, cmdstanr=", has_cmdstanr, ", greta=", has_greta, "\n\n")

results <- list()

# Helper: run a gretaR model and time it
run_gretaR <- function(setup_fn, n_samples = 500, warmup = 500, chains = 2) {
  reset_gretaR_env()
  model_obj <- setup_fn()
  t0 <- proc.time()
  draws <- mcmc(model_obj, n_samples = n_samples, warmup = warmup,
                chains = chains, sampler = "nuts", verbose = FALSE)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  summ <- posterior::summarise_draws(draws)
  list(summary = summ, time = elapsed, draws = draws)
}

# Helper: run a Stan model via cmdstanr
run_stan <- function(stan_code, data_list, n_samples = 500, warmup = 500, chains = 2) {
  if (!has_cmdstanr) return(NULL)
  tmp <- tempfile(fileext = ".stan")
  writeLines(stan_code, tmp)
  mod <- cmdstanr::cmdstan_model(tmp, quiet = TRUE)
  t0 <- proc.time()
  fit <- mod$sample(data = data_list, iter_sampling = n_samples,
                    iter_warmup = warmup, chains = chains,
                    refresh = 0, show_messages = FALSE)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  summ <- fit$summary()
  list(summary = summ, time = elapsed)
}

# Helper: run a greta model
run_greta <- function(setup_fn, n_samples = 500, warmup = 500, chains = 2) {
  if (!has_greta) return(NULL)
  tryCatch({
    model_and_targets <- setup_fn()
    t0 <- proc.time()
    draws <- greta::mcmc(model_and_targets$model, n_samples = n_samples,
                         warmup = warmup, chains = chains, verbose = FALSE)
    elapsed <- (proc.time() - t0)[["elapsed"]]
    # Convert coda to posterior
    draws_arr <- posterior::as_draws_array(draws)
    summ <- posterior::summarise_draws(draws_arr)
    list(summary = summ, time = elapsed)
  }, error = function(e) {
    cat("  greta error:", e$message, "\n")
    NULL
  })
}

# =============================================================================
# Benchmark 1: Normal mean (known variance)
# =============================================================================
cat("--- Benchmark 1: Normal mean (known variance) ---\n")
set.seed(42)
true_mu <- 5; n_obs <- 100
y1 <- rnorm(n_obs, true_mu, 1)

b1_gretaR <- run_gretaR(function() {
  mu <- normal(0, 10)
  y <- as_data(y1)
  distribution(y) <- normal(mu, 1)
  model(mu)
})
cat("  gretaR: mu=", round(b1_gretaR$summary$mean[1], 3),
    " time=", round(b1_gretaR$time, 1), "s\n")

b1_stan <- run_stan("
data { int N; vector[N] y; }
parameters { real mu; }
model { mu ~ normal(0, 10); y ~ normal(mu, 1); }
", list(N = n_obs, y = y1))
if (!is.null(b1_stan)) {
  mu_row <- b1_stan$summary[b1_stan$summary$variable == "mu", ]
  cat("  Stan:   mu=", round(mu_row$mean, 3), " time=", round(b1_stan$time, 1), "s\n")
}

b1_greta <- run_greta(function() {
  mu <- greta::normal(0, 10)
  y <- greta::as_data(y1)
  greta::distribution(y) <- greta::normal(mu, 1)
  list(model = greta::model(mu))
})
if (!is.null(b1_greta)) {
  cat("  greta:  mu=", round(b1_greta$summary$mean[1], 3),
      " time=", round(b1_greta$time, 1), "s\n")
}

results[["1_normal_mean"]] <- list(
  true = c(mu = true_mu),
  gretaR = b1_gretaR, stan = b1_stan, greta = b1_greta
)

# =============================================================================
# Benchmark 2: Normal mean + sigma
# =============================================================================
cat("\n--- Benchmark 2: Normal mean + sigma ---\n")
set.seed(42)
true_mu2 <- 3; true_sigma2 <- 1.5
y2 <- rnorm(100, true_mu2, true_sigma2)

b2_gretaR <- run_gretaR(function() {
  mu <- normal(0, 10)
  sigma <- half_cauchy(5)
  y <- as_data(y2)
  distribution(y) <- normal(mu, sigma)
  model(mu, sigma)
})
cat("  gretaR: mu=", round(b2_gretaR$summary$mean[1], 3),
    " sigma=", round(b2_gretaR$summary$mean[2], 3),
    " time=", round(b2_gretaR$time, 1), "s\n")

b2_stan <- run_stan("
data { int N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; }
model { mu ~ normal(0, 10); sigma ~ cauchy(0, 5); y ~ normal(mu, sigma); }
", list(N = 100, y = y2))
if (!is.null(b2_stan)) {
  cat("  Stan:   mu=", round(b2_stan$summary$mean[b2_stan$summary$variable == "mu"], 3),
      " sigma=", round(b2_stan$summary$mean[b2_stan$summary$variable == "sigma"], 3),
      " time=", round(b2_stan$time, 1), "s\n")
}

results[["2_normal_mu_sigma"]] <- list(
  true = c(mu = true_mu2, sigma = true_sigma2),
  gretaR = b2_gretaR, stan = b2_stan
)

# =============================================================================
# Benchmark 3: Linear regression
# =============================================================================
cat("\n--- Benchmark 3: Linear regression ---\n")
set.seed(123)
n3 <- 100; x3 <- rnorm(n3)
y3 <- 2 + 3 * x3 + rnorm(n3, 0, 0.5)

b3_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 10)
  beta <- normal(0, 10)
  sigma <- half_cauchy(2)
  x <- as_data(x3); y <- as_data(y3)
  mu <- alpha + beta * x
  distribution(y) <- normal(mu, sigma)
  model(alpha, beta, sigma)
})
s3 <- b3_gretaR$summary
cat("  gretaR: alpha=", round(s3$mean[1], 3), " beta=", round(s3$mean[2], 3),
    " sigma=", round(s3$mean[3], 3), " time=", round(b3_gretaR$time, 1), "s\n")

b3_stan <- run_stan("
data { int N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ cauchy(0, 2);
  y ~ normal(alpha + beta * x, sigma);
}
", list(N = n3, x = x3, y = y3))
if (!is.null(b3_stan)) {
  cat("  Stan:   alpha=", round(b3_stan$summary$mean[b3_stan$summary$variable == "alpha"], 3),
      " beta=", round(b3_stan$summary$mean[b3_stan$summary$variable == "beta"], 3),
      " sigma=", round(b3_stan$summary$mean[b3_stan$summary$variable == "sigma"], 3),
      " time=", round(b3_stan$time, 1), "s\n")
}

results[["3_linear_regression"]] <- list(
  true = c(alpha = 2, beta = 3, sigma = 0.5),
  gretaR = b3_gretaR, stan = b3_stan
)

# =============================================================================
# Benchmark 4: Multiple regression (3 predictors)
# =============================================================================
cat("\n--- Benchmark 4: Multiple regression (3 predictors) ---\n")
set.seed(42)
n4 <- 150
X4 <- matrix(rnorm(n4 * 3), ncol = 3)
true_beta4 <- c(1, -2, 0.5)
y4 <- 3 + X4 %*% true_beta4 + rnorm(n4, 0, 1)

b4_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 10)
  beta <- normal(0, 5, dim = c(3, 1))
  sigma <- half_cauchy(3)
  X <- as_data(X4); y <- as_data(y4)
  mu <- alpha + X %*% beta
  distribution(y) <- normal(mu, sigma)
  model(alpha, beta, sigma)
})
s4 <- b4_gretaR$summary
cat("  gretaR: alpha=", round(s4$mean[1], 3),
    " beta=", paste(round(s4$mean[2:4], 3), collapse = ","),
    " time=", round(b4_gretaR$time, 1), "s\n")

results[["4_multiple_regression"]] <- list(
  true = c(alpha = 3, beta1 = 1, beta2 = -2, beta3 = 0.5, sigma = 1),
  gretaR = b4_gretaR
)

# =============================================================================
# Benchmark 5: Logistic regression
# =============================================================================
cat("\n--- Benchmark 5: Logistic regression ---\n")
set.seed(42)
n5 <- 200; x5 <- rnorm(n5)
p5 <- plogis(0.5 + 1.2 * x5)
y5 <- rbinom(n5, 1, p5)

b5_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 5)
  beta <- normal(0, 5)
  x <- as_data(x5); y <- as_data(y5)
  p <- logistic_link(alpha + beta * x)
  distribution(y) <- bernoulli(p)
  model(alpha, beta)
})
s5 <- b5_gretaR$summary
cat("  gretaR: alpha=", round(s5$mean[1], 3), " beta=", round(s5$mean[2], 3),
    " time=", round(b5_gretaR$time, 1), "s\n")

b5_stan <- run_stan("
data { int N; vector[N] x; array[N] int y; }
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 5); beta ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + beta * x);
}
", list(N = n5, x = x5, y = as.integer(y5)))
if (!is.null(b5_stan)) {
  cat("  Stan:   alpha=", round(b5_stan$summary$mean[b5_stan$summary$variable == "alpha"], 3),
      " beta=", round(b5_stan$summary$mean[b5_stan$summary$variable == "beta"], 3),
      " time=", round(b5_stan$time, 1), "s\n")
}

results[["5_logistic_regression"]] <- list(
  true = c(alpha = 0.5, beta = 1.2),
  gretaR = b5_gretaR, stan = b5_stan
)

# =============================================================================
# Benchmark 6: Poisson regression
# =============================================================================
cat("\n--- Benchmark 6: Poisson regression ---\n")
set.seed(42)
n6 <- 150; x6 <- rnorm(n6)
y6 <- rpois(n6, exp(1 + 0.5 * x6))

b6_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 5)
  beta <- normal(0, 5)
  x <- as_data(x6); y <- as_data(y6)
  rate <- exp(alpha + beta * x)
  distribution(y) <- poisson_dist(rate)
  model(alpha, beta)
})
s6 <- b6_gretaR$summary
cat("  gretaR: alpha=", round(s6$mean[1], 3), " beta=", round(s6$mean[2], 3),
    " time=", round(b6_gretaR$time, 1), "s\n")

results[["6_poisson_regression"]] <- list(
  true = c(alpha = 1, beta = 0.5),
  gretaR = b6_gretaR
)

# =============================================================================
# Benchmark 7: Gamma regression
# =============================================================================
cat("\n--- Benchmark 7: Gamma GLM ---\n")
set.seed(42)
n7 <- 100; x7 <- rnorm(n7)
shape7 <- 2
rate7 <- shape7 / exp(1 + 0.5 * x7)
y7 <- rgamma(n7, shape = shape7, rate = rate7)

b7_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 5)
  beta <- normal(0, 5)
  shape <- half_cauchy(3)
  x <- as_data(x7); y <- as_data(y7)
  mu <- exp(alpha + beta * x)
  rate <- shape / mu
  distribution(y) <- gamma_dist(shape, rate)
  model(alpha, beta, shape)
})
s7 <- b7_gretaR$summary
cat("  gretaR: alpha=", round(s7$mean[1], 3), " beta=", round(s7$mean[2], 3),
    " shape=", round(s7$mean[3], 3), " time=", round(b7_gretaR$time, 1), "s\n")

results[["7_gamma_glm"]] <- list(
  true = c(alpha = 1, beta = 0.5, shape = 2),
  gretaR = b7_gretaR
)

# =============================================================================
# Benchmark 8: Robust regression (Student-t errors)
# =============================================================================
cat("\n--- Benchmark 8: Robust regression (Student-t) ---\n")
set.seed(42)
n8 <- 100; x8 <- rnorm(n8)
y8 <- 2 + 3 * x8 + rt(n8, df = 3)  # heavy-tailed errors

b8_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 10)
  beta <- normal(0, 10)
  sigma <- half_cauchy(5)
  x <- as_data(x8); y <- as_data(y8)
  mu <- alpha + beta * x
  distribution(y) <- student_t(df = 3, mu = mu, sigma = sigma)
  model(alpha, beta, sigma)
})
s8 <- b8_gretaR$summary
cat("  gretaR: alpha=", round(s8$mean[1], 3), " beta=", round(s8$mean[2], 3),
    " sigma=", round(s8$mean[3], 3), " time=", round(b8_gretaR$time, 1), "s\n")

results[["8_robust_regression"]] <- list(
  true = c(alpha = 2, beta = 3),
  gretaR = b8_gretaR
)

# =============================================================================
# Benchmark 9: Beta regression
# =============================================================================
cat("\n--- Benchmark 9: Beta regression ---\n")
set.seed(42)
n9 <- 100; x9 <- rnorm(n9)
mu9 <- plogis(0.5 + 0.8 * x9)
phi9 <- 10
y9 <- rbeta(n9, mu9 * phi9, (1 - mu9) * phi9)

b9_gretaR <- run_gretaR(function() {
  alpha <- normal(0, 5)
  beta_coef <- normal(0, 5)
  phi <- half_cauchy(10)
  x <- as_data(x9); y <- as_data(y9)
  mu <- logistic_link(alpha + beta_coef * x)
  a <- mu * phi; b <- (1 - mu) * phi
  distribution(y) <- beta_dist(a, b)
  model(alpha, beta_coef, phi)
})
s9 <- b9_gretaR$summary
cat("  gretaR: alpha=", round(s9$mean[1], 3), " beta=", round(s9$mean[2], 3),
    " phi=", round(s9$mean[3], 3), " time=", round(b9_gretaR$time, 1), "s\n")

results[["9_beta_regression"]] <- list(
  true = c(alpha = 0.5, beta = 0.8, phi = 10),
  gretaR = b9_gretaR
)

# =============================================================================
# Benchmark 10: Hierarchical model (random intercepts)
# =============================================================================
cat("\n--- Benchmark 10: Hierarchical model (5 groups) ---\n")
set.seed(42)
n_groups <- 5; n_per <- 20; n10 <- n_groups * n_per
gid <- rep(1:n_groups, each = n_per)
true_mu10 <- 5; true_tau10 <- 2; true_sigma10 <- 1
true_alpha10 <- rnorm(n_groups, true_mu10, true_tau10)
y10 <- rnorm(n10, true_alpha10[gid], true_sigma10)

b10_gretaR <- run_gretaR(function() {
  # Non-centred parameterisation for better geometry
  mu <- normal(0, 10)
  tau <- half_cauchy(5)
  alpha_raw <- normal(0, 1, dim = c(n_groups, 1))
  sigma <- half_cauchy(5)
  alpha <- mu + tau * alpha_raw
  y <- as_data(y10)
  fitted_vals <- alpha[gid]
  distribution(y) <- normal(fitted_vals, sigma)
  model(mu, tau, sigma, alpha_raw)
})
s10 <- b10_gretaR$summary
cat("  gretaR: mu=", round(s10$mean[1], 3), " tau=", round(s10$mean[2], 3),
    " sigma=", round(s10$mean[3], 3),
    " time=", round(b10_gretaR$time, 1), "s\n")

b10_stan <- run_stan("
data { int N; int J; array[N] int gid; vector[N] y; }
parameters {
  real mu; real<lower=0> tau; real<lower=0> sigma;
  vector[J] alpha_raw;
}
transformed parameters {
  vector[J] alpha = mu + tau * alpha_raw;
}
model {
  mu ~ normal(0, 10); tau ~ cauchy(0, 5); sigma ~ cauchy(0, 5);
  alpha_raw ~ std_normal();
  y ~ normal(alpha[gid], sigma);
}
", list(N = n10, J = n_groups, gid = gid, y = y10))
if (!is.null(b10_stan)) {
  cat("  Stan:   mu=", round(b10_stan$summary$mean[b10_stan$summary$variable == "mu"], 3),
      " tau=", round(b10_stan$summary$mean[b10_stan$summary$variable == "tau"], 3),
      " sigma=", round(b10_stan$summary$mean[b10_stan$summary$variable == "sigma"], 3),
      " time=", round(b10_stan$time, 1), "s\n")
}

results[["10_hierarchical"]] <- list(
  true = c(mu = true_mu10, tau = true_tau10, sigma = true_sigma10),
  gretaR = b10_gretaR, stan = b10_stan
)

# =============================================================================
# Summary table
# =============================================================================
cat("\n\n=== SUMMARY ===\n\n")
cat(sprintf("%-30s  %-10s  %-10s  %-10s\n", "Model", "gretaR(s)", "Stan(s)", "greta(s)"))
cat(paste(rep("-", 65), collapse = ""), "\n")

for (name in names(results)) {
  r <- results[[name]]
  gt <- round(r$gretaR$time, 1)
  st <- if (!is.null(r$stan)) round(r$stan$time, 1) else "N/A"
  gr <- if (!is.null(r$greta)) round(r$greta$time, 1) else "N/A"
  cat(sprintf("%-30s  %-10s  %-10s  %-10s\n", name, gt, st, gr))
}

cat("\nValidation complete.\n")
