#!/usr/bin/env Rscript
# =============================================================================
# gretaR Backend Benchmark: Torch vs Stan (direct cmdstanr) vs greta
# Models from GLMM_Model_Formulations_Reference_v5_corrected.xlsx
# =============================================================================

library(torch)
library(R6)
library(cli)
library(posterior)
library(cmdstanr)

for (f in list.files("R", pattern="[.]R$", full.names=TRUE)) source(f)

has_greta <- requireNamespace("greta", quietly = TRUE)

cat("=== Backend Benchmark: gretaR (torch) vs Stan vs greta ===\n")
cat("Date:", format(Sys.time()), "\n\n")

results <- list()
N_SAMPLES <- 500L
WARMUP <- 500L
CHAINS <- 2L

# Helper: time a gretaR torch model
run_torch <- function(setup_fn) {
  reset_gretaR_env()
  m <- setup_fn()
  t0 <- proc.time()
  fit <- mcmc(m, n_samples=N_SAMPLES, warmup=WARMUP, chains=CHAINS,
              backend="torch", verbose=FALSE)
  list(fit=fit, time=(proc.time()-t0)[["elapsed"]])
}

# Helper: time a Stan model
run_stan_direct <- function(stan_code, data_list) {
  tmp <- tempfile(fileext=".stan")
  writeLines(stan_code, tmp)
  t0 <- proc.time()
  mod <- cmdstan_model(tmp, quiet=TRUE)
  fit <- mod$sample(data=data_list, chains=CHAINS,
                    iter_sampling=N_SAMPLES, iter_warmup=WARMUP,
                    refresh=0, show_messages=FALSE)
  elapsed <- (proc.time()-t0)[["elapsed"]]
  draws <- as_draws_array(fit$draws())
  summ <- summarise_draws(draws)
  list(summ=summ, time=elapsed)
}

# Helper: time a greta model
run_greta_model <- function(setup_fn) {
  if (!has_greta) return(NULL)
  tryCatch({
    res <- setup_fn()
    t0 <- proc.time()
    draws <- greta::mcmc(res$model, n_samples=N_SAMPLES, warmup=WARMUP,
                         chains=CHAINS, verbose=FALSE)
    elapsed <- (proc.time()-t0)[["elapsed"]]
    draws_arr <- as_draws_array(draws)
    summ <- summarise_draws(draws_arr)
    list(summ=summ, time=elapsed)
  }, error = function(e) { cat("  greta error:", e$message, "\n"); NULL })
}

report <- function(name, torch_res, stan_res, greta_res=NULL) {
  cat(sprintf("\n--- %s ---\n", name))

  t_time <- torch_res$time
  s_time <- if (!is.null(stan_res)) stan_res$time else NA
  g_time <- if (!is.null(greta_res)) greta_res$time else NA

  t_summ <- torch_res$fit$summary
  s_summ <- if (!is.null(stan_res)) stan_res$summ else NULL

  cat(sprintf("  Torch:  %6.1fs", t_time))
  if (!is.null(t_summ)) cat(sprintf(" | %s", paste(sprintf("%s=%.3f", t_summ$variable[1:min(3,nrow(t_summ))], t_summ$mean[1:min(3,nrow(t_summ))]), collapse=", ")))
  cat("\n")

  cat(sprintf("  Stan:   %6.1fs", s_time))
  if (!is.null(s_summ)) {
    vars <- s_summ$variable[!grepl("lp__|__", s_summ$variable)]
    s_filt <- s_summ[s_summ$variable %in% vars[1:min(3,length(vars))], ]
    cat(sprintf(" | %s", paste(sprintf("%s=%.3f", s_filt$variable, s_filt$mean), collapse=", ")))
  }
  cat("\n")

  if (!is.na(g_time)) cat(sprintf("  greta:  %6.1fs\n", g_time))

  if (!is.na(s_time)) cat(sprintf("  Speedup (Stan/Torch): %.1fx\n", t_time/s_time))

  list(name=name, torch=t_time, stan=s_time, greta=g_time)
}

# =============================================================================
# M1: Intercept-only (m_1a)
# =============================================================================
set.seed(42)
y1 <- rnorm(500, 5, 2)

r1_t <- run_torch(function() {
  mu <- normal(0, 10); sigma <- half_cauchy(5)
  y <- as_data(y1); distribution(y) <- normal(mu, sigma)
  model(mu, sigma)
})

r1_s <- run_stan_direct("
data { int N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; }
model { mu ~ normal(0,10); sigma ~ cauchy(0,5); y ~ normal(mu, sigma); }
", list(N=500L, y=y1))

r1_g <- run_greta_model(function() {
  mu <- greta::normal(0,10); sigma <- greta::cauchy(0,5,truncation=c(0,Inf))
  y <- greta::as_data(y1); greta::distribution(y) <- greta::normal(mu, sigma)
  list(model=greta::model(mu, sigma))
})

results[[1]] <- report("M1: Intercept-only (n=500)", r1_t, r1_s, r1_g)

# =============================================================================
# M2: Fixed effects regression (m_1b)
# =============================================================================
set.seed(123)
n2 <- 1000; x2 <- rnorm(n2); y2 <- 2 + 3*x2 + rnorm(n2, 0, 1)

r2_t <- run_torch(function() {
  a <- normal(0,10); b <- normal(0,10); s <- half_cauchy(5)
  x <- as_data(x2); y <- as_data(y2)
  distribution(y) <- normal(a + b * x, s)
  model(a, b, s)
})

r2_s <- run_stan_direct("
data { int N; vector[N] x; vector[N] y; }
parameters { real a; real b; real<lower=0> s; }
model { a ~ normal(0,10); b ~ normal(0,10); s ~ cauchy(0,5);
  y ~ normal(a + b * x, s); }
", list(N=1000L, x=x2, y=y2))

results[[2]] <- report("M2: Linear regression (n=1000)", r2_t, r2_s)

# =============================================================================
# M3: Random intercept LMM (m_2a)
# =============================================================================
set.seed(42)
J3 <- 20; n3 <- 1000
gid3 <- sample(1:J3, n3, replace=TRUE)
tau3 <- 2; sig3 <- 1; mu3 <- 5
alpha3 <- rnorm(J3, mu3, tau3)
y3 <- rnorm(n3, alpha3[gid3], sig3)

r3_t <- run_torch(function() {
  mu <- normal(0,10); tau <- half_cauchy(5)
  z <- normal(0,1,dim=c(20L,1L)); sigma <- half_cauchy(5)
  y <- as_data(y3)
  fitted <- mu + tau * z
  distribution(y) <- normal(fitted[gid3], sigma)
  model(mu, tau, sigma, z)
})

r3_s <- run_stan_direct("
data { int N; int J; array[N] int gid; vector[N] y; }
parameters { real mu; real<lower=0> tau; real<lower=0> sigma; vector[J] z; }
transformed parameters { vector[J] alpha = mu + tau * z; }
model { mu ~ normal(0,10); tau ~ cauchy(0,5); sigma ~ cauchy(0,5);
  z ~ std_normal(); y ~ normal(alpha[gid], sigma); }
", list(N=n3, J=J3, gid=gid3, y=y3))

results[[3]] <- report("M3: Random intercept LMM (n=1000, J=20)", r3_t, r3_s)

# =============================================================================
# M4: Crossed random effects (m_2b)
# =============================================================================
set.seed(42)
J4a <- 30; J4b <- 10; n4 <- 2000
gid4a <- sample(1:J4a, n4, replace=TRUE)
gid4b <- sample(1:J4b, n4, replace=TRUE)
a4 <- rnorm(J4a, 0, 1.5); b4 <- rnorm(J4b, 0, 1)
y4 <- 5 + a4[gid4a] + b4[gid4b] + rnorm(n4, 0, 0.8)

r4_t <- run_torch(function() {
  mu <- normal(0,10)
  tau_a <- half_cauchy(3); tau_b <- half_cauchy(3)
  za <- normal(0,1,dim=c(30L,1L)); zb <- normal(0,1,dim=c(10L,1L))
  sigma <- half_cauchy(3)
  y <- as_data(y4)
  fitted <- mu + tau_a * za[gid4a] + tau_b * zb[gid4b]
  distribution(y) <- normal(fitted, sigma)
  model(mu, tau_a, tau_b, sigma, za, zb)
})

r4_s <- run_stan_direct("
data { int N; int Ja; int Jb; array[N] int ga; array[N] int gb; vector[N] y; }
parameters { real mu; real<lower=0> tau_a; real<lower=0> tau_b;
  real<lower=0> sigma; vector[Ja] za; vector[Jb] zb; }
transformed parameters { vector[Ja] a=tau_a*za; vector[Jb] b=tau_b*zb; }
model { mu~normal(0,10); tau_a~cauchy(0,3); tau_b~cauchy(0,3); sigma~cauchy(0,3);
  za~std_normal(); zb~std_normal();
  y ~ normal(mu + a[ga] + b[gb], sigma); }
", list(N=n4, Ja=J4a, Jb=J4b, ga=gid4a, gb=gid4b, y=y4))

results[[4]] <- report("M4: Crossed random effects (n=2000, J=30+10)", r4_t, r4_s)

# =============================================================================
# M5: Logistic GLMM (m_5 type)
# =============================================================================
set.seed(42)
J5 <- 15; n5 <- 1500
gid5 <- sample(1:J5, n5, replace=TRUE)
x5 <- rnorm(n5)
a5 <- rnorm(J5, 0, 1)
p5 <- plogis(0.5 + 0.8*x5 + a5[gid5])
y5 <- rbinom(n5, 1, p5)

r5_t <- run_torch(function() {
  alpha <- normal(0,5); beta <- normal(0,5)
  tau <- half_cauchy(2); z <- normal(0,1,dim=c(15L,1L))
  x <- as_data(x5); y <- as_data(y5)
  eta <- alpha + beta * x + tau * z[gid5]
  p <- logistic_link(eta)
  distribution(y) <- bernoulli(p)
  model(alpha, beta, tau, z)
})

r5_s <- run_stan_direct("
data { int N; int J; array[N] int gid; vector[N] x; array[N] int y; }
parameters { real alpha; real beta; real<lower=0> tau; vector[J] z; }
transformed parameters { vector[J] a = tau * z; }
model { alpha~normal(0,5); beta~normal(0,5); tau~cauchy(0,2); z~std_normal();
  for (i in 1:N) y[i] ~ bernoulli_logit(alpha + beta*x[i] + a[gid[i]]); }
", list(N=n5, J=J5, gid=gid5, x=x5, y=as.integer(y5)))

results[[5]] <- report("M5: Logistic GLMM (n=1500, J=15)", r5_t, r5_s)

# =============================================================================
# M6: Poisson GLMM (m_4 type)
# =============================================================================
set.seed(42)
J6 <- 20; n6 <- 1000
gid6 <- sample(1:J6, n6, replace=TRUE)
x6 <- rnorm(n6)
a6 <- rnorm(J6, 0, 0.5)
y6 <- rpois(n6, exp(1 + 0.5*x6 + a6[gid6]))

r6_t <- run_torch(function() {
  alpha <- normal(0,5); beta <- normal(0,5)
  tau <- half_cauchy(2); z <- normal(0,1,dim=c(20L,1L))
  x <- as_data(x6); y <- as_data(y6)
  rate <- exp(alpha + beta * x + tau * z[gid6])
  distribution(y) <- poisson_dist(rate)
  model(alpha, beta, tau, z)
})

r6_s <- run_stan_direct("
data { int N; int J; array[N] int gid; vector[N] x; array[N] int y; }
parameters { real alpha; real beta; real<lower=0> tau; vector[J] z; }
transformed parameters { vector[J] a = tau * z; }
model { alpha~normal(0,5); beta~normal(0,5); tau~cauchy(0,2); z~std_normal();
  for (i in 1:N) y[i] ~ poisson_log(alpha + beta*x[i] + a[gid[i]]); }
", list(N=n6, J=J6, gid=gid6, x=x6, y=as.integer(y6)))

results[[6]] <- report("M6: Poisson GLMM (n=1000, J=20)", r6_t, r6_s)

# =============================================================================
# M7: Random slopes (m_2d)
# =============================================================================
set.seed(42)
J7 <- 15; n7 <- 750
gid7 <- sample(1:J7, n7, replace=TRUE)
x7 <- rnorm(n7)
a7 <- rnorm(J7, 5, 2); b7 <- rnorm(J7, 1, 0.5)
y7 <- a7[gid7] + b7[gid7]*x7 + rnorm(n7, 0, 0.8)

r7_t <- run_torch(function() {
  mu_a <- normal(0,10); mu_b <- normal(0,5)
  tau_a <- half_cauchy(3); tau_b <- half_cauchy(2)
  za <- normal(0,1,dim=c(15L,1L)); zb <- normal(0,1,dim=c(15L,1L))
  sigma <- half_cauchy(3)
  x <- as_data(x7); y <- as_data(y7)
  fitted <- (mu_a + tau_a * za[gid7]) + (mu_b + tau_b * zb[gid7]) * x
  distribution(y) <- normal(fitted, sigma)
  model(mu_a, mu_b, tau_a, tau_b, sigma, za, zb)
})

r7_s <- run_stan_direct("
data { int N; int J; array[N] int gid; vector[N] x; vector[N] y; }
parameters { real mu_a; real mu_b; real<lower=0> tau_a; real<lower=0> tau_b;
  real<lower=0> sigma; vector[J] za; vector[J] zb; }
transformed parameters {
  vector[J] a = mu_a + tau_a*za; vector[J] b = mu_b + tau_b*zb; }
model { mu_a~normal(0,10); mu_b~normal(0,5);
  tau_a~cauchy(0,3); tau_b~cauchy(0,2); sigma~cauchy(0,3);
  za~std_normal(); zb~std_normal();
  y ~ normal(a[gid] + b[gid] .* x, sigma); }
", list(N=n7, J=J7, gid=gid7, x=x7, y=y7))

results[[7]] <- report("M7: Random slopes (n=750, J=15)", r7_t, r7_s)

# =============================================================================
# Summary
# =============================================================================
cat("\n\n========== SUMMARY ==========\n\n")
cat(sprintf("%-50s  %8s  %8s  %8s  %8s\n", "Model", "Torch(s)", "Stan(s)", "greta(s)", "Speedup"))
cat(paste(rep("=", 90), collapse=""), "\n")
for (r in results) {
  sp <- if (!is.na(r$stan)) sprintf("%.0fx", r$torch/r$stan) else "N/A"
  gt <- if (!is.na(r$greta)) sprintf("%.1f", r$greta) else "---"
  cat(sprintf("%-50s  %8.1f  %8.1f  %8s  %8s\n", r$name, r$torch,
              ifelse(is.na(r$stan), 0, r$stan), gt, sp))
}
cat("\nBenchmark complete.\n")
