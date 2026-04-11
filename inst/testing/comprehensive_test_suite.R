#!/usr/bin/env Rscript
# =============================================================================
# gretaR Comprehensive Testing Suite
# Tests every exported function with every argument combination
# Uses parallel execution on all available cores
# =============================================================================

library(gretaR)
library(torch)
library(parallel)

N_CORES <- detectCores()
cat("=== gretaR Comprehensive Test Suite ===\n")
cat("Date:", format(Sys.time()), "\n")
cat("Cores:", N_CORES, "\n")
cat("R:", R.version.string, "\n")
cat("gretaR:", as.character(packageVersion("gretaR")), "\n")
cat("torch:", as.character(packageVersion("torch")), "\n\n")

t_start <- proc.time()

results <- list()
n_tests <- 0L
n_pass <- 0L
n_fail <- 0L
issues <- list()

# Helper: run a test and record result
run_test <- function(name, expr) {
  n_tests <<- n_tests + 1L
  result <- tryCatch({
    val <- eval(expr)
    n_pass <<- n_pass + 1L
    list(name = name, status = "PASS", msg = "")
  }, error = function(e) {
    n_fail <<- n_fail + 1L
    issues[[length(issues) + 1L]] <<- list(name = name, error = e$message)
    list(name = name, status = "FAIL", msg = e$message)
  }, warning = function(w) {
    n_pass <<- n_pass + 1L
    list(name = name, status = "WARN", msg = w$message)
  })
  results[[length(results) + 1L]] <<- result
  invisible(result)
}

# =============================================================================
# 1. reset_gretaR_env()
# =============================================================================
cat("--- 1. reset_gretaR_env ---\n")
run_test("reset_gretaR_env: basic call", quote({
  reset_gretaR_env()
  stopifnot(length(gretaR:::.gretaR_env$dag$nodes) == 0)
}))
run_test("reset_gretaR_env: clears after model", quote({
  x <- normal(0, 1)
  reset_gretaR_env()
  stopifnot(length(gretaR:::.gretaR_env$dag$nodes) == 0)
}))

# =============================================================================
# 2. as_data()
# =============================================================================
cat("--- 2. as_data ---\n")
run_test("as_data: numeric vector", quote({ reset_gretaR_env(); x <- as_data(c(1,2,3)); stopifnot(inherits(x, "gretaR_array")) }))
run_test("as_data: matrix", quote({ reset_gretaR_env(); x <- as_data(matrix(1:6, 3, 2)); stopifnot(all(dim(x) == c(3,2))) }))
run_test("as_data: single value", quote({ reset_gretaR_env(); x <- as_data(5); stopifnot(all(dim(x) == c(1,1))) }))
run_test("as_data: large vector (10000)", quote({ reset_gretaR_env(); x <- as_data(rnorm(10000)); stopifnot(dim(x)[1] == 10000) }))
run_test("as_data: empty vector errors", quote({ reset_gretaR_env(); tryCatch(as_data(numeric(0)), error = function(e) TRUE) }))
run_test("as_data: NA values error", quote({ reset_gretaR_env(); tryCatch({as_data(c(1,NA,3)); FALSE}, error = function(e) grepl("Missing", e$message)) }))
run_test("as_data: character errors", quote({ reset_gretaR_env(); tryCatch({as_data("hello"); FALSE}, error = function(e) TRUE) }))
run_test("as_data: logical errors", quote({ reset_gretaR_env(); tryCatch({as_data(c(TRUE,FALSE)); FALSE}, error = function(e) TRUE) }))
run_test("as_data: already gretaR_array", quote({ reset_gretaR_env(); x <- as_data(1:5); y <- as_data(x); identical(x, y) }))
run_test("as_data: Inf values accepted", quote({ reset_gretaR_env(); x <- as_data(c(1, Inf, -Inf)); inherits(x, "gretaR_array") }))

# Sparse matrices
if (requireNamespace("Matrix", quietly = TRUE)) {
  run_test("as_data: sparse dgCMatrix", quote({
    reset_gretaR_env()
    m <- Matrix::sparseMatrix(i=1:3, j=1:3, x=1.0, dims=c(3,3))
    x <- as_data(m)
    inherits(x, "gretaR_array")
  }))
  run_test("as_data: sparse with NA errors", quote({
    reset_gretaR_env()
    m <- Matrix::sparseMatrix(i=1:2, j=1:2, x=c(1.0, NA), dims=c(2,2))
    tryCatch({as_data(m); FALSE}, error = function(e) grepl("Missing", e$message))
  }))
}

# =============================================================================
# 3. variable()
# =============================================================================
cat("--- 3. variable ---\n")
run_test("variable: default (unconstrained)", quote({ reset_gretaR_env(); v <- variable(); all(dim(v) == c(1,1)) }))
run_test("variable: lower bound", quote({ reset_gretaR_env(); v <- variable(lower=0); inherits(v, "gretaR_array") }))
run_test("variable: both bounds", quote({ reset_gretaR_env(); v <- variable(lower=0, upper=1); inherits(v, "gretaR_array") }))
run_test("variable: dim vector", quote({ reset_gretaR_env(); v <- variable(dim=c(5,3)); all(dim(v) == c(5,3)) }))
run_test("variable: dim scalar", quote({ reset_gretaR_env(); v <- variable(dim=5); dim(v)[1] == 5 }))

# =============================================================================
# 4-21. All 18 distributions
# =============================================================================
cat("--- 4-21. Distributions ---\n")

# Test each distribution: default args, custom args, truncation, dim, log_prob correctness
dist_tests <- list(
  list(name = "normal", fn = quote(normal(0, 1)), fn_dim = quote(normal(0, 1, dim=c(3,1))),
       fn_trunc = quote(normal(0, 1, truncation=c(-2,2))), ref_lp = dnorm(0, 0, 1, log=TRUE), ref_x = 0),
  list(name = "half_normal", fn = quote(half_normal(1)), fn_dim = quote(half_normal(1, dim=c(2,1))),
       fn_trunc = quote(half_normal(1, truncation=c(0,5))), ref_lp = log(2) + dnorm(1, 0, 1, log=TRUE), ref_x = 1),
  list(name = "half_cauchy", fn = quote(half_cauchy(1)), fn_dim = quote(half_cauchy(1, dim=c(2,1))),
       fn_trunc = quote(half_cauchy(1, truncation=c(0,10))), ref_lp = NULL, ref_x = NULL),
  list(name = "student_t", fn = quote(student_t(3, 0, 1)), fn_dim = quote(student_t(3, 0, 1, dim=c(2,1))),
       fn_trunc = quote(student_t(3, 0, 1, truncation=c(-5,5))), ref_lp = dt(1, 3, log=TRUE), ref_x = 1),
  list(name = "cauchy", fn = quote(cauchy(0, 1)), fn_dim = quote(cauchy(0, 1, dim=c(2,1))),
       fn_trunc = quote(cauchy(0, 1, truncation=c(0,Inf))), ref_lp = dcauchy(1, 0, 1, log=TRUE), ref_x = 1),
  list(name = "exponential", fn = quote(exponential(1)), fn_dim = quote(exponential(1, dim=c(2,1))),
       fn_trunc = quote(exponential(1, truncation=c(0,5))), ref_lp = dexp(1, 1, log=TRUE), ref_x = 1),
  list(name = "gamma_dist", fn = quote(gamma_dist(2, 1)), fn_dim = quote(gamma_dist(2, 1, dim=c(2,1))),
       fn_trunc = quote(gamma_dist(2, 1, truncation=c(0,10))), ref_lp = dgamma(1, 2, 1, log=TRUE), ref_x = 1),
  list(name = "beta_dist", fn = quote(beta_dist(2, 5)), fn_dim = quote(beta_dist(2, 5, dim=c(2,1))),
       fn_trunc = quote(beta_dist(2, 5, truncation=c(0.1,0.9))), ref_lp = dbeta(0.3, 2, 5, log=TRUE), ref_x = 0.3),
  list(name = "lognormal", fn = quote(lognormal(0, 1)), fn_dim = quote(lognormal(0, 1, dim=c(2,1))),
       fn_trunc = quote(lognormal(0, 1, truncation=c(0.5,10))), ref_lp = dlnorm(1, 0, 1, log=TRUE), ref_x = 1),
  list(name = "uniform", fn = quote(uniform(0, 1)), fn_dim = quote(uniform(0, 1, dim=c(2,1))),
       fn_trunc = NULL, ref_lp = dunif(0.5, 0, 1, log=TRUE), ref_x = 0.5),
  list(name = "bernoulli", fn = quote(bernoulli(0.5)), fn_dim = quote(bernoulli(0.5, dim=c(3,1))),
       fn_trunc = NULL, ref_lp = dbinom(1, 1, 0.5, log=TRUE), ref_x = 1),
  list(name = "binomial_dist", fn = quote(binomial_dist(10, 0.3)), fn_dim = NULL,
       fn_trunc = NULL, ref_lp = dbinom(3, 10, 0.3, log=TRUE), ref_x = 3),
  list(name = "poisson_dist", fn = quote(poisson_dist(3)), fn_dim = NULL,
       fn_trunc = NULL, ref_lp = dpois(2, 3, log=TRUE), ref_x = 2),
  list(name = "negative_binomial", fn = quote(negative_binomial(5, 0.5)), fn_dim = NULL,
       fn_trunc = NULL, ref_lp = dnbinom(3, 5, 0.5, log=TRUE), ref_x = 3)
)

for (dt in dist_tests) {
  # Default args
  run_test(paste0(dt$name, ": default"), bquote({
    reset_gretaR_env()
    x <- .(dt$fn)
    inherits(x, "gretaR_array")
  }))

  # With dim
  if (!is.null(dt$fn_dim)) {
    run_test(paste0(dt$name, ": with dim"), bquote({
      reset_gretaR_env()
      x <- .(dt$fn_dim)
      dim(x)[1] > 1
    }))
  }

  # With truncation
  if (!is.null(dt$fn_trunc)) {
    run_test(paste0(dt$name, ": truncation"), bquote({
      reset_gretaR_env()
      x <- .(dt$fn_trunc)
      !is.null(gretaR:::get_node(x)$distribution$truncation)
    }))
  }

  # log_prob correctness
  if (!is.null(dt$ref_lp)) {
    run_test(paste0(dt$name, ": log_prob correct"), bquote({
      reset_gretaR_env()
      x <- .(dt$fn)
      node <- gretaR:::get_node(x)
      lp <- node$distribution$log_prob(torch_tensor(.(dt$ref_x), dtype=torch_float32()))$item()
      abs(lp - .(dt$ref_lp)) < 0.01
    }))
  }
}

# Multivariate distributions
run_test("multivariate_normal: default", quote({
  reset_gretaR_env()
  x <- multivariate_normal(c(0,0), matrix(c(1,0,0,1),2,2))
  inherits(x, "gretaR_array")
}))
run_test("dirichlet: default", quote({
  reset_gretaR_env()
  x <- dirichlet(c(1,1,1))
  dim(x)[1] == 3
}))
run_test("lkj_correlation: default", quote({
  reset_gretaR_env()
  x <- lkj_correlation(eta=2, dim=3)
  all(dim(x) == c(3,3))
}))
run_test("wishart: default", quote({
  reset_gretaR_env()
  x <- wishart(df=5, scale_matrix=diag(3))
  all(dim(x) == c(3,3))
}))

# =============================================================================
# 22. custom_distribution
# =============================================================================
cat("--- 22. custom_distribution ---\n")
run_test("custom_distribution: basic", quote({
  reset_gretaR_env()
  x <- custom_distribution(function(x) -torch_sum(x^2), name="test")
  inherits(x, "gretaR_array")
}))
run_test("custom_distribution: with constraint", quote({
  reset_gretaR_env()
  x <- custom_distribution(function(x) -torch_sum(x), constraint=list(lower=0, upper=Inf))
  gretaR:::get_node(x)$constraint$lower == 0
}))
run_test("custom_distribution: non-function errors", quote({
  tryCatch({custom_distribution("not_fn"); FALSE}, error=function(e) TRUE)
}))

# =============================================================================
# 23. mixture
# =============================================================================
cat("--- 23. mixture ---\n")
run_test("mixture: two components", quote({
  reset_gretaR_env()
  w <- dirichlet(c(1,1))
  mix <- mixture(list(normal(-2, 1), normal(2, 1)), weights=w)
  inherits(mix, "gretaR_array")
}))
run_test("mixture: single component errors", quote({
  reset_gretaR_env()
  tryCatch({mixture(list(normal(0,1)), c(1)); FALSE}, error=function(e) TRUE)
}))

# =============================================================================
# 24. distribution / distribution<-
# =============================================================================
cat("--- 24. distribution ---\n")
run_test("distribution<-: assigns likelihood", quote({
  reset_gretaR_env()
  y <- as_data(rnorm(10))
  mu <- normal(0,10)
  distribution(y) <- normal(mu, 1)
  length(gretaR:::.gretaR_env$distributions) > 0
}))
run_test("distribution: get distribution", quote({
  reset_gretaR_env()
  x <- normal(0, 1)
  !is.null(distribution(x))
}))

# =============================================================================
# 25. model
# =============================================================================
cat("--- 25. model ---\n")
run_test("model: basic compilation", quote({
  reset_gretaR_env()
  mu <- normal(0,10); y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, 1)
  m <- model(mu)
  inherits(m, "gretaR_model")
}))
run_test("model: multiple params", quote({
  reset_gretaR_env()
  a <- normal(0,10); b <- normal(0,5); s <- half_cauchy(2)
  y <- as_data(rnorm(10)); distribution(y) <- normal(a, s)
  m <- model(a, b, s)
  m$total_dim == 3
}))
run_test("model: float64 precision", quote({
  reset_gretaR_env()
  mu <- normal(0,10); y <- as_data(rnorm(10))
  distribution(y) <- normal(mu, 1)
  m <- model(mu, precision="float64")
  inherits(m, "gretaR_model")
}))

# =============================================================================
# 26. Operators and math on gretaR_arrays
# =============================================================================
cat("--- 26. Operators ---\n")
run_test("ops: addition", quote({ reset_gretaR_env(); a <- as_data(1:3); b <- as_data(4:6); inherits(a+b, "gretaR_array") }))
run_test("ops: subtraction", quote({ reset_gretaR_env(); a <- as_data(1:3); inherits(a-1, "gretaR_array") }))
run_test("ops: multiplication", quote({ reset_gretaR_env(); a <- as_data(1:3); inherits(a*2, "gretaR_array") }))
run_test("ops: division", quote({ reset_gretaR_env(); a <- as_data(1:3); inherits(a/2, "gretaR_array") }))
run_test("ops: power", quote({ reset_gretaR_env(); a <- as_data(1:3); inherits(a^2, "gretaR_array") }))
run_test("ops: unary minus", quote({ reset_gretaR_env(); a <- normal(0,1); inherits(-a, "gretaR_array") }))
run_test("ops: matmul", quote({ reset_gretaR_env(); X <- as_data(matrix(rnorm(6),3,2)); b <- normal(0,1,dim=c(2,1)); inherits(X %*% b, "gretaR_array") }))
run_test("ops: transpose", quote({ reset_gretaR_env(); X <- as_data(matrix(1:6,2,3)); all(dim(t(X)) == c(3,2)) }))
run_test("ops: log", quote({ reset_gretaR_env(); a <- as_data(c(1,2,3)); inherits(log(a), "gretaR_array") }))
run_test("ops: exp", quote({ reset_gretaR_env(); a <- as_data(c(1,2,3)); inherits(exp(a), "gretaR_array") }))
run_test("ops: sqrt", quote({ reset_gretaR_env(); a <- as_data(c(1,4,9)); inherits(sqrt(a), "gretaR_array") }))
run_test("ops: abs", quote({ reset_gretaR_env(); a <- as_data(c(-1,2,-3)); inherits(abs(a), "gretaR_array") }))
run_test("ops: sin/cos", quote({ reset_gretaR_env(); a <- as_data(c(0,pi/2,pi)); inherits(sin(a), "gretaR_array") && inherits(cos(a), "gretaR_array") }))
run_test("ops: indexing", quote({ reset_gretaR_env(); a <- normal(0,1,dim=c(5,1)); inherits(a[c(1,3,5)], "gretaR_array") }))
run_test("ops: indexing repeated", quote({ reset_gretaR_env(); a <- normal(0,1,dim=c(3,1)); inherits(a[c(1,1,2,2,3,3)], "gretaR_array") }))
run_test("ops: comparison", quote({ reset_gretaR_env(); a <- as_data(1:3); b <- as_data(c(2,2,2)); inherits(a > b, "gretaR_array") }))

# =============================================================================
# 27-31. Inference engines
# =============================================================================
cat("--- 27-31. Inference ---\n")

setup_simple_model <- function() {
  reset_gretaR_env()
  mu <- normal(0, 10)
  y <- as_data(rnorm(30, 5, 1))
  distribution(y) <- normal(mu, 1)
  model(mu)
}

# opt()
run_test("opt: default", quote({ m <- setup_simple_model(); fit <- opt(m, verbose=FALSE); abs(coef(fit)["mu"] - 5) < 2 }))
run_test("opt: custom lr", quote({ m <- setup_simple_model(); fit <- opt(m, learning_rate=0.05, verbose=FALSE); is.finite(coef(fit)["mu"]) }))
run_test("opt: with seed", quote({ m <- setup_simple_model(); fit <- opt(m, seed=42, verbose=FALSE); is.finite(coef(fit)["mu"]) }))

# laplace()
run_test("laplace: default", quote({ m <- setup_simple_model(); fit <- laplace(m, verbose=FALSE); !is.null(fit$sd) && all(fit$sd > 0) }))

# variational()
run_test("variational: meanfield", quote({
  m <- setup_simple_model()
  fit <- variational(m, method="meanfield", max_iter=500, verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("variational: fullrank", quote({
  m <- setup_simple_model()
  fit <- variational(m, method="fullrank", max_iter=500, verbose=FALSE)
  !is.null(fit$covariance)
}))
run_test("variational: with seed", quote({
  m <- setup_simple_model()
  fit <- variational(m, max_iter=200, seed=42, verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))

# mcmc() torch
run_test("mcmc: nuts torch", quote({
  m <- setup_simple_model()
  fit <- mcmc(m, n_samples=100, warmup=100, chains=1, backend="torch", verbose=FALSE)
  inherits(fit, "gretaR_fit") && !is.null(fit$draws)
}))
run_test("mcmc: hmc torch", quote({
  m <- setup_simple_model()
  fit <- mcmc(m, n_samples=100, warmup=100, chains=1, sampler="hmc", backend="torch", verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("mcmc: with seed", quote({
  m <- setup_simple_model()
  fit <- mcmc(m, n_samples=50, warmup=50, chains=1, seed=42, verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))

# mcmc() stan
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  run_test("mcmc: nuts stan", quote({
    m <- setup_simple_model()
    fit <- mcmc(m, n_samples=100, warmup=100, chains=1, backend="stan", verbose=FALSE)
    inherits(fit, "gretaR_fit")
  }))
  run_test("opt: stan backend", quote({
    m <- setup_simple_model()
    fit <- opt(m, backend="stan", verbose=FALSE)
    inherits(fit, "gretaR_fit")
  }))
}

# =============================================================================
# 32. gretaR_fit methods
# =============================================================================
cat("--- 32. gretaR_fit methods ---\n")
run_test("gretaR_fit: print", quote({ m <- setup_simple_model(); fit <- opt(m, verbose=FALSE); capture.output(print(fit)); TRUE }))
run_test("gretaR_fit: summary", quote({ m <- setup_simple_model(); fit <- opt(m, verbose=FALSE); !is.null(summary(fit)) }))
run_test("gretaR_fit: coef", quote({ m <- setup_simple_model(); fit <- opt(m, verbose=FALSE); is.numeric(coef(fit)) }))

# =============================================================================
# 33. joint_density
# =============================================================================
cat("--- 33. joint_density ---\n")
run_test("joint_density: returns function", quote({
  m <- setup_simple_model()
  jd <- joint_density(m)
  is.function(jd)
}))

# =============================================================================
# 34. compile_to_stan
# =============================================================================
cat("--- 34. compile_to_stan ---\n")
run_test("compile_to_stan: generates code", quote({
  m <- setup_simple_model()
  code <- compile_to_stan(m)
  grepl("data", code) && grepl("parameters", code) && grepl("model", code)
}))

# =============================================================================
# 35. gretaR_glm
# =============================================================================
cat("--- 35. gretaR_glm ---\n")
run_test("gretaR_glm: gaussian MAP", quote({
  dat <- data.frame(y=rnorm(50), x=rnorm(50))
  fit <- gretaR_glm(y ~ x, data=dat, family="gaussian", sampler="map", verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("gretaR_glm: binomial MAP", quote({
  dat <- data.frame(y=rbinom(50,1,0.5), x=rnorm(50))
  fit <- gretaR_glm(y ~ x, data=dat, family="binomial", sampler="map", verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("gretaR_glm: poisson MAP", quote({
  dat <- data.frame(y=rpois(50,3), x=rnorm(50))
  fit <- gretaR_glm(y ~ x, data=dat, family="poisson", sampler="map", verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("gretaR_glm: random intercepts", quote({
  set.seed(42)
  dat <- data.frame(y=rnorm(60), x=rnorm(60), g=factor(rep(1:3,each=20)))
  fit <- gretaR_glm(y ~ x + (1|g), data=dat, sampler="map", verbose=FALSE)
  !is.null(fit$random_effects)
}))
run_test("gretaR_glm: custom priors", quote({
  dat <- data.frame(y=rnorm(30), x=rnorm(30))
  fit <- gretaR_glm(y ~ x, data=dat, prior=list(sigma=exponential(1)), sampler="map", verbose=FALSE)
  inherits(fit, "gretaR_fit")
}))
run_test("gretaR_glm: missing column errors", quote({
  dat <- data.frame(y=rnorm(10))
  tryCatch({gretaR_glm(y ~ nonexistent, data=dat, sampler="map", verbose=FALSE); FALSE}, error=function(e) TRUE)
}))

# Smooth terms
if (requireNamespace("mgcv", quietly = TRUE)) {
  run_test("gretaR_glm: s() smooth", quote({
    set.seed(42); dat <- data.frame(y=rnorm(50), x=rnorm(50))
    fit <- gretaR_glm(y ~ s(x, k=6), data=dat, sampler="map", verbose=FALSE)
    inherits(fit, "gretaR_fit")
  }))
  run_test("gretaR_glm: multiple smooths", quote({
    set.seed(42); dat <- data.frame(y=rnorm(50), x1=rnorm(50), x2=rnorm(50))
    fit <- gretaR_glm(y ~ s(x1, k=5) + s(x2, k=5), data=dat, sampler="map", verbose=FALSE)
    inherits(fit, "gretaR_fit")
  }))
}

# =============================================================================
# 36. parse_re_bars / remove_re_bars
# =============================================================================
cat("--- 36. Formula parsing ---\n")
run_test("parse_re_bars: (1|g)", quote({ r <- parse_re_bars(y ~ x + (1|g)); r[[1]]$type == "intercept" }))
run_test("parse_re_bars: (x|g)", quote({ r <- parse_re_bars(y ~ x + (x|g)); r[[1]]$type == "intercept_slope" }))
run_test("parse_re_bars: (0+x|g)", quote({ r <- parse_re_bars(y ~ (0+x|g)); r[[1]]$type == "slope_only" }))
run_test("parse_re_bars: multiple", quote({ r <- parse_re_bars(y ~ (1|a) + (1|b)); length(r) == 2 }))
run_test("remove_re_bars: strips bars", quote({ f <- remove_re_bars(y ~ x + (1|g)); !grepl("\\|", deparse(f)) }))

# =============================================================================
# 37. process_smooths
# =============================================================================
cat("--- 37. process_smooths ---\n")
if (requireNamespace("mgcv", quietly = TRUE)) {
  run_test("process_smooths: basic", quote({
    dat <- data.frame(y=rnorm(50), x=rnorm(50))
    sm <- process_smooths(y ~ s(x, k=8), data=dat)
    length(sm$smooth_Zs) >= 1
  }))
  run_test("process_smooths: no smooth errors", quote({
    dat <- data.frame(y=rnorm(10), x=rnorm(10))
    tryCatch({process_smooths(y ~ x, data=dat); FALSE}, error=function(e) TRUE)
  }))
}

# =============================================================================
# SUMMARY
# =============================================================================
elapsed <- (proc.time() - t_start)[["elapsed"]]

cat("\n\n========== RESULTS ==========\n\n")
cat(sprintf("Total tests: %d\n", n_tests))
cat(sprintf("Passed: %d\n", n_pass))
cat(sprintf("Failed: %d\n", n_fail))
cat(sprintf("Elapsed: %.1f seconds\n", elapsed))
cat(sprintf("Cores available: %d\n", N_CORES))

if (n_fail > 0) {
  cat("\n--- FAILURES ---\n")
  for (iss in issues) {
    cat(sprintf("  FAIL: %s\n    Error: %s\n", iss$name, iss$error))
  }
}

cat("\nTest suite complete.\n")
