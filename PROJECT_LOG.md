# PROJECT_LOG — gretaR

## Compressed Project Context

**Objective:** Build gretaR — a torch-based Bayesian probabilistic programming package for R. Fills the gap left by greta's decline (maintenance-only, TF/Python fragility). Provides R-native DSL, compiled to torch tensors, with HMC/NUTS inference and zero Python dependency.

**Approach:** Ground-up R package using R6 classes for DAG construction, torch for autograd/tensors, posterior package for output. Phase 1 targets: 12 P0 distributions, gretaR_array DSL with operator overloading, model compilation to log-joint density, HMC + NUTS samplers, posterior::draws_array output.

**Key assumptions:** R torch autograd is stable and sufficient. Distribution log_prob can be implemented directly as torch ops. Single torch backend (no TF fallback). R6 first, S7 migration later.

**Status:** Phase 2b complete. 15 distributions, hierarchical models, sparse matrices, MAP, Laplace, ADVI (mean-field + full-rank), formula interface (gretaR_glm), 3 vignettes, 129 passing tests. GitHub: max578/gretaR with CI. R CMD check: 0 errors, 0 warnings.

**Open issues:** (1) Validate hierarchical model MCMC recovery (in progress). (2) Profile NUTS on >10 param models. (3) Test on Windows/Linux CI. (4) Benchmarking vs Stan. (5) Phase 3: advanced formula, GP/ODE extensions, JOSS paper.

---

## Log Entries

### 2026-04-10 — v0.1.0-dev: Initial Implementation

**Summary:** Created complete package skeleton and implemented all Phase 1 core components.

**Files created:**
- `DESCRIPTION`, `NAMESPACE`, `LICENSE`, `.Rbuildignore`, `NEWS.md`
- `R/gretaR-package.R` — package docs, global state environment
- `R/zzz.R` — .onLoad, node registration, ID generation
- `R/transforms.R` — 6 bijectors (Identity, Log, Logit, ScaledLogit, Softplus, LowerBound)
- `R/distributions.R` — 12 P0 distributions (Normal, HalfNormal, HalfCauchy, StudentT, Uniform, Bernoulli, Binomial, Poisson, Gamma, Beta, Exponential, MultivariateNormal)
- `R/array.R` — gretaR_array R6+S3 class, operator overloading (Ops, Math), as_data(), variable(), distribution<-()
- `R/model.R` — model() compilation, log_prob(), grad_log_prob(), DAG traversal
- `R/inference_hmc.R` — static HMC with leapfrog, dual averaging, diagonal mass matrix
- `R/inference_nuts.R` — NUTS with iterative doubling, multinomial trajectory
- `R/mcmc.R` — mcmc() user function, posterior::draws_array output, S3 methods
- `tests/testthat/` — 5 test files covering transforms, distributions, arrays, model, integration
- `vignettes/getting-started.Rmd` — tutorial vignette

**Key decisions:**
- Single torch backend (no TF abstraction) — avoids premature generalisation
- R6 + S3 hybrid: R6 for mutable DAG nodes, S3 for operator dispatch
- Own distribution implementations (torch ops for log_prob) rather than wrapping sparse R torch distributions
- Global environment (.gretaR_env) for DAG state — simple, follows greta's pattern
- posterior::draws_array for output — full ecosystem interop

**Next steps:**
1. Generate roxygen2 man pages for all exported functions
2. Profile NUTS performance on larger models (hierarchical, >10 params)
3. Verify gradient correctness with finite-difference checks
4. Test on Windows/Linux CI
5. Phase 2: P1 distributions (Dirichlet, NegBin, Wishart, LKJ)

### 2026-04-10 — v0.1.0-dev: Samplers Fixed and Validated

**Summary:** Fixed critical bugs in HMC/NUTS samplers. Package now produces correct posterior samples.

**Key bugs fixed:**
1. `resolve_param()` didn't handle gretaR_array S3 wrapper → segfault. Fixed to call `get_node(x)$compute()`.
2. Likelihood distribution template nodes were being included as free variables. Fixed by excluding nodes registered in `.gretaR_env$distributions`.
3. `mcmc()` was overriding the auto-tuned step size with a default of 0.1. Removed the default.
4. Warmup mass matrix adaptation invalidated the adapted step size. Implemented 3-phase windowed warmup (step-size → mass matrix → re-adapt step-size).
5. `find_initial_values()` was random — replaced with Adam gradient ascent toward MAP.
6. HalfNormal log_prob constant had wrong sign (0.2258 should be -0.2258).
7. torch `with_no_grad` scoping caused leapfrog updates to not propagate. Rewrote leapfrog to use numeric vectors.

**Validation results:**
- Normal model (known variance): mu=4.98 (true=4.96), R-hat=1.02, ESS=430
- Normal model (unknown variance): mu=3.05 (true=3), sigma=1.58 (true=1.5), R-hat≈1.0
- Linear regression: alpha=1.95 (true=2), beta=2.97 (true=3), sigma=0.49 (true=0.5), ESS>386
