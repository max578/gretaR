# Critical Review: gretaR Package
## Independent Assessment — 2026-04-11

---

## 1. Global Mutable State Architecture

**Issue:** The entire DAG (nodes, distributions, node counter) lives in a single global environment `.gretaR_env`. Every call to `as_data()`, `normal()`, `variable()`, or `distribution<-()` mutates this global state. Users must call `reset_gretaR_env()` between models or face node contamination.

**Consequences:**
- Two models defined in sequence without reset share nodes silently.
- `gretaR_glm()` calls `reset_gretaR_env()` internally, but DSL users must remember to call it.
- Thread-unsafe: parallel chain evaluation via `future` or `parallel` would corrupt the global DAG.
- `.gretaR_env$active_grad_fn` is set globally during sampling — a second concurrent sampler would overwrite it.

**Recommendation:** Encapsulate the DAG in the model object. Each call to `model()` should snapshot the current DAG and become self-contained. The global environment should be a staging area that gets frozen on compilation, not a permanent shared state. This is how Stan operates — the model is immutable after compilation.

**Justification:** greta has the same global-state design and suffers from the same problems. brms avoids it entirely by generating Stan code from formula objects with no mutable state. This is gretaR's most fundamental architectural weakness.

---

## 2. Silent NaN/Inf Accumulation in log_prob

**Issue:** The `log_prob()` function accumulates prior and likelihood terms without checking for NaN or Inf at any point. If any distribution's `log_prob` returns NaN (e.g., from `sd = 0`, `rate < 0`, or numerical overflow), the total is silently NaN. The sampler then operates on NaN gradients.

**Consequences:**
- `eval_grad()` replaces NaN gradients with 0 (line 133 of inference_utils.R), which is mathematically incorrect — it makes the sampler think the gradient is flat, not undefined.
- The HMC leapfrog continues with zero gradients, producing random walks instead of informed proposals.
- Divergence detection may or may not catch this, depending on the energy check.
- **Users get wrong posteriors with no warning.**

**Recommendation:** Add a `check_finite()` call after each `log_prob` evaluation in the prior and likelihood loops. If NaN or Inf is detected, throw an informative error identifying which distribution and which parameter value caused it. This is what Stan does — it rejects the proposal and reports the issue.

**Justification:** Stan's `normal_lpdf` errors explicitly when `sigma <= 0`. gretaR's Normal `log_prob` silently returns -Inf or NaN. This is the single most dangerous correctness issue.

---

## 3. No Distribution Parameter Validation

**Issue:** None of the 18 distributions validate their parameters at construction time or evaluation time. Invalid parameters (`sd = 0`, `rate = -1`, `alpha = 0` for Beta) silently produce NaN/Inf log-probabilities.

**Affected distributions:** All of them. Every distribution accepts any numeric value for every parameter without checking positivity, bounds, or finiteness.

**Recommendation:** Add `torch_clamp` guards in `log_prob` methods (already present for some, like `torch_clamp(p, min = 1e-7)` in Bernoulli), and add explicit R-level validation in constructors for fixed numeric parameters. For gretaR_array parameters (which are learned), the guards in `log_prob` are sufficient since the transform should keep values in the valid range.

**Justification:** torch silently propagates NaN through operations. Stan errors immediately. Users expect Bayesian software to either work correctly or error explicitly — never to produce wrong answers silently.

---

## 4. Stan Code Generator Correctness

**Issue:** The Stan code generator produces invalid Stan code in several cases:

1. **Element-wise ops on scalars:** `(%s .* %s)` is invalid Stan for scalar × scalar (should be `*`). The generator doesn't distinguish between scalar and vector operations.
2. **Matrix parameter bounds:** `matrix<lower=0>[d1, d2]` is invalid Stan syntax. Bounds apply only to scalar `real` parameters in Stan. For constrained matrices, you need `cholesky_factor_corr` or similar types.
3. **Integer data detection is heuristic:** The generator infers `array[N] int` by checking which data nodes are in discrete likelihoods, but this misses cases where the same data is used in both continuous and discrete contexts.
4. **Expression indexing:** `(mu + tau .* z)[data_8]` works in Stan for some cases but fails for others (e.g., when the expression result type doesn't support indexing).

**Recommendation:** The code generator needs a **type system**. Each DAG node should carry its Stan type (real, vector, matrix, array int) and the generator should emit type-correct expressions. This is a significant refactor but necessary for correctness.

**Short-term fix:** Add a `stan_type` field to `GretaRArray` (alongside `op_type`) and propagate it during DAG construction.

**Justification:** brms generates thousands of lines of valid Stan code by maintaining an internal type system throughout its code generation pipeline. gretaR's current approach of inferring types at generation time is fundamentally fragile.

---

## 5. Torch Backend Performance Architecture

**Issue:** The torch backend is 13–152× slower than Stan. The performance optimisations (compiled log_prob, autograd_grad, JIT tracing) closed the gap from 50–64× to 13–16× for small models, but the fundamental architecture limits further improvement:

1. **Per-gradient R↔C++ boundary crossing:** Each `eval_grad()` call creates a torch tensor from R numeric, evaluates the DAG through torch C++ calls, and extracts the gradient back to R. This round-trip is ~0.7ms per call.
2. **R-level leapfrog loop:** The leapfrog operates in pure R with numeric vectors, calling torch only for gradient evaluation. This is actually optimal for small models (R arithmetic is fast) but prevents GPU utilisation.
3. **Dynamic computation graph:** torch rebuilds the computation graph every forward pass. Stan's static graph (compiled C++) avoids this entirely.

**Recommendation:** Accept that the torch backend will never match Stan for standard models. Position it explicitly as the backend for:
- Custom distributions that Stan can't express
- GPU-accelerated large models (when they exist)
- Rapid prototyping with MAP/ADVI before committing to Stan
- Models requiring torch-specific operations (neural network likelihoods)

The Stan backend should be the default recommendation for production inference.

**Justification:** This is an architectural reality, not a bug. TensorFlow Probability (the backend greta uses) has the same issue. The interpreted-language-calling-into-C++ overhead cannot be eliminated without rewriting the core in C++.

---

## 6. NUTS Implementation Correctness

**Issue:** The NUTS implementation has several subtle correctness concerns:

1. **Divergence threshold is arbitrary:** `delta < -1000` is used without justification. Stan uses a configurable `max_treedepth` and energy error threshold.
2. **No energy fraction (E-BFMI) diagnostic:** Stan computes the Bayesian Fraction of Missing Information; gretaR doesn't.
3. **Mass matrix estimation uses sample variance:** The windowed warmup computes diagonal mass matrix from sample variance during Phase 2. Stan uses a more sophisticated windowed estimator with multiple adaptation windows.
4. **No dense mass matrix option:** Only diagonal mass matrix is supported. For models with strong parameter correlations, this limits sampling efficiency.
5. **Tree depth is not reported per-sample:** gretaR stores tree depth per iteration during sampling but doesn't include it in the gretaR_fit output for diagnostics.

**Recommendation:** These are not bugs but quality-of-implementation issues. For the torch backend, they limit sampling efficiency but don't produce wrong results (validated against Stan). For production use, recommend the Stan backend.

**Justification:** Stan's NUTS implementation has been refined over 12+ years. gretaR's is a correct first implementation suitable for validation and prototyping. The Stan backend provides production-grade NUTS.

---

## 7. Truncation Implementation

**Issue:** Truncation is implemented by tightening the parameter constraint and applying the corresponding transform. This is correct for HMC sampling (the sampler stays within bounds) but **does not adjust the normalising constant** in the log_prob.

**Consequence:** For HMC/NUTS, this is fine — the normalising constant `log(F(b) - F(a))` is a constant that doesn't affect gradients or the Metropolis acceptance ratio. However:
- **ADVI is affected:** The ELBO includes the entropy term, which depends on the normalising constant. Omitting it biases the ELBO.
- **Model comparison is affected:** The log marginal likelihood from Laplace approximation will be wrong by the missing constant.
- **Bayes factors are wrong** if computed from models with truncated vs untruncated priors.

**Recommendation:** Compute and include the normalising constant for distributions where the CDF is available in closed form (Normal, Exponential, Gamma, Beta via regularised incomplete beta function). For others, document that the normalising constant is omitted and that this affects VI and model comparison but not MCMC.

**Justification:** Stan includes the normalising constant for truncated distributions. greta omits it (same as gretaR). This is a known limitation of greta's approach. For MCMC-only use, it's harmless. For VI or model comparison, it's a bug.

---

## 8. Smooth Term Integration Limitations

**Issue:** The mgcv smooth integration uses `smooth2random()` to decompose penalised bases into fixed + random effects. This is the brms approach and is correct, but has limitations:

1. **No prediction machinery:** After fitting, there's no way to compute `f(x_new)` for new covariate values. This requires `mgcv::PredictMat()` and the back-transformation from the smooth2random parameterisation — neither is implemented.
2. **No EDF computation:** The effective degrees of freedom of each smooth cannot be computed from the posterior (it requires the penalty matrix ratio, not just the variance component).
3. **Tensor products can be large:** `te(x1, x2, k=c(20, 20))` produces 399 parameters. With non-centred parameterisation, that's 399 `z_raw` + variance components. HMC will struggle with this many parameters on the torch backend.
4. **Factor-by smooths untested:** `s(x, by = factor_var)` produces one smooth per factor level. The current implementation handles this via `smoothCon` returning a list, but it's not explicitly tested.

**Recommendation:** Add a `predict.gretaR_glm_fit()` method that uses `mgcv::PredictMat()` to construct the prediction matrix for smooth terms. Store the original smooth objects in the fit for this purpose.

---

## 9. Memory Management in Long Sampling Runs

**Issue:** During NUTS sampling, each gradient evaluation creates torch tensors that are not explicitly freed. R's garbage collector handles cleanup, but:
- Torch tensors are reference-counted in C++, not managed by R's GC.
- If the GC doesn't run frequently enough, torch memory accumulates.
- For models with many parameters or long chains, this can cause memory pressure.

**Specific locations:**
- `eval_grad()`: creates a new tensor every call (`torch_tensor(theta_vec, dtype = ...)`)
- `unconstrained_to_constrained()`: creates intermediate tensors for each variable
- `leapfrog_vec()`: creates tensors in `eval_grad` at every step

**Recommendation:** Pre-allocate a persistent tensor for `theta` and reuse it via `$copy_()` instead of creating new tensors. Profile memory usage on a 1000-iteration run to quantify the actual impact. The issue may be theoretical if R's GC is frequent enough.

---

## 10. Reproducibility

**Issue:** gretaR has no `seed` parameter in any inference function. Reproducibility depends on the user calling `set.seed()` before inference, which controls R's RNG but **not torch's RNG**.

**Consequences:**
- `torch_randn()` in ADVI uses torch's internal RNG, which is separate from R's `set.seed()`.
- `rnorm()` in HMC/NUTS uses R's RNG, which IS controlled by `set.seed()`.
- The combination means partially reproducible results: momentum samples are reproducible but VI samples are not.

**Recommendation:** Add a `seed` parameter to `mcmc()`, `variational()`, and `opt()`. Internally, call `set.seed(seed)` for R's RNG and `torch_manual_seed(seed)` for torch's RNG. Document this in the vignette.

**Justification:** Stan, brms, and NIMBLE all accept seed parameters. This is expected by users and reviewers.

---

## 11. API Completeness vs greta

**Issue:** gretaR aims for greta compatibility but is missing several greta features:

| greta feature | gretaR status |
|---|---|
| `truncation` | Implemented (this session) |
| `chol()`, `solve()`, `crossprod()` | Not exposed to users |
| `zeros()`, `ones()` | Not implemented |
| `greta_array` from existing tensor | Not exposed |
| `calculate()` for derived quantities | Not implemented |
| `joint()` distribution | Not implemented (mixture covers some cases) |
| Plotting of greta_arrays | Not implemented |
| `opt()` with multiple restarts | Single-start only |
| Prior/posterior simulation | `sample()` exists on distributions but not integrated |
| `mcmc()` progress bar | CLI messages only, no progress bar |

**Recommendation:** Prioritise `calculate()` (for posterior predictive checks), `chol()`/`solve()` (for multivariate models), and an MCMC progress bar (for user experience). These are the most-requested features in greta's issue tracker.

---

## 12. Test Coverage Gaps

**Issue:** 206 tests is respectable, but critical paths are undertested:

1. **No test for model with >10 parameters** via MCMC (only MAP/compilation tested for hierarchical).
2. **No test for Stan backend correctness** — the benchmark scripts test it but the testthat suite doesn't.
3. **No test for `compile_log_prob` producing identical results** to uncompiled `log_prob`.
4. **No test for mixed formula + Stan backend** pathway.
5. **No test for truncated distribution in a full MCMC run**.
6. **No negative tests** for invalid inputs (wrong dimension, non-numeric, etc.).
7. **No property-based tests** (e.g., gradient finite-difference checks).

**Recommendation:** Add a `test-correctness.R` file with:
- Gradient check: compare autograd gradient to finite-difference approximation for 5 models.
- Stan agreement: fit 3 models with both backends and check posterior mean agreement.
- Compiled log_prob check: verify `compile_log_prob(m)(theta) == log_prob(m, theta)` for 3 models.

---

## 13. CRAN Readiness

**Issue:** The package is close to CRAN-ready but has blockers:

1. **`cmdstanr` in Suggests with `Additional_repositories`:** CRAN policy allows this but it's unusual and may trigger manual review.
2. **`torch` dependency:** torch requires a post-install step (`install_torch()`). CRAN examples and vignettes must handle torch not being available.
3. **Vignette evaluation:** Currently `eval = FALSE` on most vignettes. CRAN prefers at least one evaluable vignette.
4. **Package size:** With `inst/validation/benchmark_report.pdf` (107KB), the installed package is larger than necessary.
5. **Namespace:** No `.onAttach` message telling users about `install_torch()`.

**Recommendation:** Add a `.onAttach` message, remove the PDF from inst/, ensure at least one vignette evaluates cleanly on CRAN (the getting-started vignette with `eval = torch::torch_is_installed()`), and consider making the Stan backend an extension package to avoid the cmdstanr dependency.

---

## 14. Formula Interface Robustness

**Issue:** The formula interface has several fragility points:

1. **lme4 regex parsing:** The regex `\\(([^()]+)\\|([^()]+)\\)` for random effects bars can match non-lme4 patterns (e.g., `abs(x|y)` in a math context).
2. **No formula validation:** If the user passes a formula referencing columns not in `data`, the error comes from `model.matrix()` deep in the call stack, not from gretaR.
3. **Factor handling:** If a factor has only 1 level, `model.matrix()` drops it silently. If a random-effect grouping variable has 1 level, the model has a degenerate variance component.
4. **Missing `na.action`:** The formula interface uses `na.fail` but doesn't tell the user which rows have NAs or in which columns.

**Recommendation:** Validate the formula and data before calling `model.matrix()`. Check that all referenced variables exist, that factors have >1 level, and that there are no NAs. Emit clear gretaR-specific error messages.

---

## 15. Documentation Accuracy

**Issue:** Some documentation is outdated after the rapid development:

1. `vi()` was renamed to `variational()` but some vignettes and examples may still reference `vi()`.
2. `binomial()` was renamed to `binomial_dist()` but the migration guide and technical documentation may reference the old name.
3. The `gretaR_draws` class still exists in `mcmc.R` but `mcmc()` now returns `gretaR_fit`. Some docs reference `gretaR_draws`.
4. The benchmark report references timing numbers from before performance optimisations.

**Recommendation:** Run a full-text search for `vi(`, `binomial(`, and `gretaR_draws` across all `.Rmd`, `.md`, and `.R` files. Update all references.

---

## Summary: Priority Ranking

| Priority | Issue | Impact | Effort |
|---|---|---|---|
| **P0** | Silent NaN/Inf in log_prob (#2) | Wrong posteriors | 1 day |
| **P0** | No parameter validation (#3) | Wrong posteriors | 1 day |
| **P0** | Stan code generator type bugs (#4) | Non-compilable code | 2-3 days |
| **P1** | Global mutable state (#1) | Model contamination | 1 week (refactor) |
| **P1** | Reproducibility / seed (#10) | Non-reproducible results | 2 hours |
| **P1** | Test coverage gaps (#12) | Undetected bugs | 2-3 days |
| **P1** | Documentation accuracy (#15) | User confusion | 1 day |
| **P2** | Truncation normalising constant (#7) | Wrong VI/model comparison | 1-2 days |
| **P2** | Smooth prediction (#8) | Missing feature | 1 week |
| **P2** | Memory management (#9) | Potential memory issues | Profile first |
| **P2** | API completeness vs greta (#11) | Migration friction | 1-2 weeks |
| **P2** | Formula robustness (#14) | Poor error messages | 2-3 days |
| **P3** | NUTS quality (#6) | Reduced efficiency | Ongoing |
| **P3** | CRAN readiness (#13) | Submission blocker | 2-3 days |
| **P3** | Torch performance (#5) | Architectural limit | Accept |
