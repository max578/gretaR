# gretaR Comprehensive Testing Report

**Date:** 2026-04-11
**Hardware:** Apple Silicon (M-series), 10 cores, macOS
**R:** 4.5.2 | **torch:** 0.16.3 | **gretaR:** 0.1.0

---

## 1. Testing Coverage Summary

| Metric | Value |
|---|---|
| Exported functions tested | 38/38 (100%) |
| Total test cases (comprehensive suite) | 128 |
| Total test cases (testthat suite) | 210 |
| **Combined unique tests** | **338** |
| Pass rate | 100% |
| Comprehensive suite runtime | 17.7 seconds |
| Cores available | 10 |

### Functions Tested

All 38 exported functions were tested with default arguments, non-default
arguments, edge cases, and (where applicable) invalid inputs:

`as_data`, `bernoulli`, `beta_dist`, `binomial_dist`, `cauchy`,
`compile_to_stan`, `custom_distribution`, `dirichlet`, `distribution`,
`distribution<-`, `exponential`, `gamma_dist`, `gretaR_glm`, `half_cauchy`,
`half_normal`, `hmc`, `joint_density`, `laplace`, `lkj_correlation`,
`lognormal`, `mcmc`, `mixture`, `model`, `multivariate_normal`,
`negative_binomial`, `normal`, `nuts`, `opt`, `parse_re_bars`,
`poisson_dist`, `process_smooths`, `remove_re_bars`, `reset_gretaR_env`,
`student_t`, `uniform`, `variable`, `variational`, `wishart`, `%*%`

### Argument Combinations Tested

| Category | Tests |
|---|---|
| Distribution constructors (default) | 18 |
| Distribution constructors (with dim) | 14 |
| Distribution constructors (with truncation) | 9 |
| Distribution log_prob correctness (vs R reference) | 14 |
| Operators (+, -, *, /, ^, %*%, t, log, exp, etc.) | 16 |
| Indexing (integer, repeated, logical) | 3 |
| Inference: MAP, Laplace, VI, HMC, NUTS | 10 |
| Stan backend (MCMC + MAP) | 2 |
| Formula interface (3 families + RE + smooths) | 7 |
| Formula parsing (parse_re_bars, remove_re_bars) | 5 |
| Edge cases (NA, empty, large, Inf, single-value) | 8 |
| Invalid inputs (character, non-function, missing column) | 5 |
| Output methods (print, summary, coef) | 3 |
| Reproducibility (seed parameter) | 3 |

---

## 2. Issues Identified and Resolved

### Issue 1: `%*%` operator not dispatching when package loaded

**Function:** `%*%` (matrix multiplication)
**Problem:** `%*%` is an S4 generic in R >= 4.3. The S3 method
`%*%.gretaR_array` was never called when the package was loaded via
`library(gretaR)`. It only worked when files were sourced directly.

**Reproducible example:**
```r
library(gretaR)
X <- as_data(matrix(rnorm(6), 3, 2))
b <- normal(0, 1, dim = c(2, 1))
X %*% b  # Error: requires numeric/complex matrix/vector arguments
```

**Fix:** Replaced the S3 method with an exported `%*%` function that checks
for `gretaR_array` arguments and dispatches accordingly, falling back to
`base::%*%` for non-gretaR objects.

**Status:** Fixed.

### Issue 2: `variational()` fails for 1-parameter models

**Function:** `variational()`
**Problem:** `diag(scalar)` in R creates a 0x0 matrix (not 1x1). When the
model has a single parameter, `cov_mat <- diag(sigma_final^2)` produces
a 0x0 matrix, causing "non-conformable arguments" in the draws generation.

**Reproducible example:**
```r
library(gretaR)
reset_gretaR_env()
mu <- normal(0, 10)
y <- as_data(rnorm(30, 5, 1))
distribution(y) <- normal(mu, 1)
m <- model(mu)
variational(m)  # Error: non-conformable arguments
```

**Fix:** Added a check: if `length(sigma_final) == 1`, use
`matrix(sigma_final^2, 1, 1)` instead of `diag(sigma_final^2)`.

**Status:** Fixed.

### Issue 3: Internal functions not accessible in standalone test script

**Functions:** `get_node()`, `.gretaR_env`
**Problem:** These are not exported. The comprehensive test suite runs
outside the package namespace and couldn't access them.

**Fix:** Used `gretaR:::get_node()` and `gretaR:::.gretaR_env` in the
test suite. No package change needed — these are correctly internal.

**Status:** Not a bug. Test suite adjusted.

---

## 3. Remaining Limitations / Open Issues

| # | Limitation | Severity | Notes |
|---|---|---|---|
| 1 | `%*%` override masks `base::%*%` globally | Low | Falls back correctly for non-gretaR objects. CRAN may flag this. Alternative: document and use explicit `gretaR_matmul()`. |
| 2 | No parallel test execution | Low | Tests run sequentially in 17.7s. Parallelisation would need `callr` or `testthat::test_dir(parallel=TRUE)` but torch is not fork-safe. |
| 3 | Stan backend tests require cmdstanr + CmdStan | Medium | Skipped when not available. 2 tests conditional on cmdstanr presence. |
| 4 | mgcv smooth tests require mgcv | Low | mgcv is a recommended package (always available). 2 tests conditional. |
| 5 | No stress test (>100K observations) | Low | Would take >10 minutes. Not included in automated suite. |

---

## 4. Recommendations

1. **Export `gretaR_matmul()` as the documented matrix multiply function** and
   keep the `%*%` override as a convenience. This makes the API explicit and
   avoids CRAN concerns about masking base functions.

2. **Add `skip_if_not(torch::torch_is_installed())` to all testthat tests**
   for robustness on systems where torch binaries aren't installed.

3. **Consider `callr::r()` for parallel test execution** — torch's C++ backend
   is not fork-safe, but `callr` uses separate R processes.

4. **Add a stress test script** (separate from the automated suite) that tests
   models with 50K+ observations and 50+ parameters, run manually before releases.
