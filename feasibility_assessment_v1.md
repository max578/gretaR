# Feasibility Assessment: Sparse Matrices, Symbolic Input, Missing Data

**Author:** Max Moldovan
**Date:** 2026-04-10
**Package:** gretaR v0.1.0-dev
**Status:** Pre-Phase 2 assessment

---

## Executive Summary

| Feature | Recommendation | Timing | Rationale |
|---|---|---|---|
| **Sparse Matrices** | Include (targeted) | Phase 2 (mid-stage) | High value for large-scale models; R torch COO support exists; Matrix is a zero-cost dependency. But kernel/Gram matrices are rarely truly sparse — focus on sparse *design matrices*, not sparse kernels. |
| **Symbolic Input (formula)** | Include (layered) | Phase 2-3 (mid-to-late) | High usability payoff; build a `bf()`-style formula layer *on top* of the core DSL, not as a replacement. Use `model.matrix()` internally. Do not adopt rlang/tidyselect. |
| **Missing Data** | Exclude from core | Never (core) | Keep out of the inference engine. Recommend external preprocessing. Optionally add `na.action`-style guards and informative errors in Phase 2. Consider a `mi()`-style formula extension in Phase 3 only if formula interface is built first. |

**Overall strategy:** gretaR's core value is the programmatic DSL + torch inference engine. Protect that simplicity. Add sparse matrix support as an optimisation layer. Add formula interface as a convenience layer. Keep missing data handling external.

---

## 1. Sparse Matrices Integration

### 1.1 Benefits

| Benefit | Impact | Who benefits |
|---|---|---|
| **Memory efficiency for large design matrices** | High | Users with high-dimensional predictors (genomics, NLP, spatial) |
| **Faster matrix-vector products** | Medium-High | Models with `X %*% beta` where X is sparse |
| **Enables larger models** | High | Currently limited by dense `torch_tensor` memory |
| **Ecosystem alignment** | Medium | `Matrix::dgCMatrix` is the R standard; 254+ packages depend on it |

**Critical nuance:** Kernel/Gram matrices (K(x_i, x_j)) are almost never sparse. They are dense n×n matrices by construction. Sparse matrix support benefits *design matrices* (X), *precision matrices* (Q), and *adjacency matrices* — not kernel matrices directly. For large-scale kernel methods, the correct optimisation is *low-rank approximation* (Nyström, inducing points), not sparsity.

### 1.2 Technical Feasibility

**R torch sparse support exists but is immature:**

| Capability | Status | Risk |
|---|---|---|
| `torch_sparse_coo_tensor()` | Works, exported | Low |
| Sparse × Dense matmul | Works | Low |
| Sparse + Sparse | Works | Low |
| Dense → Sparse (`$to_sparse()`) | **Segfaults on Apple Silicon** | High |
| CSR format | Beta, printing crashes | Medium |
| `Matrix::dgCMatrix` → torch | No built-in conversion | Low (easy to write) |

**Conversion path (reliable):**
```
dgCMatrix → as(x, "TsparseMatrix") → dgTMatrix
dgTMatrix → torch_sparse_coo_tensor(indices, values, size)
```

**Implementation estimate:** ~2-3 days for:
- `as_data.sparseMatrix()` S3 method
- Internal conversion functions (`sparse_to_torch()`, `torch_to_sparse()`)
- Sparse-aware `%*%` dispatch in `gretaR_matmul()`

**Dependencies:** `Matrix` package (recommended, ships with R, zero install burden, safe for CRAN).

### 1.3 Trade-offs

| Pro | Con |
|---|---|
| Enables models with 10k+ predictors | Adds code paths that must be tested on dense AND sparse |
| Zero dependency cost (Matrix is recommended) | R torch sparse support is immature — potential upstream bugs |
| Natural extension of existing `as_data()` API | Sparse autograd is less tested than dense — gradient bugs possible |
| Aligns with glmnet, lme4 ecosystem | Kernel/Gram matrices remain dense — doesn't solve the n² scaling problem |

**CRAN impact:** None. Matrix is a recommended package. Adding `importFrom(Matrix, sparseMatrix)` is standard practice.

### 1.4 Recommended Timing: **Phase 2 (Months 4-5)**

**Why not Phase 1:** The current DSL handles dense matrices correctly. Sparse support is an optimisation, not a correctness requirement.

**Why not Phase 3:** Users fitting GLMs with many predictors will hit memory limits quickly. This is a practical blocker for adoption.

**Implementation sequence:**
1. Add `Matrix` to `Imports` in DESCRIPTION
2. Implement `as_data.dgCMatrix()` → sparse torch tensor
3. Add sparse-aware path in `gretaR_matmul()` (dispatch on input type)
4. Test gradient flow through sparse matmul
5. Benchmark: sparse vs dense for `X %*% beta` at various dimensions
6. Document in vignette: "Working with Large Datasets"

### 1.5 Alternatives if Excluded

- Users convert sparse matrices to dense before passing to `as_data()` (works but defeats the purpose)
- For very large problems, recommend `brms`/`cmdstanr` which handle sparse design matrices natively via Stan's `csr_matrix_times_vector()`

---

## 2. Symbolic Input Support (Formula Interface)

### 2.1 Benefits

| Benefit | Impact | Who benefits |
|---|---|---|
| **Lower barrier to entry** | Very High | R users familiar with `lm(y ~ x)` syntax |
| **Automatic design matrix construction** | High | Handles factors, interactions, intercepts automatically |
| **Closer parity with brms** | High | brms users evaluating gretaR as alternative |
| **Self-documenting models** | Medium | Formula captures model structure in one line |

### 2.2 Technical Feasibility

**Approach:** Build a formula layer *on top* of the existing DSL, not as a replacement. The DSL is gretaR's core; the formula interface is syntactic sugar.

**Architecture:**

```
User writes:        gretaR_model(y ~ x1 + x2, data = df,
                                 prior = list(beta = normal(0, 5)))
                                      │
                                      ▼
Formula parser:     model.matrix(~ x1 + x2, data = df)  →  design matrix X
                    Intercept column added automatically
                    Factors expanded to dummy variables
                                      │
                                      ▼
DSL translation:    x <- as_data(X)
                    y <- as_data(df$y)
                    beta <- normal(0, 5, dim = ncol(X))
                    sigma <- half_cauchy(2)
                    mu <- x %*% beta
                    distribution(y) <- normal(mu, sigma)
                    m <- model(beta, sigma)
```

**Key design decisions:**

| Decision | Recommended | Rationale |
|---|---|---|
| Use `model.matrix()` for linear terms | Yes | Battle-tested, handles contrasts, factors, interactions. Every major R package uses it. |
| Use rlang/tidyselect | **No** | No major statistical package has adopted it for formula parsing. Adds dependency for no proven benefit. |
| Support `bf()`-style multi-part formulas | Phase 3 | Distributional regression (`sigma ~ x1`) requires brms-like `bf()`. Complex but high value. |
| Support `s()` smooth terms | Phase 3+ | Requires basis function expansion. Can integrate with mgcv. Very complex. |
| Support random effects `(1|group)` | Phase 3 | Requires hierarchical model scaffolding (plates, indexing). Phase 2 prerequisite. |

**Implementation estimate:**
- Basic `gretaR_model(y ~ x, data)`: ~1 week
- Prior specification DSL: ~3-5 days
- Family support (gaussian, binomial, poisson): ~1 week
- Random effects: ~2-3 weeks (depends on hierarchical model infrastructure)

**Dependencies:** None beyond base R (`stats::model.matrix`). Optional `rlang` if we want non-standard evaluation sugar, but this is not recommended.

### 2.3 Trade-offs

| Pro | Con |
|---|---|
| Dramatically lowers learning curve | Two APIs to maintain (formula + DSL) |
| Familiar to 90% of R statisticians | Formula interface imposes structure — loses DSL flexibility |
| Enables quick model specification for standard models | Risk of trying to replicate brms (scope creep) |
| Attracts users from lm/glm/brms | Must decide on prior specification syntax (no R standard) |

**Key risk:** Scope creep. brms has had 8+ years of development to support its formula interface. gretaR should offer a *basic* formula interface for standard GLMs and point users to the DSL for anything non-standard.

**API stability:** Formula interface is a higher-level API that should be clearly separate from the core DSL. Changes to the formula interface should not break the DSL. Use a separate file/module (`R/formula.R`).

### 2.4 Recommended Timing: **Phase 2 (basic) + Phase 3 (advanced)**

**Phase 2 (Months 5-6):** Basic formula interface for GLMs
- `gretaR_glm(y ~ x1 + x2, data, family = "gaussian")`
- Automatic intercept, design matrix via `model.matrix()`
- Default priors with user override
- Families: gaussian, binomial, poisson

**Phase 3 (Months 8-10):** Advanced formula features
- `bf()`-style distributional formulas
- Random effects `(1 | group)`
- Smooth terms (requires mgcv integration)

### 2.5 Alternatives if Excluded

- Users specify models via the DSL (current approach — more flexible but higher learning curve)
- Provide helper functions: `design_matrix(formula, data)` that returns a gretaR_array ready for `%*%`
- Point users to brms for formula-based models, position gretaR as the "programmatic" alternative

---

## 3. Missing Data Handling

### 3.1 Benefits

| Benefit | Impact | Who benefits |
|---|---|---|
| Convenience (no preprocessing) | Medium | Users with messy real-world data |
| Bayesian imputation (treating missing as parameters) | High | Principled missing data handling |
| Reduced user error | Low-Medium | Prevents silent NA propagation in torch |

### 3.2 Technical Feasibility

**Three approaches, ranked by complexity:**

#### A. Guard-and-error (trivial)
```r
as_data <- function(x) {
  if (anyNA(x)) cli_abort("NA values detected. Remove or impute before modelling.")
  ...
}
```
Effort: 30 minutes. **Already the recommended minimum.**

#### B. Complete-case / masking (moderate)
```r
as_data <- function(x, na.action = c("error", "omit", "mask")) {
  na.action <- match.arg(na.action)
  if (na.action == "omit") x <- na.omit(x)
  if (na.action == "mask") {
    mask <- !is.na(x)
    x[is.na(x)] <- 0  # placeholder
    attr(result, "mask") <- mask
  }
  ...
}
```
Effort: 1-2 days. Masking requires likelihood to only sum over non-missing observations.

#### C. Bayesian imputation (complex — treat missing as parameters)
This is the JAGS/NIMBLE approach: missing values become latent variables sampled alongside model parameters.

```r
y <- as_data(y_obs_with_NA)  # NAs detected
# Internally: split into observed and missing indices
# Missing values become variable() nodes
# Sampled via HMC alongside other parameters
```

**Effort:** 2-4 weeks. Requires:
- Splitting data into observed/missing index vectors
- Creating variable nodes for missing entries
- Modifying log_prob to handle partially-observed data
- Careful handling of constraints (missing counts must be non-negative, etc.)
- Testing convergence properties with augmented parameter space

**Comparison with major Bayesian engines:**

| Engine | Approach | Built-in? | Effort to implement |
|---|---|---|---|
| JAGS/NIMBLE | NA → latent variable | Yes | Already done for them |
| Stan | Explicit modelling required | No | User's job |
| brms | `mi()` formula extension | Semi-automated | brms generates Stan code |
| gretaR (proposed) | Guard-and-error (Phase 2), optional imputation (Phase 3+) | No (recommended) | See above |

### 3.3 Trade-offs

| Pro | Con |
|---|---|
| Bayesian imputation is principled and powerful | Adds significant complexity to the inference engine |
| JAGS-like missing data handling would be a differentiator | Augments parameter space → slower sampling, harder convergence |
| Reduces data preprocessing burden | Every distribution must handle partial observations |
| | Testing burden: must verify correctness with missing data patterns |
| | CRAN review: handling missing data incorrectly is worse than not handling it |

**Critical concern:** Implementing Bayesian imputation incorrectly gives users **wrong answers silently**. This is far worse than refusing to handle NAs. The risk/reward ratio is unfavourable for early development.

### 3.4 Recommended Timing: **Exclude from core**

**Phase 2 (Month 4): Guard-and-error only**
- Add `NA` checks in `as_data()` with informative error messages
- Document recommended preprocessing workflows in vignette
- Point users to `mice`, `missRanger`, `naniar`, `tidyr::drop_na()`

**Phase 3 (Month 9+): Optional, only if formula interface exists**
- If a `bf()`-style formula interface is implemented, consider a `mi()` extension (following brms)
- This would generate the DSL code that handles missing data — keeping the core engine clean

**Never in core engine:**
- Do not modify `log_prob()` to handle NAs natively
- Do not add latent variable imputation to the sampler
- Keep the torch tensor pipeline NA-free

### 3.5 Alternatives (recommended user workflow)

```r
# Recommended approach: preprocess before gretaR
library(mice)
imputed <- complete(mice(raw_data, m = 1))

# Or: complete cases
clean_data <- na.omit(raw_data)

# Or: domain-specific imputation
library(missRanger)
imputed <- missRanger(raw_data)

# Then use gretaR
y <- as_data(imputed$y)
x <- as_data(imputed$x)
...
```

For users needing principled Bayesian missing data handling, recommend **brms** (which automates `mi()`) or **JAGS/NIMBLE** (which handle it natively).

---

## Prioritised Roadmap Integration

```
PHASE 1 (Current — Complete)
  ✅ Core DSL, 12 distributions, HMC/NUTS, tests, vignette

PHASE 2 (Months 4-6)
  Month 4:
    ├── P1 distributions (Dirichlet, NegBin, LogNormal, Cauchy, Wishart, LKJ)
    ├── Hierarchical model support (plates, indexing, [.gretaR_array)
    ├── NA guard-and-error in as_data()                          ← MISSING DATA
    └── roxygen2 docs, CRAN prep

  Month 5:
    ├── Sparse matrix support via Matrix package                 ← SPARSE MATRICES
    │   ├── as_data.dgCMatrix() → sparse torch tensor
    │   ├── Sparse-aware %*% dispatch
    │   └── Benchmark and validate gradients
    ├── Variational inference (ADVI)
    └── MAP estimation

  Month 6:
    ├── Basic formula interface: gretaR_glm(y ~ x, data)        ← SYMBOLIC INPUT
    │   ├── model.matrix() for design matrix construction
    │   ├── Default priors with user override
    │   └── Families: gaussian, binomial, poisson
    ├── pkgdown website, benchmarks
    └── CRAN submission

PHASE 3 (Months 7-12)
  ├── Advanced formula: bf()-style distributional, random effects
  ├── Optional mi() for missing predictors (formula layer only)
  ├── Mixture model helpers
  ├── GP module, ODE module
  ├── GPU benchmarking
  └── JOSS paper
```

---

## Architectural Adjustments

### For Sparse Matrices

1. **Add `Matrix` to `Imports` in DESCRIPTION** — zero-cost, safe for CRAN
2. **Add S3 method dispatch in `as_data()`:**
   ```r
   as_data.default()        # current numeric path
   as_data.dgCMatrix()      # sparse → torch_sparse_coo_tensor
   as_data.dgTMatrix()      # triplet → torch_sparse_coo_tensor
   ```
3. **Modify `gretaR_matmul()`** to check if either input is sparse and use `torch_mm()` (which handles sparse×dense natively)
4. **No changes to model.R or inference code** — the log_prob pipeline operates on torch tensors regardless of sparsity. This is a key advantage of the torch backend.

### For Formula Interface

1. **Create `R/formula.R`** as a separate module — formula layer calls into DSL, not vice versa
2. **Do not modify the DSL** — the formula interface is a *consumer* of the DSL, not part of it
3. **Export `gretaR_glm()` alongside existing `model()`** — two entry points for different user profiles
4. **Prior specification:** Use a named list: `prior = list(beta = normal(0, 5), sigma = half_cauchy(2))`

### For Missing Data

1. **Add `NA` check in `as_data()`** (guard-and-error) — 5 lines of code
2. **No architectural changes** to model.R, inference, or distributions
3. **Document** recommended preprocessing in vignette and function help

---

## Risk Matrix

| Feature | Implementation Risk | Correctness Risk | Maintenance Burden | Recommendation |
|---|---|---|---|---|
| Sparse matrices | Low-Medium (R torch sparse is beta) | Medium (gradient verification needed) | Low (isolated code path) | **Include Phase 2** |
| Formula interface (basic) | Low | Low (translates to tested DSL) | Medium (two APIs) | **Include Phase 2** |
| Formula interface (advanced) | High | Medium | High | **Phase 3, cautiously** |
| Missing data (guard) | Negligible | Negligible | Negligible | **Include Phase 2** |
| Missing data (Bayesian) | Very High | Very High | Very High | **Exclude** |

---

*Assessment complete. Awaiting review before proceeding to Phase 2 implementation.*
