# gretaR: Development Plan v1.0

**Author:** Max Moldovan
**Date:** 2026-04-10
**Status:** Draft — awaiting review and approval

---

## Executive Summary

Build **gretaR** — a torch-based Bayesian probabilistic programming package for R that fills the gap left by greta's decline. The package provides an elegant R-native DSL for defining Bayesian models, compiled to torch tensors for GPU-accelerated HMC/NUTS inference — with **zero Python dependency**.

This is not a greta clone. It is a ground-up redesign informed by greta's strengths (R-native syntax, lazy graph construction) and weaknesses (TF/Python fragility, limited inference, stalled development), built on a modern, actively-maintained backend (R torch).

---

## 1. Objective

| Aspect | Detail |
|---|---|
| **Goal** | A production-quality R package for Bayesian inference using torch autograd |
| **Target users** | Applied statisticians, ecologists, biomedical researchers, data scientists who want R-native Bayesian modelling without Stan syntax or Python dependencies |
| **Success criteria** | (1) Can reproduce all standard greta examples; (2) HMC/NUTS produces correct posteriors validated against Stan; (3) Installable with `install.packages("gretaR")` + `torch::install_torch()`; (4) Passes `R CMD check --as-cran` with zero warnings |
| **Non-goals (Phase 1)** | Multi-backend abstraction, JAX/TF support, discrete parameter samplers, GP/ODE modules |

---

## 2. Critical Assessment of the Original Draft

The original plan has the right vision but several unrealistic elements that would sink execution:

### What to keep
- R-native DSL with lazy graph construction — proven by greta, loved by users
- torch as primary backend — correct choice (active development, no Python, full autograd)
- HMC/NUTS as core inference — the right starting point

### What to cut or defer

| Original proposal | Problem | Decision |
|---|---|---|
| "Exact syntax compatibility with greta" | greta has ~600 stars and is declining. Locking ourselves to its API limits design. Users need *easy migration*, not carbon-copy syntax. | **Inspire, don't clone.** Adopt greta's best patterns (lazy arrays, `distribution()`, piped syntax) but design a cleaner API from scratch. Provide a migration vignette. |
| TensorFlow fallback backend | Defeats the entire motivation. Doubles maintenance for a dying ecosystem. | **Cut entirely.** Single backend: torch. |
| "Backend abstraction / pluggable architecture" | Classic over-engineering. Premature abstraction for backends that may never materialise. | **Cut.** Design clean internals, but no abstract backend interface. Refactor later if needed. |
| "2-5x speedup over greta on GPU" | Unsubstantiated. greta's overhead is reticulate, not TF compute. For pure HMC, torch vs TF performance is comparable. | **Don't promise speedups.** Claim: "eliminates Python overhead, enables GPU acceleration, competitive with Stan on large models." Benchmark honestly. |
| "Full discrete parameter support" | This is an open research problem (beyond reparameterisation tricks). Not an engineering task. | **Defer to Phase 2.** Note: marginalization helpers for common discrete cases (mixture models) can come in Phase 1.5. |
| "1-2 month prototype" | Implementing a distributions library + HMC + DSL in 1-2 months is unrealistic for quality work. | **Revised timeline below.** |
| "Hybrid samplers, Stan backend plugin, ONNX export" | Scope creep. Each is a project in itself. | **Defer to Phase 3+.** |

### The biggest risk the original plan missed

**R torch has only 8 probability distributions** (Normal, Bernoulli, Categorical, Chi2, Gamma, MixtureOfSameFamily, MultivariateNormal, Poisson). PyTorch has 30+. For a Bayesian PPL, we need at minimum ~20 distributions with:
- `log_prob()` methods (differentiable)
- Parameter constraints and transforms (bijectors)
- Sampling support

**This is the single largest implementation effort** and must be prioritised in Phase 1.

---

## 3. Technical Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────┐
│  User-Facing DSL Layer                      │
│  gretaR_array objects (R6/S7)               │
│  variable(), distribution(), as_data()      │
│  Standard R ops overloaded (+, *, [, etc.)  │
└──────────────────┬──────────────────────────┘
                   │ builds
┌──────────────────▼──────────────────────────┐
│  Computation Graph (DAG)                    │
│  Nodes: data, variable, operation, distrib  │
│  Tracks dependencies, shapes, constraints   │
│  Constructs log-joint-density function      │
└──────────────────┬──────────────────────────┘
                   │ compiles to
┌──────────────────▼──────────────────────────┐
│  Torch Execution Layer                      │
│  log_prob as torch function (autograd-able) │
│  Parameter transforms (constrained ↔ free)  │
│  Gradient computation via torch::autograd   │
└──────────────────┬──────────────────────────┘
                   │ feeds into
┌──────────────────▼──────────────────────────┐
│  Inference Engines                          │
│  Phase 1: HMC, NUTS                        │
│  Phase 2: ADVI, MAP, Laplace               │
│  Diagnostics: R-hat, ESS, trace plots      │
└─────────────────────────────────────────────┘
```

### 3.2 Core Components

#### A. Distributions Module (Priority 1 — the critical gap)

Implement a `gretaR.distributions` sub-module wrapping torch tensors with:

| Priority | Distribution | Parameters | Notes |
|---|---|---|---|
| P0 | Normal | mean, sd | Already in torch |
| P0 | HalfNormal | sd | Truncated normal transform |
| P0 | HalfCauchy | scale | Common default prior |
| P0 | StudentT | df, loc, scale | Robust regression |
| P0 | Uniform | lower, upper | Already partially in torch |
| P0 | Bernoulli | prob/logit | Already in torch |
| P0 | Binomial | size, prob | Missing from R torch |
| P0 | Poisson | rate | Already in torch |
| P0 | Gamma | shape, rate | Already in torch |
| P0 | Beta | alpha, beta | Missing — critical for proportions |
| P0 | Exponential | rate | Missing |
| P0 | MultivariateNormal | mean, covariance | Already in torch |
| P1 | Dirichlet | concentration | Missing — needed for categorical models |
| P1 | NegativeBinomial | size, prob | Missing — overdispersed counts |
| P1 | LogNormal | meanlog, sdlog | Transform of Normal |
| P1 | Cauchy | location, scale | |
| P1 | Wishart / InvWishart | df, scale | Covariance priors |
| P1 | LKJCholesky | eta | Correlation matrix prior |
| P2 | VonMises | mu, kappa | Circular data |
| P2 | Categorical | probs | Already in torch |
| P2 | Multinomial | probs, total_count | Missing |

Each distribution must provide:
- `log_prob(x)` — differentiable via torch autograd
- `sample(n)` — for prior predictive checks
- Constraint info (e.g., `lower = 0` for Gamma)
- Default bijector for unconstrained parameterisation (log, logit, softmax, Cholesky, etc.)

**Implementation strategy:** Start with distributions already in R torch (wrap with our API). For missing ones, implement `log_prob` directly as torch tensor operations (they're mostly straightforward closed-form expressions). Do NOT try to port all of PyTorch's distributions module — implement only what we need, one at a time.

#### B. DSL / Array System

The user-facing object: `gretaR_array` (R6 class, migrate to S7 when stable).

```r
# Target API (inspired by greta, not identical)
library(gretaR)

# Data
x <- as_data(iris$Sepal.Length)
y <- as_data(iris$Sepal.Width)

# Priors
alpha <- normal(0, 10)
beta  <- normal(0, 5)
sigma <- halfcauchy(1)

# Likelihood
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)

# Fit
m <- model(alpha, beta, sigma)
draws <- mcmc(m, n_samples = 1000, warmup = 500)
```

Key design decisions:
- **Lazy evaluation**: operations build a DAG, not immediate computation
- **Shape inference**: automatic broadcasting following R/torch rules
- **Operator overloading**: `+`, `*`, `-`, `/`, `^`, `[`, `%*%` on gretaR_array
- **Math functions**: `log()`, `exp()`, `sqrt()`, `abs()`, `sum()`, `mean()`, etc.
- **Matrix ops**: `t()`, `chol()`, `solve()`, `crossprod()`

#### C. Computation Graph & Log-Joint Construction

- DAG nodes track: type (data/variable/operation/distribution), parents, shape, constraint
- `model()` call triggers graph compilation:
  1. Identify all free variables and their constraints
  2. Create bijector chain (constrained → unconstrained space)
  3. Build `log_joint(theta_free)` function: sum of all `log_prob` terms + Jacobian adjustments
  4. Return a compiled model object with `log_prob()` and `grad_log_prob()` methods
- The `log_joint` function must be a pure torch computation graph for autograd

#### D. Inference: HMC and NUTS

**Phase 1 implementation — static HMC:**
1. Leapfrog integrator using torch tensors
2. Dual averaging for step-size adaptation (Nesterov 2009)
3. Mass matrix estimation (diagonal, then dense)
4. Warmup with windowed adaptation (Stan-style)

**Phase 1b — NUTS:**
- Implement the No-U-Turn Sampler (Hoffman & Gelman 2014)
- Multinomial trajectory sampling (Betancourt 2017)
- Max tree depth control

**Implementation notes:**
- All sampling operations in torch tensor space (GPU-compatible)
- Multiple chains via parallel torch tensors (vectorised, not R-level parallelism)
- Return `posterior::draws_array` objects for direct compatibility with `posterior`, `bayesplot`, `tidybayes`

#### E. Diagnostics & Output

- Return `posterior::draws_array` or `posterior::draws_df` objects
- Leverage existing R ecosystem: `posterior` (R-hat, ESS, MCSE), `bayesplot` (visualisation), `loo` (model comparison)
- Built-in: `summary()`, `plot()`, `print()` methods
- Prior predictive and posterior predictive checks

---

## 4. Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| Backend | `torch` (R package) | Native C++, no Python, active Posit development, full autograd |
| Object system | R6 (Phase 1), S7 migration (Phase 2) | R6 is battle-tested for mutable state (DAG nodes). S7 when ecosystem matures. |
| Output format | `posterior::draws_array` | Interop with bayesplot, loo, tidybayes — the entire modern Bayesian R ecosystem |
| Testing | `testthat` 3e | Standard, with snapshot tests for numerical validation |
| Docs | `roxygen2` + `pkgdown` | Standard R package documentation |
| CI | GitHub Actions | R-CMD-check on Linux/macOS/Windows, GPU tests on self-hosted runner |
| Vignettes | Quarto (`.qmd`) | Modern, reproducible, publishable |

---

## 5. Step-by-Step Implementation Plan

### Phase 1: Foundation (Months 1-3)

**Goal:** A working package that can fit a Bayesian linear regression with HMC and produce correct posteriors.

| Step | Task | Est. Effort | Deliverable |
|---|---|---|---|
| 1.1 | Package skeleton: DESCRIPTION, NAMESPACE, R/, tests/, vignettes/ | 1 day | Clean `devtools::check()` pass |
| 1.2 | Distributions module: P0 distributions (Normal, HalfNormal, HalfCauchy, StudentT, Uniform, Bernoulli, Binomial, Poisson, Gamma, Beta, Exponential, MultivariateNormal) with `log_prob`, `sample`, constraints, bijectors | 2-3 weeks | Unit-tested distributions matching Stan/scipy reference values |
| 1.3 | `gretaR_array` class: creation, operator overloading, shape inference, DAG construction | 2-3 weeks | Can express `alpha + beta * x` as a DAG |
| 1.4 | `distribution()` assignment and `model()` compilation: log-joint construction, variable transforms, gradient verification | 2 weeks | `model()` returns object with correct `log_prob()` and `grad_log_prob()` |
| 1.5 | Static HMC sampler with dual averaging | 2 weeks | Correct posteriors for conjugate models (validate analytically) |
| 1.6 | NUTS sampler | 2 weeks | Validated against Stan on 3-5 benchmark models |
| 1.7 | Output integration: `posterior::draws_array`, `summary()`, `plot()` | 1 week | Works with bayesplot, posterior, loo |
| 1.8 | First vignette: "Getting Started with gretaR" (linear regression) | 3-5 days | Executable, reproducible vignette |
| 1.9 | Validation suite: compare gretaR posteriors to Stan on 10 benchmark models | 1 week | Numerical equivalence within MCMC error |

### Phase 2: Expansion (Months 4-6)

| Step | Task | Notes |
|---|---|---|
| 2.1 | P1 distributions (Dirichlet, NegBin, LogNormal, Cauchy, Wishart, LKJ) | Enables hierarchical and multivariate models |
| 2.2 | Hierarchical model support: plates/indexing, mixed effects | Critical for applied use |
| 2.3 | Variational inference (ADVI — automatic differentiation VI) | Mean-field and full-rank Gaussian approximation |
| 2.4 | MAP estimation and Laplace approximation | Quick model checks |
| 2.5 | Comprehensive vignettes: hierarchical models, GLMs, model comparison | 3-5 vignettes |
| 2.6 | Migration guide from greta | Lower adoption barrier |
| 2.7 | `pkgdown` website, hex logo, README | Public-facing polish |
| 2.8 | Performance benchmarking: gretaR vs Stan vs greta on standard models | Honest, published benchmarks |
| 2.9 | CRAN submission | Zero errors/warnings/notes |

### Phase 3: Advanced Features (Months 7-12)

| Task | Notes |
|---|---|
| Mixture model helpers (marginalising discrete components) | Practical discrete-ish support without full discrete samplers |
| Custom likelihood functions | User-defined `log_prob` in torch |
| GPU benchmarking and optimisation | Large-model / large-data use cases |
| Gaussian Process module (`gretaR.gp`) | Extension package |
| ODE-based models | Extension package |
| Community extension framework | Template + docs for user-contributed modules |
| S7 migration | When S7 is CRAN-stable and widely adopted |
| JOSS / R Journal publication | Academic credibility |

---

## 6. Key Design Decisions & Trade-offs

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| Single backend (torch only) | **Yes** | Multi-backend abstraction | Avoids premature abstraction. torch is actively maintained, no Python. If torch dies, refactor then. |
| R6 over S7 | **Phase 1** | S7 from start | S7 is not yet mature enough for production R6-style mutable state patterns. Plan migration path. |
| Own distributions module vs wrapping torch | **Own** | Pure torch wrappers | R torch only has 8 distributions. Writing `log_prob` in torch ops is straightforward and gives us full control. |
| Inspired by greta, not API-compatible | **Yes** | Exact greta API clone | Frees us to design a better API. greta's user base is small (~600 stars). Migration guide is sufficient. |
| `posterior::draws_array` output | **Yes** | Custom output format | Ecosystem interop (bayesplot, loo, tidybayes) is worth the dependency. Don't reinvent the wheel. |
| No reticulate anywhere | **Yes** | Optional Python interop | Clean break. The whole point is no Python. |
| Vectorised chains (torch-level) | **Yes** | R-level `parallel::mclapply` | Single torch tensor holding all chains. GPU-friendly, no IPC overhead. |

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **R torch distributions too sparse** | Certain | High | Build our own — `log_prob` implementations are closed-form, well-documented. Budget 2-3 weeks for P0 set. |
| **HMC/NUTS implementation bugs** | Medium | Critical | Validate against Stan on conjugate + non-conjugate benchmarks. Use Stan's adaptation algorithm as reference (Carpenter et al. 2017). |
| **torch R package instability / breaking changes** | Low | High | Pin torch version in DESCRIPTION. torch has Posit backing and stable API. |
| **Performance worse than Stan** | Medium | Medium | Acceptable for Phase 1 — correctness first. Stan has 12+ years of optimisation. Compete on ease-of-use and GPU scaling, not raw CPU speed. |
| **Low adoption** | Medium | Medium | Target greta refugees + users frustrated with Stan syntax. Publish benchmarks, vignettes, and a JOSS paper. Engage R-bloggers, Twitter/Mastodon, rOpenSci. |
| **Scope creep** | High | High | This plan. Ruthlessly defer anything not in Phase 1. |
| **Solo developer bottleneck** | Medium | Medium | Open-source early, seek contributors from R Bayesian community. Consider rOpenSci review for credibility. |

---

## 8. What This Package Is NOT

To maintain focus:

- **Not a Stan replacement.** Stan is a mature, optimised inference engine with its own language. gretaR targets users who want to stay in R.
- **Not brms.** brms provides formula-based model specification. gretaR is lower-level — define models programmatically. A future `brms`-like layer could sit on top of gretaR.
- **Not a deep learning framework.** We use torch for autograd and tensors, not for neural networks (though neural network likelihoods are possible in Phase 3).
- **Not trying to solve discrete parameters in Phase 1.** This is a research problem. We'll support marginalisation helpers for common cases.

---

## 9. Competitive Positioning

```
                    High-level syntax
                         │
                  brms   │
                    ●    │
                         │
                         │
Stan syntax ─────────────┼──────────── R-native syntax
required                 │
                         │
         cmdstanr ●      │      ● gretaR (target)
                         │      ● greta (declining)
                         │
                  NIMBLE ●│
                         │
                    Low-level / programmable
```

**gretaR's niche:** R-native programmable Bayesian modelling with GPU support. Between brms (higher-level, formula-based) and cmdstanr (lower-level, Stan syntax).

---

## 10. Validation Strategy

Correctness is non-negotiable for a Bayesian inference package.

1. **Conjugate models** — compare posterior moments to analytical solutions
2. **Stan cross-validation** — fit identical models in cmdstanr, compare posterior summaries (mean, sd, quantiles) within MCMC error
3. **Simulation-based calibration (SBC)** — Talts et al. (2018): verify posterior calibration across many simulated datasets
4. **Numerical gradient checks** — compare torch autograd gradients to finite-difference approximations
5. **Geweke test** — verify sampler targets the correct stationary distribution

---

## 11. Documentation Plan

| Deliverable | Phase | Description |
|---|---|---|
| Function documentation | 1 | roxygen2 for all exports, with executable examples |
| "Getting Started" vignette | 1 | Linear regression, interpretation, diagnostics |
| "Hierarchical Models" vignette | 2 | Random effects, partial pooling |
| "GLMs with gretaR" vignette | 2 | Logistic, Poisson, negative binomial regression |
| "Migrating from greta" guide | 2 | Side-by-side comparison, gotchas |
| "Custom Models" vignette | 2 | User-defined likelihoods, advanced usage |
| pkgdown website | 2 | Reference, articles, news |
| JOSS paper | 3 | Peer-reviewed software publication |
| Benchmarking report | 2 | Honest comparison with Stan, greta, NIMBLE |

---

## 12. Timeline Summary

```
Month 1-2:  Distributions + DSL + DAG + log-joint compilation
Month 2-3:  HMC + NUTS + output integration + validation
Month 3:    First vignette, internal release, validation suite
Month 4-5:  P1 distributions, hierarchical models, VI
Month 5-6:  Polish, benchmarks, CRAN submission
Month 7-12: Advanced features, GP/ODE extensions, JOSS paper
```

**Milestones:**
- **M1 (Month 3):** Working package — can fit linear + logistic regression with NUTS, validated against Stan
- **M2 (Month 6):** CRAN release — hierarchical models, VI, comprehensive vignettes, benchmarks
- **M3 (Month 12):** Mature package — extensions, community, JOSS publication

---

## 13. References

- Carpenter, B. et al. (2017). Stan: A probabilistic programming language. *J. Statistical Software*, 76(1).
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*, 15, 1593-1623.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
- Talts, S. et al. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv:1804.06788*.
- Kucukelbir, A. et al. (2017). Automatic differentiation variational inference. *JMLR*, 18(1), 430-474.
- Golding, N. (2019). greta: simple and scalable statistical modelling in R. *JOSS*, 4(40), 1601.

---


---

## Phase 2 Decisions (2026-04-10)

### Approved and implemented:

1. **Roxygen2 documentation** — Generated for all exported functions.

2. **Hierarchical models** — `[.gretaR_array` indexing implemented. Supports `alpha[group_id]` pattern for multi-level models with >3 parameters.

3. **P1 distributions** — Dirichlet, Negative Binomial, LKJ Correlation added.

4. **GitHub repo** — Private repo under `max578`, with GitHub Actions CI.

5. **Sparse matrices** — `Matrix::dgCMatrix` → torch sparse COO conversion via `as_data()`. Sparse-aware `%*%` dispatch. Zero-cost dependency (Matrix is recommended, ships with R).

6. **Formula interface** — Layered on top of DSL (not a replacement).
   - Phase 2: Basic `gretaR_glm(y ~ x, data, family)` with `model.matrix()` internally.
   - Phase 3: Advanced `bf()`-style distributional formulas, random effects.
   - Support formula styles from lme4, glmmTMB, asreml, mgcv, brms, rstanarm.
   - Auto-detection of formula style with optional explicit `style=` parameter.
   - **No** rlang or tidyselect.
   - Research LaTeX-style formula input for Phase 3.

7. **Missing data** — Excluded from core engine.
   - NA guard in `as_data()` with informative error message.
   - Documented: recommend `mice`, `missRanger`, `tidyr::drop_na()`.
   - Phase 3 optional: `mi()`-style extension if formula interface is built.

### Updated Roadmap:

```
Phase 2a (Current):
  ✅ Roxygen2 docs for all exports
  ✅ P1 distributions (Dirichlet, NegBin, LKJ)
  ✅ Hierarchical model indexing ([.gretaR_array)
  ✅ NA guard in as_data()
  ✅ Sparse matrix support (Matrix package)
  → GitHub repo + CI (max578, private)
  → Validate hierarchical models with MCMC
  → CRAN-ready check (zero errors/warnings)

Phase 2b (Next):
  → Variational inference (ADVI)
  → MAP estimation + Laplace approximation
  → Basic formula interface (gretaR_glm)
  → Comprehensive vignettes
  → pkgdown website
  → Benchmarking vs Stan/greta

Phase 3:
  → Advanced formula (bf(), random effects, smooth terms)
  → Optional mi() extension for missing data
  → Mixture model helpers
  → GP module, ODE module
  → LaTeX formula input research
  → JOSS paper
```
