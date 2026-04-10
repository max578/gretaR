# Numerical Study Plan v1.0: Benchmarking R Bayesian and GLMM Packages

**Version:** 1.0 (revised)
**Date:** 2026-04-10
**Author:** Max Moldovan

---

## Critical Review of v0

The original plan (v0) was ambitious and well-structured but had several practical issues:

| Issue | Problem | Fix in v1 |
|---|---|---|
| **Too many packages** | 15+ packages listed, many redundant or non-Bayesian (asreml, lme4, glmmTMB, sommer are frequentist). Mixing Bayesian and frequentist tools in a Bayesian benchmark muddles the message. | Split into Bayesian core (6 packages) + frequentist reference (4 packages). Drop truly redundant ones. |
| **Unrealistic scenario S4** | 100M rows is not feasible for MCMC on any free hardware. Even 1M rows with HMC is hours per chain. | Cap at 500K rows. S4 is GPU-focused on model complexity, not raw data size. |
| **Missing cost analysis** | "High-end server with A100" mentioned but no costing or free alternatives identified. | Full free-compute resource table included. |
| **Vague model definitions** | "Use the complete range from Excel file" — no concrete model specifications in the plan itself. | All models explicitly defined with formulas and priors. |
| **No power/sample-size justification** | Why 3 reps? Why 2000 iterations? No statistical justification. | 5 reps for timing (reduce variance); 2000 iterations is standard but justified. |
| **Missing convergence criteria** | Plan collects R-hat but doesn't define what constitutes a "successful" fit. | R-hat < 1.05, ESS > 400, no divergences = success. |
| **Delivery timeline too optimistic** | 3-7 days for the entire study including implementation. | 2-3 weeks realistic. Phased delivery. |

---

## 1. Objectives

1. **Correctness**: Verify gretaR produces valid posteriors by cross-validating against Stan on all models.
2. **Performance**: Measure wall-clock time, ESS/second, and peak RAM across model complexity and sample sizes.
3. **Scalability**: Identify where each package breaks down (data size, parameter count, model complexity).
4. **Usability**: Compare lines of code, installation effort, and API ergonomics.
5. **Publication**: Produce a reproducible, publication-ready report (JOSS companion or standalone).

---

## 2. Packages

### Bayesian (core comparison)

| Package | Backend | Why include |
|---|---|---|
| **gretaR** | torch (R) | Our package — the subject |
| **brms/cmdstanr** | Stan (C++) | Gold standard Bayesian GLMMs |
| **greta** | TensorFlow (Python) | Direct predecessor to gretaR |
| **NIMBLE** | C++ codegen | Programmable MCMC, different architecture |
| **INLA** | Laplace approx | Fast approximate Bayesian, no MCMC |
| **rstanarm** | Stan (pre-compiled) | Fastest Stan-based option for standard models |

### Frequentist (reference only, for speed/accuracy calibration)

| Package | Why include |
|---|---|
| **lme4** | Industry standard for LMMs/GLMMs |
| **glmmTMB** | Broader family support than lme4 |
| **mgcv** | GAMMs, smooth terms |

**Dropped from v0**: asreml (commercial, not reproducible), sommer (niche), rjags/runjags (redundant with NIMBLE), BGLR (genomics-specific).

### Pre-flight check

Before running any benchmark, verify each package is installed and functional with a toy model. Drop any that fail, with documented reason.

---

## 3. Models (10 benchmark models)

All models specified with explicit formulas, data-generating process, and priors.

### M1: Simple LMM (random intercepts)

$$y_{ij} = \mu + \alpha_i + \epsilon_{ij}, \quad \alpha_i \sim N(0, \tau^2), \quad \epsilon_{ij} \sim N(0, \sigma^2)$$

- **Priors**: $\mu \sim N(0, 10)$, $\tau \sim \text{HalfCauchy}(2)$, $\sigma \sim \text{HalfCauchy}(2)$
- **Data**: $n = 1000$, $J = 20$ groups

### M2: LMM with random slopes

$$y_{ij} = (\mu + \alpha_i) + (\beta + b_i) x_{ij} + \epsilon_{ij}$$

- Random intercepts and slopes with covariance
- **Data**: $n = 2000$, $J = 30$ groups

### M3: Crossed random effects

$$y_{ijk} = \mu + \alpha_i + \beta_j + \epsilon_{ijk}$$

- Two crossed grouping factors (e.g., student × school)
- **Data**: $n = 5000$, $J_1 = 50$, $J_2 = 20$

### M4: Poisson GLMM

$$y_{ij} \sim \text{Poisson}(\exp(\mu + \alpha_i + \beta x_{ij})), \quad \alpha_i \sim N(0, \tau^2)$$

- Log link, count data
- **Data**: $n = 2000$, $J = 25$ groups

### M5: Logistic GLMM

$$y_{ij} \sim \text{Bernoulli}(\text{logit}^{-1}(\mu + \alpha_i + \beta x_{ij}))$$

- Logit link, binary outcomes
- **Data**: $n = 3000$, $J = 40$ groups

### M6: Negative Binomial GLMM

$$y_{ij} \sim \text{NegBin}(\mu_{ij}, \phi), \quad \log \mu_{ij} = \alpha_i + \beta x_{ij}$$

- Overdispersed counts
- **Data**: $n = 2000$, $J = 20$ groups

### M7: Robust regression (Student-t errors)

$$y_{ij} \sim t_\nu(\mu + \alpha_i + \beta x_{ij}, \sigma), \quad \nu \sim \text{Gamma}(2, 0.1)$$

- Degrees of freedom estimated from data
- **Data**: $n = 1000$, $J = 15$ groups, 5% outliers

### M8: Multiple regression + hierarchical

$$y_{ij} = \alpha_i + X_{ij} \beta + \epsilon_{ij}, \quad \alpha_i \sim N(\mu_\alpha, \tau^2)$$

- 5 predictors, random intercepts
- **Data**: $n = 10000$, $J = 100$ groups

### M9: Large-scale random effects

Same as M1 but:
- **Data**: $n = 100000$, $J = 2000$ groups
- Tests scalability

### M10: High-dimensional crossed effects

Same as M3 but:
- **Data**: $n = 50000$, $J_1 = 500$, $J_2 = 100$
- Tests scalability of crossed RE

---

## 4. Benchmark Scenarios

| Scenario | Models | n | Groups | Chains × Iter | Purpose |
|---|---|---|---|---|---|
| **S1: Quick** | M1, M4, M5 | 1K-3K | 20-40 | 2 × 1000 | Sanity check, all packages |
| **S2: Standard** | M1-M8 | 1K-10K | 15-100 | 4 × 2000 | Core comparison |
| **S3: Scale** | M9, M10 | 50K-100K | 500-2000 | 2 × 1000 | Scalability limits |

- **Repetitions**: 5 per scenario (report median ± MAD)
- **Convergence criteria**: R-hat < 1.05, bulk ESS > 400, zero divergences
- **Timeout**: 2 hours per model × package. If exceeded, report "DNF" (did not finish)

---

## 5. Metrics

| Category | Metric | How measured |
|---|---|---|
| **Correctness** | Posterior mean bias (vs Stan reference) | $|\hat\theta - \theta_{\text{Stan}}|$ |
| **Correctness** | 95% CI coverage | Proportion of true values in posterior 95% interval |
| **Convergence** | R-hat, bulk ESS, tail ESS | `posterior::summarise_draws()` |
| **Speed** | Wall-clock time (seconds) | `system.time()` or `proc.time()` |
| **Efficiency** | ESS per second | $\text{min\_ESS} / \text{time}$ |
| **Memory** | Peak RAM (MB) | `bench::bench_process_memory()` or `/proc/self/status` |
| **Usability** | Lines of code | Count of non-comment R lines |
| **Failures** | Divergences, timeouts, crashes | Logged per run |

---

## 6. Free Computational Resources

| Platform | Cost | CPU | GPU | RAM | Time Limit | Best for |
|---|---|---|---|---|---|---|
| **Adelaide Phoenix HPC** | Free (staff) | Multi-node | V100/A100 | 128-384 GB | 3-7 days | **Primary: all S1-S3 benchmarks** |
| **Kaggle Notebooks** | Free | 4 vCPU | 2×T4 (30hr/wk) | 30 GB | 9-12 hr | **Best free R + GPU torch** |
| **GitHub Actions** | Free (public repo) | 4 vCPU | None | 16 GB | 6 hr/job | Automated CI benchmarks |
| **NCI Gadi** | Free (application) | 48 cores/node | V100 (32 GB) | 192 GB | 48 hr | Scale-up if Phoenix insufficient |
| **NECTAR Research Cloud** | Free (AU researchers) | 2+ vCPU | Limited | 8+ GB | Continuous | Persistent VM for dashboards |
| **Google Colab** | Free (rationed) | ~2 vCPU | T4 (limited) | 12.7 GB | ~12 hr | Quick prototyping only |

### Recommended Strategy

1. **Develop and debug** on local Mac (Apple Silicon) — S1 scenarios only.
2. **Run S1 and S2** on Adelaide Phoenix HPC — submit SLURM batch jobs, 4 parallel jobs.
3. **Run S3** on Phoenix with GPU nodes or NCI Gadi.
4. **Kaggle Notebooks** for reproducible, shareable S1 demos that anyone can re-run.
5. **GitHub Actions** for automated regression: re-run S1 benchmarks on every commit.

### Estimated Compute Budget

| Scenario | Models | Per-model time | Total (with 5 reps × 9 packages) | Resource |
|---|---|---|---|---|
| S1 | 3 | ~2-10 min | ~5-8 hours | Local Mac or Phoenix |
| S2 | 8 | ~5-60 min | ~30-60 hours | Phoenix (parallel) |
| S3 | 2 | ~30-120 min | ~15-30 hours | Phoenix GPU or NCI |

**Total: ~50-100 compute-hours** — well within Phoenix free allocation.

---

## 7. Implementation Structure

```
numerical_study/
├── R/
│   ├── 00_setup.R              # Install/check all packages
│   ├── 01_data_generators.R    # Parameterised data generation
│   ├── 02_model_wrappers.R     # One function per package per model
│   ├── 03_run_benchmarks.R     # Main benchmark runner
│   ├── 04_collect_results.R    # Parse outputs, compute metrics
│   └── 05_generate_report.R    # Tables, figures, report
├── data/                       # Generated datasets (gitignored)
├── results/                    # Raw results (RDS files)
├── figures/                    # Publication-quality plots
├── slurm/                      # SLURM job scripts for Phoenix
├── report/
│   ├── benchmark_report.qmd    # Quarto report
│   └── references.bib
└── README.md
```

### Model Wrapper Pattern

Each package gets a wrapper function with a consistent interface:

```r
fit_model_brms <- function(model_id, data, n_chains, n_iter, n_warmup) {
  t0 <- proc.time()
  fit <- brms::brm(formula, data, chains = n_chains, iter = n_iter, ...)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  list(
    package = "brms",
    model_id = model_id,
    time = elapsed,
    summary = posterior::summarise_draws(posterior::as_draws_array(fit)),
    n_divergences = ...,
    converged = ...,
    peak_ram = ...
  )
}
```

---

## 8. Execution Phases

| Phase | Duration | Tasks |
|---|---|---|
| **1: Setup** | 2 days | Install all packages, verify, write data generators |
| **2: Wrappers** | 3-4 days | Implement model wrappers for all 9 packages × 10 models |
| **3: S1 runs** | 1 day | Quick benchmarks, debug any failures |
| **4: S2 runs** | 2-3 days | Core comparison on Phoenix |
| **5: S3 runs** | 1-2 days | Scale-up benchmarks on Phoenix GPU / NCI |
| **6: Analysis** | 2 days | Compile results, generate figures, write report |

**Total: 2-3 weeks** (realistic, including iteration on failures).

---

## 9. Deliverables

1. **Reproducible benchmark suite** — `numerical_study/` directory with all code, data generators, and SLURM scripts.
2. **Publication-ready report** — Quarto document with executive summary, comparison tables, and figures.
3. **Raw results** — RDS files for all runs, enabling re-analysis.
4. **Kaggle Notebook** — Shareable S1 benchmark that anyone can re-run for free.
5. **GitHub Actions workflow** — Automated S1 regression on every push.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Package installation failures | Pre-flight check script; document and exclude failures |
| MCMC non-convergence | Increase warmup; use non-centred parameterisations; report DNF |
| Timeout on large models | 2-hour cap; report partial results |
| Phoenix queue wait times | Submit jobs in off-peak hours; use short-queue for S1 |
| Irreproducible results (random seeds) | Fixed `set.seed()` per model; save all RNG states |
| greta TF installation failure | Document the failure — this IS part of the comparison (installation fragility) |

---

## 11. Publication Strategy

The benchmark results support two publication outlets:

1. **JOSS paper for gretaR** — validation section references these benchmarks
2. **Standalone methods comparison paper** — target: Journal of Statistical Software, R Journal, or Computational Statistics & Data Analysis

The report should be written to serve both purposes: gretaR-focused narrative with honest, comprehensive comparison.

---

*Plan approved and ready for implementation.*
