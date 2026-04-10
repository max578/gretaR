# Numerical Study Plan: Benchmarking R Bayesian and GLMM Packages

**Version:** 1.0  
**Date:** April 10, 2026  
**Author:** Max Moldovan  
**Purpose:** Design and execute a comprehensive, reproducible numerical study comparing the original **greta** package, the proposed new **torch-based package** (working title: `torchbayes` or `greta2`), and leading competitor packages for Bayesian GLMMs in R.

## 1. Objectives

- Compare **computational performance** (speed, memory usage, scalability) across model complexity and sample sizes.
- Assess **versatility** (ease of model specification, support for discrete parameters, extensibility).
- Identify strengths/weaknesses of each package under realistic scenarios ranging from simple/quick analyses to super-complex/large-scale models.
- Provide quantitative evidence to guide development priorities for the new torch-based package.
- Outline everything in publication ready paper.

**Packages to Compare** (all run with default/recommended settings for fairness):
- Original **greta**, installed on the machine + **gretaR**, https://github.com/AAGI-AUS/bayesreml, **asreml**, **mgcv**, **lme4**, **glmmTMB**, **sommer**, **mgcv**, **brms**, **NIMBLE**, **rjags**, **runjags**, **INLA**, **runjags**, and **BGLR**.
- Make sure that all the packages are installed and functional, drop any if not functional, with explanation.
- Try not to push a model to execution if it is unlikely to handle a task.


## 2. Models from Reference File

**Primary source:** Use the complete range of GLMM formulations specified in  
`@GLMM_Model_Formulations_Reference_v5_corrected.xlsx`.

This file already covers a strong spectrum (Gaussian, Poisson, binomial, logit/probit links, random intercepts/slopes, crossed/nested effects, etc.). All models will be implemented with **identical priors and structure** across packages where possible to ensure comparability.

## 3. Suggested Additional Models (Not Yet Considered)

To make the study more comprehensive and forward-looking, include the following models (add as new tabs/sheets in the Excel file if desired):

1. **Zero-inflated / Hurdle GLMMs** (Poisson or negative-binomial with zero-inflation)
2. **Spatio-temporal GLMM** (e.g., Gaussian process random field + AR(1) temporal effect)
3. **Multivariate GLMM** (joint modeling of 2–3 responses with correlated random effects)
4. **Nonlinear mixed-effects model** (e.g., Michaelis-Menten or logistic growth with random parameters)
5. **Survival / frailty model** (Weibull and Cox and Fine-Gray with random effects)
6. **High-dimensional random effects** (e.g., 1,000–5,000 grouping levels or factor loadings)
7. **Models with discrete parameters** (via auxiliary variables or reparameterization – to test new package versatility)
8. Any other feasible options.

These additions stress-test scalability, GPU utilization, and advanced inference features.

## 4. Data Generation

**Base script:** Modify `@r_code_06b_asreml_to_greta_Claude_v0_data_only.R` as necessary, using it as an example.

**Required modifications:**
- Parameterize sample size (`n_obs`), number of grouping levels, and effect sizes via function arguments.
- Ensure reproducible seeds (`set.seed()`) and balanced/unbalanced designs.
- Output clean `data.frame` objects (or arrow/Parquet for very large datasets) that can be loaded by all packages.
- Include a “massive” data generator that produces up to 500,000 to up to 100,000,000 rows (as feasible) while remaining executable on a modern workstation/GPU server (avoid out-of-memory crashes).

**Output:** A single flexible data-generation function (or script) that can produce any scenario on demand.

## 5. Benchmark Scenarios

Create **representative scenarios** covering the full spectrum (as a reference example):

| Scenario | Model Complexity | Sample Size (n_obs) | Grouping Levels | Expected Runtime (target) | Purpose |
|----------|------------------|---------------------|-----------------|---------------------------|---------|
| S1 Small-Simple | Basic LMM / Poisson GLMM | 500–2,000 | 20–50 | < 5 min | Quick sanity check |
| S2 Medium | Random slopes + crossed effects | 10,000–50,000 | 200–500 | 10–30 min | Typical applied use |
| S3 Large-Complex | Full crossed + multiple random effects | 100,000–250,000 | 1,000–2,000 | 30–90 min | Scalability test |
| S4 Massive | Spatio-temporal or high-dim + zero-inflated | 500,000–1,000,000 | 5,000+ | 2–8 hours | Extreme scaling (GPU-focused) |

- Run **4 chains**, **2,000 iterations** (1,000 warmup + 1,000 sampling) per model/scenario (adjust for fairness).
- Repeat each scenario **3 times** and report mean ± SD.

## 6. Metrics to Collect

- **Accuracy / Fidelity**: R-hat, ESS (bulk + tail), posterior mean/quantile agreement with reference (brms or NIMBLE as gold standard).
- **Performance**:
  - Wall-clock time (total + per iteration)
  - Peak RAM usage
  - GPU utilization / memory (where applicable)
- **Usability**:
  - Lines of code required
  - Installation / setup time
  - Convergence warnings / failures
- **Other**: Effective sample size per second, memory efficiency (samples per GB).

## 7. Approximate Computational Effort & Resources

| Scenario | Hardware Recommendation | Est. Time per Model (single run) | Est. RAM | Total Study Effort (all models + 3 reps) |
|----------|--------------------------------|----------------------------------|----------|-------------------------------------------|
| S1 Small-Simple | Laptop (8-core CPU, 16 GB RAM) | 2–5 min | 2–4 GB | ~1–2 hours |
| S2 Medium | Workstation (16-core, 32 GB) | 15–40 min | 8–16 GB | ~4–8 hours |
| S3 Large-Complex | Workstation + GPU (RTX 3060/4070 or equivalent) | 45–120 min | 16–32 GB | ~12–24 hours |
| S4 Massive | High-end server (32+ cores, 64+ GB, A100/4090 GPU) | 2–8 hours | 32–128 GB | ~2–4 days (parallelized) |

**Total estimated study time:** 3–7 days on a single high-end machine (or faster with cloud parallelization on multiple instances).

**Recommended test environment:**
- R 4.4+ on Ubuntu 24.04 / macOS 15 / Windows 11
- torch + CUDA (if available) for new package
- CmdStan 2.36+ for brms
- 4–8 parallel jobs using `future` or `parallel` package

## 8. Execution Phases

1. **Preparation (1 day)** – Finalize Excel models + update data-generation script.
2. **Implementation (2–3 days)** – Write wrapper functions for each package to run identical models.
3. **Execution (3–5 days)** – Run benchmarks (start with small scenarios, scale up).
4. **Analysis (1–2 days)** – Compile results into tables, plots (ggplot2 + patchwork), and summary report.
5. **Documentation** – Produce reproducible Quarto/R Markdown report + raw results (RDS/CSV).

## 9. Deliverables

- `numerical_study_results/` folder with raw data, plots, and summary tables.
- Final report: `benchmark_results_report.qmd` (or .html) with executive summary, detailed tables, and recommendations for `torchbayes` development.
- Reproducible GitHub repository structure for the entire study.
- Outline everything in publication ready paper.

---

**Next Steps Recommendation**  
1. Confirm the additional models you want to include from Section 3.  
2. Share the updated `@r_code_06b_asreml_to_greta_Claude_v0_data_only.R` (or let me help modify it).  
3. Approve this plan so we can start implementation.

This plan ensures the study is **rigorous, reproducible, and directly actionable** for guiding the development of the new torch-based package while maintaining full compatibility with greta’s original functionality.

Ready for your review and feedback!