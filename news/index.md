# Changelog

## gretaR 0.2.0

### Packaging & infrastructure

- Language set to `en-AU`; parallel testthat enabled.
- Maintainer role extended with `cph`.
- Replaced [`match.arg()`](https://rdrr.io/r/base/match.arg.html) with
  [`rlang::arg_match()`](https://rlang.r-lib.org/reference/arg_match.html)
  across all R sources for clearer error messages; `rlang` added to
  `Imports`.
- Added `inst/CITATION` and `CITATION.cff` for machine-readable citation
  metadata.
- Added `codemeta.json` for software citation and discovery.
- Added `inst/WORDLIST` to silence spellcheck false positives.
- Added `.lintr` (snake_case, tidyverse-style, 100-char lines) and
  `air.toml` for reproducible formatting.
- Added `cran-comments.md` and `codecov.yml` templates.
- `.gitignore` and `.Rbuildignore` tightened; `.DS_Store` untracked.

### Continuous integration

- Added `pkgdown.yaml`, `test-coverage.yaml`, `lint.yaml`,
  `revdep.yaml`, and `pr-commands.yaml` workflows.

### Tests

- Added `tests/testthat/setup.R` that pins locale to `C`, seeds torch
  deterministically, redirects user cache dirs to tempfiles, and strips
  auth-style env vars.

### Documentation

- README converted to `README.Rmd` source with badges (R-CMD-check,
  pkgdown, test-coverage, lifecycle, codecov). Install instruction
  switched from `remotes::install_github()` to
  [`pak::pak()`](https://pak.r-lib.org/reference/pak.html).

## gretaR 0.1.0

### Core Features

- R-native DSL:
  [`as_data()`](https://max578.github.io/gretaR/reference/as_data.md),
  [`variable()`](https://max578.github.io/gretaR/reference/variable.md),
  [`distribution()`](https://max578.github.io/gretaR/reference/distribution.md),
  [`model()`](https://max578.github.io/gretaR/reference/model.md).
- Operator overloading: `+`, `-`, `*`, `/`, `^`, `%*%`, `[`,
  [`log()`](https://rdrr.io/r/base/Log.html),
  [`exp()`](https://rdrr.io/r/base/Log.html),
  [`sqrt()`](https://rdrr.io/r/base/MathFun.html),
  [`sin()`](https://rdrr.io/r/base/Trig.html),
  [`cos()`](https://rdrr.io/r/base/Trig.html), and more on
  `gretaR_array` objects.
- Lazy DAG construction with automatic compilation via
  [`model()`](https://max578.github.io/gretaR/reference/model.md).

### Distributions (18 + custom + mixture)

- Continuous:
  [`normal()`](https://max578.github.io/gretaR/reference/normal.md),
  [`half_normal()`](https://max578.github.io/gretaR/reference/half_normal.md),
  [`half_cauchy()`](https://max578.github.io/gretaR/reference/half_cauchy.md),
  [`student_t()`](https://max578.github.io/gretaR/reference/student_t.md),
  [`cauchy()`](https://max578.github.io/gretaR/reference/cauchy.md),
  [`exponential()`](https://max578.github.io/gretaR/reference/exponential.md),
  [`gamma_dist()`](https://max578.github.io/gretaR/reference/gamma_dist.md),
  [`beta_dist()`](https://max578.github.io/gretaR/reference/beta_dist.md),
  [`lognormal()`](https://max578.github.io/gretaR/reference/lognormal.md),
  [`uniform()`](https://max578.github.io/gretaR/reference/uniform.md),
  [`multivariate_normal()`](https://max578.github.io/gretaR/reference/multivariate_normal.md).
- Discrete:
  [`bernoulli()`](https://max578.github.io/gretaR/reference/bernoulli.md),
  [`binomial_dist()`](https://max578.github.io/gretaR/reference/binomial_dist.md),
  [`poisson_dist()`](https://max578.github.io/gretaR/reference/poisson_dist.md),
  [`negative_binomial()`](https://max578.github.io/gretaR/reference/negative_binomial.md).
- Multivariate:
  [`dirichlet()`](https://max578.github.io/gretaR/reference/dirichlet.md),
  [`lkj_correlation()`](https://max578.github.io/gretaR/reference/lkj_correlation.md),
  [`wishart()`](https://max578.github.io/gretaR/reference/wishart.md).
- Custom:
  [`custom_distribution()`](https://max578.github.io/gretaR/reference/custom_distribution.md)
  — user-defined torch log-probability.
- Mixture:
  [`mixture()`](https://max578.github.io/gretaR/reference/mixture.md) —
  log-sum-exp marginalisation over discrete components.
- Truncation: `truncation = c(lower, upper)` on all continuous
  distributions (greta-compatible syntax).

### Inference Engines

- NUTS and static HMC with windowed warmup adaptation.
- Variational inference
  ([`variational()`](https://max578.github.io/gretaR/reference/variational.md))
  — mean-field and full-rank ADVI.
- MAP estimation
  ([`opt()`](https://max578.github.io/gretaR/reference/opt.md)) via Adam
  optimiser.
- Laplace approximation
  ([`laplace()`](https://max578.github.io/gretaR/reference/laplace.md))
  with Hessian-based posterior covariance.
- Stan backend: `mcmc(m, backend = "stan")` for 30–150x faster inference
  on standard models via cmdstanr.

### Formula Interface

- [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md)
  — high-level GLM specification with gaussian, binomial, and poisson
  families.
- lme4-style random effects: `(1|group)`, `(x|group)`, `(0+x|group)`.
- mgcv-style smooth terms: [`s()`](https://rdrr.io/pkg/mgcv/man/s.html),
  [`te()`](https://rdrr.io/pkg/mgcv/man/te.html),
  [`ti()`](https://rdrr.io/pkg/mgcv/man/te.html),
  [`t2()`](https://rdrr.io/pkg/mgcv/man/t2.html) via smooth2random
  decomposition (all 21 mgcv basis types supported).
- Auto-detection of formula style (base, lme4, mgcv).

### Hierarchical Models

- Array indexing `alpha[group_id]` for group-level parameters.
- Non-centred parameterisation by default in formula interface.

### Additional Features

- Sparse matrix support via Matrix package
  ([`as_data()`](https://max578.github.io/gretaR/reference/as_data.md)
  accepts `dgCMatrix`).
- Compiled log-probability function for 3–4x torch backend speedup.
- Unified `gretaR_fit` output with
  [`print()`](https://rdrr.io/r/base/print.html),
  [`summary()`](https://rdrr.io/r/base/summary.html),
  [`coef()`](https://rdrr.io/r/stats/coef.html),
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html).
- `seed` parameter on all inference functions for reproducibility.
- [`compile_to_stan()`](https://max578.github.io/gretaR/reference/compile_to_stan.md)
  for inspecting generated Stan code.

### Documentation

- 5 vignettes: complete guide, getting started, GLMs, hierarchical
  models, migration from greta.
- JOSS paper draft.
- Technical documentation (.pdf, .tex, .md).
