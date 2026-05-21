# gretaR 0.2.0

## User-facing API

* Added `lifecycle::badge("experimental")` markers to the user-facing inference
  API (`mcmc()`, `nuts()`, `hmc()`, `variational()`, `opt()`, `laplace()`,
  `gretaR_glm()`) so users see the stability tier in `?fun`. `lifecycle` added
  to `Imports`.

## Documentation

* Long-form articles `complete-guide` and `migrating-from-greta` moved to
  `vignettes/articles/` (pkgdown-only). They remain visible on the website but
  no longer ship with the installed package. The three bundled vignettes
  (`getting-started`, `glm-models`, `hierarchical-models`) cover the core API.
* README install section now defaults to r-universe binaries; source-build
  via `remotes::install_github()` kept as the secondary path.
* DESCRIPTION `URL:` now lists both the GitHub repo and the pkgdown site.

## Continuous integration

* CI matrix extended with `ubuntu-latest × R oldrel-1` (now 3 OS × 3 R versions).

## Packaging & infrastructure

* Language set to `en-AU`; parallel testthat enabled.
* Maintainer role extended with `cph`.
* Replaced `match.arg()` with `rlang::arg_match()` across all R sources for
  clearer error messages; `rlang` added to `Imports`.
* Added `inst/CITATION` and `CITATION.cff` for machine-readable citation metadata.
* Added `codemeta.json` for software citation and discovery.
* Added `inst/WORDLIST` to silence spellcheck false positives.
* Added `.lintr` (snake_case, tidyverse-style, 100-char lines) and `air.toml`
  for reproducible formatting.
* Added `cran-comments.md` and `codecov.yml` templates.
* `.gitignore` and `.Rbuildignore` tightened; `.DS_Store` untracked.

## Continuous integration

* Added `pkgdown.yaml`, `test-coverage.yaml`, `lint.yaml`, `revdep.yaml`, and
  `pr-commands.yaml` workflows.

## Tests

* Added `tests/testthat/setup.R` that pins locale to `C`, seeds torch
  deterministically, redirects user cache dirs to tempfiles, and strips
  auth-style env vars.

## Documentation

* README converted to `README.Rmd` source with badges (R-CMD-check, pkgdown,
  test-coverage, lifecycle, codecov). Install instruction switched from
  `remotes::install_github()` to `pak::pak()`.

# gretaR 0.1.0

## Core Features

* R-native DSL: `as_data()`, `variable()`, `distribution()`, `model()`.
* Operator overloading: `+`, `-`, `*`, `/`, `^`, `%*%`, `[`, `log()`, `exp()`,
  `sqrt()`, `sin()`, `cos()`, and more on `gretaR_array` objects.
* Lazy DAG construction with automatic compilation via `model()`.

## Distributions (18 + custom + mixture)

* Continuous: `normal()`, `half_normal()`, `half_cauchy()`, `student_t()`,
  `cauchy()`, `exponential()`, `gamma_dist()`, `beta_dist()`, `lognormal()`,
  `uniform()`, `multivariate_normal()`.
* Discrete: `bernoulli()`, `binomial_dist()`, `poisson_dist()`,
  `negative_binomial()`.
* Multivariate: `dirichlet()`, `lkj_correlation()`, `wishart()`.
* Custom: `custom_distribution()` — user-defined torch log-probability.
* Mixture: `mixture()` — log-sum-exp marginalisation over discrete components.
* Truncation: `truncation = c(lower, upper)` on all continuous distributions
  (greta-compatible syntax).

## Inference Engines

* NUTS and static HMC with windowed warmup adaptation.
* Variational inference (`variational()`) — mean-field and full-rank ADVI.
* MAP estimation (`opt()`) via Adam optimiser.
* Laplace approximation (`laplace()`) with Hessian-based posterior covariance.
* Stan backend: `mcmc(m, backend = "stan")` for 30–150x faster inference
  on standard models via cmdstanr.

## Formula Interface

* `gretaR_glm()` — high-level GLM specification with gaussian, binomial,
  and poisson families.
* lme4-style random effects: `(1|group)`, `(x|group)`, `(0+x|group)`.
* mgcv-style smooth terms: `s()`, `te()`, `ti()`, `t2()` via smooth2random
  decomposition (all 21 mgcv basis types supported).
* Auto-detection of formula style (base, lme4, mgcv).

## Hierarchical Models

* Array indexing `alpha[group_id]` for group-level parameters.
* Non-centred parameterisation by default in formula interface.

## Additional Features

* Sparse matrix support via Matrix package (`as_data()` accepts `dgCMatrix`).
* Compiled log-probability function for 3–4x torch backend speedup.
* Unified `gretaR_fit` output with `print()`, `summary()`, `coef()`, `plot()`.
* `seed` parameter on all inference functions for reproducibility.
* `compile_to_stan()` for inspecting generated Stan code.

## Documentation

* 5 vignettes: complete guide, getting started, GLMs, hierarchical models,
  migration from greta.
* JOSS paper draft.
* Technical documentation (.pdf, .tex, .md).
