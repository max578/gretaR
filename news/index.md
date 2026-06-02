# Changelog

## gretaR (development version)

### Sampling

- **Optional dense mass-matrix metric** for the single-chain HMC/NUTS
  samplers, via `mcmc(..., metric = "dense")`. The default stays
  `"diag"` (a diagonal metric, the inverse posterior variance) – robust
  and unchanged. A dense metric estimates the full inverse posterior
  covariance during warmup and captures linear parameter correlations a
  diagonal metric is blind to: on a regression with strongly correlated
  predictors it roughly tripled effective sample size per second in our
  tests. It is opt-in by design – it adds cost and does **not** help
  (and can hurt) funnel-shaped posteriors such as hierarchical latent
  blocks, where the mixing bottleneck is non-Gaussian curvature rather
  than linear correlation. Falls back to a diagonal metric, with a
  message, when the dimension is too large or warmup draws too few. The
  momentum, velocity, and kinetic-energy operations are now routed
  through a small metric abstraction shared by HMC and NUTS, so the
  diagonal path is behaviourally identical to before (sampler oracle
  unchanged).

## gretaR 0.4.0

The programmatic model-construction API that lets an embedding package
(e.g. flexyBayes) drive gretaR from its own intermediate representation.

### Programmatic model construction

- **New
  [`model_from_arrays()`](https://max578.github.io/gretaR/reference/model_from_arrays.md)**
  – an explicit, re-entrant counterpart to
  [`model()`](https://max578.github.io/gretaR/reference/model.md).
  Targets and their names are passed as ordinary arguments rather than
  recovered by deparsing the call, so models can be built
  programmatically (under
  [`do.call()`](https://rdrr.io/r/base/do.call.html), from a code
  generator, or by a host package driving gretaR from its own
  representation) and the parameter names are caller-supplied – the hook
  an embedding package needs to map gretaR’s names onto its own scheme.
  The likelihood is scoped to the data nodes you name, so independent
  models can be compiled back to back in one session without
  [`reset_gretaR_env()`](https://max578.github.io/gretaR/reference/reset_gretaR_env.md)
  between them.
  [`model()`](https://max578.github.io/gretaR/reference/model.md) and
  [`model_from_arrays()`](https://max578.github.io/gretaR/reference/model_from_arrays.md)
  share a single compiler core, so the two paths produce identical model
  objects;
  [`model()`](https://max578.github.io/gretaR/reference/model.md) is
  unchanged.
- **Per-element parameter names.** `model_from_arrays(names = ...)` now
  accepts a list whose entries may be per-element name vectors, so a
  length-`p` coefficient vector can be labelled
  `c("(Intercept)", "x1", ...)` and those labels flow straight through
  to the posterior draws – a complete caller-supplied canonical-name
  contract, no relabel pass required.

## gretaR 0.3.0

Sampler correctness, a robustness fix, and native multi-chain batching.
Both samplers are now verified against closed-form analytic posteriors.

### Sampler correctness

- **NUTS now targets the posterior.** The previous tree selection
  counted every non-divergent leaf and chose uniformly along the
  trajectory, ignoring the density – so it did not sample the target
  distribution. Replaced with multinomial energy weighting (Betancourt
  2017); the diagonal metric is now the inverse posterior variance and
  the U-turn criterion uses the velocity . Recovers a conjugate-Normal
  posterior with (was 1.54).
- **HMC now mixes.** Fixed-length HMC resonated with the target’s
  periodic flow and dual averaging drove the step size above the
  leapfrog stability limit, so it failed to converge even on trivial
  models (, ESS ~ 10-50). HMC now adapts on the trajectory-averaged
  acceptance and draws an integration time per iteration (`n_leapfrog`
  becomes a safety cap). Robust across seeds and models; HMC now matches
  or beats NUTS.

### New feature: native multi-chain batching

- **`mcmc(..., sampler = "hmc", batched = TRUE)`** advances all chains
  together as one set of batched `torch` tensor operations instead of
  chain-by-chain. Wall-clock is then roughly flat in the number of
  chains, so many-chain runs are much faster – about 2x at 8 chains and
  4x at 16 on CPU – while remaining statistically equivalent to the
  single-chain sampler. A `device =` argument (`"cpu"`, `"mps"`,
  `"cuda"`) makes the path device-generic; CPU is the recommended
  default (gretaR’s log-density is many small ops, so GPU kernel-launch
  overhead currently dominates). Batched NUTS is not yet supported;
  single-chain NUTS remains the robust default.

### Other correctness fixes

- [`summary()`](https://rdrr.io/r/base/summary.html) on a MAP
  `gretaR_glm` fit no longer errors (read the correct fields).
- The Stan code emitter
  ([`compile_to_stan()`](https://max578.github.io/gretaR/reference/compile_to_stan.md))
  now translates comparison and modulo operators and aborts on any
  untranslatable operation, instead of silently emitting wrong code.
- `[.gretaR_array` rejects column indexing instead of silently
  row-indexing.
- [`uniform()`](https://max578.github.io/gretaR/reference/uniform.md)
  returns `-Inf` outside its support and counts observations correctly.
- Compiled-gradient NaN handling now warns (parity with the standard
  path), and JIT-trace verification uses a dtype-aware relative
  tolerance checked at several points with no silent fallback.

### Documentation

- The GPU-acceleration claim is scoped honestly: gretaR is a native-R
  `torch` (libtorch) backend; the measured performance win is CPU
  multi-chain batching. Constrained-transform support for
  [`dirichlet()`](https://max578.github.io/gretaR/reference/dirichlet.md)
  /
  [`lkj_correlation()`](https://max578.github.io/gretaR/reference/lkj_correlation.md)
  / [`wishart()`](https://max578.github.io/gretaR/reference/wishart.md)
  as samplable latents remains scheduled for a later release.

## gretaR 0.2.1

Correctness patch addressing findings from the 2026-05-22 audit. v0.2.0
was tagged but never submitted to CRAN; v0.2.1 supersedes it as the
first externally-released version.

### Correctness fixes

- **[`model()`](https://max578.github.io/gretaR/reference/model.md) no
  longer leaks unrelated variables across calls in the same R session.**
  Variable discovery now walks parent links from the requested targets
  and likelihood data nodes (reachability-based), rather than
  enumerating every variable ever created in the global DAG. Two
  independent models built in one session now compile to two independent
  posteriors.
- **Discrete latent variables
  ([`bernoulli()`](https://max578.github.io/gretaR/reference/bernoulli.md),
  [`binomial_dist()`](https://max578.github.io/gretaR/reference/binomial_dist.md),
  [`poisson_dist()`](https://max578.github.io/gretaR/reference/poisson_dist.md),
  [`negative_binomial()`](https://max578.github.io/gretaR/reference/negative_binomial.md))
  are rejected as HMC/NUTS targets** with an actionable error pointing
  to observation or marginalisation. Discrete distributions as
  likelihood RHS (`distribution(y) <- bernoulli(p)`) remain fully
  supported.
- **[`dirichlet()`](https://max578.github.io/gretaR/reference/dirichlet.md),
  [`lkj_correlation()`](https://max578.github.io/gretaR/reference/lkj_correlation.md),
  and
  [`wishart()`](https://max578.github.io/gretaR/reference/wishart.md)
  are gated as non-samplable** for this release: their constrained
  supports (simplex, correlation matrix, positive-definite matrix)
  require bijective transforms not yet implemented. The new `samplable`
  field on `GretaRDistribution` makes this declarative;
  [`model()`](https://max578.github.io/gretaR/reference/model.md)
  refuses these as free variables with a clear error. Log-prob
  evaluation still works. Sampler-ready support is scheduled for v0.3.

### API & dispatch

- [`sum.gretaR_array()`](https://max578.github.io/gretaR/reference/sum.gretaR_array.md)
  and
  [`mean.gretaR_array()`](https://max578.github.io/gretaR/reference/mean.gretaR_array.md)
  are now exported and registered as S3 methods.
  `sum(normal(0, 1, dim = c(3, 1)))` works.

### Diagnostics & timing

- `mcmc()$run_time` is now the actual elapsed sampler wall-time
  (seconds), not a placeholder copy of `n_samples`.
- [`mcmc()`](https://max578.github.io/gretaR/reference/mcmc.md)
  divergence diagnostics now use the **post-warmup** window
  consistently. The `n_divergences` field in `fit$convergence` and the
  user-visible warning count exclude warmup-phase divergences. Both
  windows remain accessible via `attr(fit$draws, "divergences")`
  (post-warmup) and `attr(fit$draws, "warmup_divergences")`.

### Truncation

- `truncation_log_adjust()` now returns `-Inf` when an observed value
  falls outside `[lower, upper]`, instead of silently returning the
  unmodified base log-prob. Truncation is wired into the `log_prob`
  methods of
  [`normal()`](https://max578.github.io/gretaR/reference/normal.md),
  [`student_t()`](https://max578.github.io/gretaR/reference/student_t.md),
  [`half_normal()`](https://max578.github.io/gretaR/reference/half_normal.md),
  [`half_cauchy()`](https://max578.github.io/gretaR/reference/half_cauchy.md),
  [`beta_dist()`](https://max578.github.io/gretaR/reference/beta_dist.md),
  [`gamma_dist()`](https://max578.github.io/gretaR/reference/gamma_dist.md),
  [`exponential()`](https://max578.github.io/gretaR/reference/exponential.md),
  [`lognormal()`](https://max578.github.io/gretaR/reference/lognormal.md),
  and [`cauchy()`](https://max578.github.io/gretaR/reference/cauchy.md).
  Docs now describe this as constrained sampling support — the
  truncated-density normalising constant `log(F(upper) - F(lower))` is
  not yet included, so truncated distributions should not be used for
  Bayes-factor / marginal likelihood comparisons. Full normalisation is
  planned for v0.3.

### Documentation

- [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md)
  docs honestly describe random-effect support: `(1|group)` and
  `(0 + x | group)` work; correlated intercept + slope `(x | group)`
  raises an informative error pointing to the
  `(1|group) + (0 + x|group)` workaround until LKJ-Cholesky lands in
  v0.3.

### Tests

- `tests/testthat/test-audit-2026-05-22.R` adds 17 regression tests
  covering every audit finding.

## gretaR 0.2.0

### User-facing API

- Added `lifecycle::badge("experimental")` markers to the user-facing
  inference API
  ([`mcmc()`](https://max578.github.io/gretaR/reference/mcmc.md),
  [`nuts()`](https://max578.github.io/gretaR/reference/nuts.md),
  [`hmc()`](https://max578.github.io/gretaR/reference/hmc.md),
  [`variational()`](https://max578.github.io/gretaR/reference/variational.md),
  [`opt()`](https://max578.github.io/gretaR/reference/opt.md),
  [`laplace()`](https://max578.github.io/gretaR/reference/laplace.md),
  [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md))
  so users see the stability tier in `?fun`. `lifecycle` added to
  `Imports`.

### Documentation

- Long-form articles `complete-guide` and `migrating-from-greta` moved
  to `vignettes/articles/` (pkgdown-only). They remain visible on the
  website but no longer ship with the installed package. The three
  bundled vignettes (`getting-started`, `glm-models`,
  `hierarchical-models`) cover the core API.
- README install section now defaults to r-universe binaries;
  source-build via `remotes::install_github()` kept as the secondary
  path.
- DESCRIPTION `URL:` now lists both the GitHub repo and the pkgdown
  site.

### Continuous integration

- CI matrix extended with `ubuntu-latest × R oldrel-1` (now 3 OS × 3 R
  versions).

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
