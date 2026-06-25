# gretaR 0.5.1

## Bug fixes

* The multivariate-normal `log_prob` called two `torch` symbols that the R
  `torch` package does not export -- `torch_linalg_cholesky()` and
  `torch_linalg_solve_triangular()` -- so any model using a
  `multivariate_normal()` variable (including a Gaussian-process prior built on
  one) errored the moment its density was evaluated. The calls now use the real
  exported symbols `linalg_cholesky()` and `linalg_solve_triangular()`. The
  log-density is verified against an independent base-R (`chol` + `backsolve`)
  reference to float precision, and a regression test
  (`test-distributions-mvn.R`) now pins the multivariate-normal density so the
  Cholesky path cannot silently regress again. The path had no test before,
  which is how the dead symbols survived.

* Every distribution constructor now validates its parameters at the call
  boundary. A `NULL`, a missing required argument, a non-numeric value, or a
  numeric carrying `NA` is rejected with a single caller-facing message
  (`` `prob` must be numeric or a <gretaR_array>. ``), mirroring the existing
  `as_data()` idiom. Previously nine constructors -- `normal()`, `student_t()`,
  `bernoulli()`, `cauchy()`, `exponential()`, `half_cauchy()`, `half_normal()`,
  `lognormal()`, `poisson_dist()` -- accepted a `NULL` parameter silently and
  failed later with an unrelated tensor error, while four -- `beta_dist()`,
  `binomial_dist()`, `gamma_dist()`, `negative_binomial()` -- leaked R's
  internal "argument is missing" message. A numeric or a `gretaR_array` graph
  node (the hierarchical-prior case, e.g. `bernoulli(beta_dist(2, 2))`) still
  passes untouched. Found by the package's two-sided conformance sweep;
  `test-distributions-validation.R` pins the behaviour across all eighteen
  distribution constructors.

* Three model- and formula-consuming entry points -- `joint_density()`,
  `compile_to_stan()`, and `remove_re_bars()` -- now reject a non-model or
  non-formula input (including `NULL`) at the call boundary instead of
  accepting it silently. `joint_density()` had returned a closure that failed
  only when later called, `compile_to_stan()` emitted an empty Stan program, and
  `remove_re_bars()` passed `NULL` straight through `reformulas::nobars()`. The
  two-sided conformance sweep, now covering 0.93 of the exported surface,
  surfaced all three; `test-validation-guards.R` pins them.

# gretaR 0.5.0

## Sampling

* **Optional dense mass-matrix metric** for the single-chain HMC/NUTS samplers,
  via `mcmc(..., metric = "dense")`. The default stays `"diag"` (a diagonal
  metric, the inverse posterior variance) -- robust and unchanged. A dense
  metric estimates the full inverse posterior covariance during warmup and
  captures linear parameter correlations a diagonal metric is blind to: on a
  regression with strongly correlated predictors it roughly tripled effective
  sample size per second in our tests. It is opt-in by design -- it adds
  \eqn{O(P^2)} cost and does **not** help (and can hurt) funnel-shaped
  posteriors such as hierarchical latent blocks, where the mixing bottleneck is
  non-Gaussian curvature rather than linear correlation. Falls back to a
  diagonal metric, with a message, when the dimension is too large or warmup
  draws too few. The momentum, velocity, and kinetic-energy operations are now
  routed through a small metric abstraction shared by HMC and NUTS, so the
  diagonal path is behaviourally identical to before (sampler oracle unchanged).

* **ChEES-HMC -- a batchable adaptive-trajectory sampler**, via
  `mcmc(..., batched = TRUE, sampler = "hmc", trajectory = "chees")`. Where the
  No-U-Turn Sampler adapts trajectory length per chain with a recursion that
  cannot be batched, ChEES-HMC (Hoffman, Radul & Sountsov 2021) adapts a single
  trajectory length across the whole chain ensemble, so all chains advance
  together as batched tensor operations. It is opt-in and most useful in a
  specific regime: it needs a reasonably large ensemble (at least about eight
  chains -- below that its criterion is noisy and it can mix worse than NUTS) and
  a well-conditioned posterior. On hierarchical models it pairs with a
  near-centred `random_effect()`: in our tests ChEES on the non-centred funnel
  matched NUTS, but ChEES on a near-centred parameterisation gave several times
  the effective sample size per second of the NUTS default, the centring and the
  adaptive trajectory acting together. Single-chain NUTS remains the robust
  default. The default `trajectory = "fixed"` keeps the integration-time HMC of
  the batched path unchanged.

## Hierarchical models

* **New `random_effect()`** -- a grouped random-effect block with an explicit
  centring weight that interpolates between the non-centred
  (\eqn{u_j = \mathrm{mean} + \mathrm{sd}\,\xi_j}, \eqn{\xi_j \sim N(0, 1)}) and
  centred (\eqn{u_j \sim N(\mathrm{mean}, \mathrm{sd}^2)}) parameterisations. The
  weight is a single number or a per-group vector in \eqn{[0, 1]} and changes
  only the sampling geometry: the implied prior on the group effects,
  \eqn{u_j \sim N(\mathrm{mean}, \mathrm{sd}^2)}, is identical for every weight,
  so it is a pure mixing control. This is the parameterisation lever the dense
  metric cannot supply: where a dense metric corrects linear correlation, the
  centring weight corrects the non-Gaussian funnel coupling between a group
  scale and its latent effects. In our tests on an informative random-intercept
  design (about 50 observations per group), a near-centred weight raised
  effective sample size per second on the group-scale parameter by roughly a
  factor of five over the non-centred default; on a sparse design (about two
  observations per group) the non-centred end remained best -- the crossover
  that makes a per-group, data-adaptive weight the principled choice. Supplying
  the weight directly is the construction-time primitive; a warmup-adaptive
  estimator that sets each weight from the data is planned.

# gretaR 0.4.0

The programmatic model-construction API that lets an embedding package (e.g.
flexyBayes) drive gretaR from its own intermediate representation.

## Programmatic model construction

* **New `model_from_arrays()`** -- an explicit, re-entrant counterpart to
  `model()`. Targets and their names are passed as ordinary arguments rather
  than recovered by deparsing the call, so models can be built programmatically
  (under `do.call()`, from a code generator, or by a host package driving gretaR
  from its own representation) and the parameter names are caller-supplied --
  the hook an embedding package needs to map gretaR's names onto its own scheme.
  The likelihood is scoped to the data nodes you name, so independent models can
  be compiled back to back in one session without `reset_gretaR_env()` between
  them. `model()` and `model_from_arrays()` share a single compiler core, so the
  two paths produce identical model objects; `model()` is unchanged.
* **Per-element parameter names.** `model_from_arrays(names = ...)` now accepts a
  list whose entries may be per-element name vectors, so a length-`p`
  coefficient vector can be labelled `c("(Intercept)", "x1", ...)` and those
  labels flow straight through to the posterior draws -- a complete
  caller-supplied canonical-name contract, no relabel pass required.

# gretaR 0.3.0

Sampler correctness, a robustness fix, and native multi-chain batching. Both
samplers are now verified against closed-form analytic posteriors.

## Sampler correctness

* **NUTS now targets the posterior.** The previous tree selection counted every
  non-divergent leaf and chose uniformly along the trajectory, ignoring the
  density -- so it did not sample the target distribution. Replaced with
  multinomial energy weighting (Betancourt 2017); the diagonal metric is now the
  inverse posterior variance and the U-turn criterion uses the velocity
  \eqn{M^{-1}p}. Recovers a conjugate-Normal posterior with \eqn{\hat R \to 1.00}
  (was 1.54).
* **HMC now mixes.** Fixed-length HMC resonated with the target's periodic flow
  and dual averaging drove the step size above the leapfrog stability limit, so
  it failed to converge even on trivial models (\eqn{\hat R \approx 1.5},
  ESS ~ 10-50). HMC now adapts on the trajectory-averaged acceptance and draws an
  integration time \eqn{T \sim U(0, 2\pi]} per iteration (`n_leapfrog` becomes a
  safety cap). Robust across seeds and models; HMC now matches or beats NUTS.

## New feature: native multi-chain batching

* **`mcmc(..., sampler = "hmc", batched = TRUE)`** advances all chains together as
  one set of batched `torch` tensor operations instead of chain-by-chain.
  Wall-clock is then roughly flat in the number of chains, so many-chain runs are
  much faster -- about 2x at 8 chains and 4x at 16 on CPU -- while remaining
  statistically equivalent to the single-chain sampler. A `device =` argument
  (`"cpu"`, `"mps"`, `"cuda"`) makes the path device-generic; CPU is the
  recommended default (gretaR's log-density is many small ops, so GPU
  kernel-launch overhead currently dominates). Batched NUTS is not yet supported;
  single-chain NUTS remains the robust default.

## Other correctness fixes

* `summary()` on a MAP `gretaR_glm` fit no longer errors (read the correct
  fields).
* The Stan code emitter (`compile_to_stan()`) now translates comparison and
  modulo operators and aborts on any untranslatable operation, instead of
  silently emitting wrong code.
* `[.gretaR_array` rejects column indexing instead of silently row-indexing.
* `uniform()` returns `-Inf` outside its support and counts observations
  correctly.
* Compiled-gradient NaN handling now warns (parity with the standard path), and
  JIT-trace verification uses a dtype-aware relative tolerance checked at several
  points with no silent fallback.

## Documentation

* The GPU-acceleration claim is scoped honestly: gretaR is a native-R `torch`
  (libtorch) backend; the measured performance win is CPU multi-chain batching.
  Constrained-transform support for `dirichlet()` / `lkj_correlation()` /
  `wishart()` as samplable latents remains scheduled for a later release.

# gretaR 0.2.1

Correctness patch addressing findings from the 2026-05-22 audit. v0.2.0 was
tagged but never submitted to CRAN; v0.2.1 supersedes it as the first
externally-released version.

## Correctness fixes

* **`model()` no longer leaks unrelated variables across calls in the same R
  session.** Variable discovery now walks parent links from the requested
  targets and likelihood data nodes (reachability-based), rather than
  enumerating every variable ever created in the global DAG. Two independent
  models built in one session now compile to two independent posteriors.
* **Discrete latent variables (`bernoulli()`, `binomial_dist()`,
  `poisson_dist()`, `negative_binomial()`) are rejected as HMC/NUTS targets**
  with an actionable error pointing to observation or marginalisation.
  Discrete distributions as likelihood RHS (`distribution(y) <- bernoulli(p)`)
  remain fully supported.
* **`dirichlet()`, `lkj_correlation()`, and `wishart()` are gated as
  non-samplable** for this release: their constrained supports (simplex,
  correlation matrix, positive-definite matrix) require bijective transforms
  not yet implemented. The new `samplable` field on `GretaRDistribution`
  makes this declarative; `model()` refuses these as free variables with a
  clear error. Log-prob evaluation still works. Sampler-ready support is
  scheduled for v0.3.

## API & dispatch

* `sum.gretaR_array()` and `mean.gretaR_array()` are now exported and
  registered as S3 methods. `sum(normal(0, 1, dim = c(3, 1)))` works.

## Diagnostics & timing

* `mcmc()$run_time` is now the actual elapsed sampler wall-time (seconds),
  not a placeholder copy of `n_samples`.
* `mcmc()` divergence diagnostics now use the **post-warmup** window
  consistently. The `n_divergences` field in `fit$convergence` and the
  user-visible warning count exclude warmup-phase divergences. Both windows
  remain accessible via `attr(fit$draws, "divergences")` (post-warmup) and
  `attr(fit$draws, "warmup_divergences")`.

## Truncation

* `truncation_log_adjust()` now returns `-Inf` when an observed value falls
  outside `[lower, upper]`, instead of silently returning the unmodified base
  log-prob. Truncation is wired into the `log_prob` methods of `normal()`,
  `student_t()`, `half_normal()`, `half_cauchy()`, `beta_dist()`,
  `gamma_dist()`, `exponential()`, `lognormal()`, and `cauchy()`. Docs now
  describe this as constrained sampling support — the truncated-density
  normalising constant `log(F(upper) - F(lower))` is not yet included, so
  truncated distributions should not be used for Bayes-factor / marginal
  likelihood comparisons. Full normalisation is planned for v0.3.

## Documentation

* `gretaR_glm()` docs honestly describe random-effect support: `(1|group)`
  and `(0 + x | group)` work; correlated intercept + slope `(x | group)`
  raises an informative error pointing to the
  `(1|group) + (0 + x|group)` workaround until LKJ-Cholesky lands in v0.3.

## Tests

* `tests/testthat/test-audit-2026-05-22.R` adds 17 regression tests
  covering every audit finding.

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
