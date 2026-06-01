# cran-comments

## Version note

This is **v0.2.1**, a correctness patch over the never-submitted v0.2.0 tag.
The 2026-05-22 audit (`audit_2026-05-22.md`, local-only) surfaced two P0
inference-correctness issues (graph reachability and discrete-latent
acceptance) and several P1/P2 issues; all are addressed in this release. No
v0.2.0 tarball was sent to CRAN, so this is still effectively a
first-submission.

## Test environments

* local: macOS 26.3.1 (Apple Silicon, aarch64-apple-darwin20), R 4.5.2
* GitHub Actions: ubuntu-latest (devel, release), macos-latest (release),
  windows-latest (release)
* win-builder (devel and release) — pending submission
* R-hub v2 (linux, macos, macos-arm64, windows) — pending submission

## R CMD check results

Local `R CMD check --as-cran` (R 4.5.2, macOS arm64):

  0 errors | 0 warnings | 1 substantive NOTE (CRAN incoming feasibility)

Two further NOTEs on the local run (`future file timestamps`, `HTML Tidy`) are
environmental and do not appear on CRAN-side machines or on win-builder.

CRAN-side incoming checks may add the following NOTEs (per-NOTE
justification, per the project policy of itemising each one):

* **NOTE: New submission.**
  Category: intrinsic-to-first-submission. This is the first CRAN
  release of `gretaR`. Expected and unavoidable for a v0.2.0 first
  submission. No action.

* **NOTE: Possibly mis-spelled words in DESCRIPTION** (if raised).
  Category: source-fix-attempted. `inst/WORDLIST` covers the legitimate
  domain vocabulary (Bayesian, HMC, NUTS, ADVI, torch, posterior,
  cmdstanr, mgcv, greta, etc.) used in `DESCRIPTION`, `NEWS.md`,
  vignettes, and Rd files. Possessives `greta's` / `gretaR's` may
  remain — these are correct English usage of the package names. No
  action.

* **NOTE: Found the following (possibly) invalid URLs:
  `https://max578.github.io/gretaR/` Status: 404.**
  Category: source-fix-attempted (deferred). The pkgdown site is built
  and deployed by the `pkgdown.yaml` GitHub Actions workflow on the
  first push to `main` after v0.2.0 release; the URL becomes live once
  that deploy completes. Submission to CRAN will be timed for *after*
  the deploy lands so this NOTE clears. No source action.

## Downstream dependencies

No reverse dependencies on CRAN at this time (first submission).

## Package-specific notes

* **`torch` is in `Imports`.** `torch::install_torch()` downloads
  LibTorch on first use; this is the established pattern for the
  `torch` R package. Examples and vignettes guard with
  `requireNamespace("torch", quietly = TRUE)` and
  `torch::torch_is_installed()` where needed; long-running MCMC tests
  are wrapped with `skip_on_cran()`.

* **`cmdstanr` in `Suggests` is sourced via
  `Additional_repositories: https://stan-dev.r-universe.dev`.** This
  is the standard pattern — CRAN does not host `cmdstanr`. All
  vignette and example uses of `cmdstanr` are guarded with
  `requireNamespace("cmdstanr", quietly = TRUE)`.

* **Test surface.** 258 PASS / 0 FAIL / 0 WARN / 22 SKIP under
  `R CMD check --as-cran`. The 22 skips are slow MCMC integration tests
  guarded by `skip_on_cran()`; they run in full CI on every push.
