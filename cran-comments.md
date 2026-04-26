# cran-comments

## Test environments

* local: macOS 26.3.1 (Apple Silicon, aarch64-apple-darwin20), R 4.5.2
* GitHub Actions: ubuntu-latest (devel, release), macos-latest (release),
  windows-latest (release)
* win-builder (devel and release) — pending submission
* R-hub v2 (linux, macos, macos-arm64, windows) — pending submission

## R CMD check results

Local `R CMD check --as-cran` (R 4.5.2, macOS arm64):

  0 errors | 0 warnings | 0 notes

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

* **Test surface.** 210 PASS / 0 FAIL / 1 WARN / 20 SKIP on CRAN. The
  20 skips are slow MCMC integration tests guarded by
  `skip_on_cran()`; they run in full CI on every push.
