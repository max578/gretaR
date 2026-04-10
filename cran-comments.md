## Test environments
* macOS 14.x (Apple Silicon), R 4.5.2
* GitHub Actions: ubuntu-latest (R release, R devel), macos-latest (R release)

## R CMD check results
There were no ERRORs or WARNINGs.

There were 2 NOTEs:

1. New submission — this is the first CRAN release of gretaR.

2. `checking for future file timestamps` — unable to verify current time.
   This is an infrastructure issue, not a package problem.

## Notes

* This package depends on `torch`, which requires a one-time download of
  libtorch binaries via `torch::install_torch()`. This is documented in the
  package README and vignettes.

* The `Matrix` package is in Suggests (not Imports) for optional sparse matrix
  support. It is a recommended package shipped with R and imposes no additional
  installation burden.
