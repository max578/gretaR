## testthat setup — runs once before all test files.
##
## Purpose: deterministic, hermetic tests — no machine-specific state leaks in.

# Deterministic collation so snapshot diffs are stable across locales.
Sys.setlocale("LC_COLLATE", "C")

# Predictable default seeds for any seed-sensitive tests that forget to set one.
set.seed(20260420L)
if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
  torch::torch_manual_seed(20260420L)
}

# Route any cache writes to a session-scoped tempdir rather than the user home.
Sys.setenv(
  R_USER_CACHE_DIR = tempfile("gretaR-cache-"),
  R_USER_DATA_DIR = tempfile("gretaR-data-"),
  R_USER_CONFIG_DIR = tempfile("gretaR-config-")
)

# Strip auth-like env vars — no test should ever hit the network authenticated.
auth_vars <- c(
  "GITHUB_PAT", "GITHUB_TOKEN", "GH_TOKEN",
  "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
)
auth_list <- as.list(rep("", length(auth_vars)))
names(auth_list) <- auth_vars
do.call(Sys.setenv, auth_list)
