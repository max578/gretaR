# Regression tests for the 2026-06-01 R99 remediation queue (Batch 1).
#
# GB3 — summary.gretaR_glm_fit MAP path read the wrong fields.
# GS5 — Stan emitter silently dropped comparison / modulo operators.
# GS6 — `[.gretaR_array` ignored the column index `j`.
# GS7 — fast_grad zeroed NaN gradients without warning (slow path warned).

# --- GB3: summary() on a MAP gretaR_glm fit -------------------------------

test_that("summary() works on a MAP gretaR_glm fit (GB3)", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(123)
  dat <- data.frame(
    y = 2 + 3 * rnorm(50) + rnorm(50, 0, 0.5),
    x = rnorm(50)
  )
  fit <- gretaR_glm(y ~ x, data = dat, family = "gaussian",
                    sampler = "map", verbose = FALSE)

  # Before the fix this errored: object$sampler is NULL (constructor stores
  # method), so `NULL == "map"` -> logical(0) -> if() error.
  expect_no_error(s <- summary(fit))
  s <- summary(fit)
  expect_s3_class(s, "summary.gretaR_glm_fit")
  expect_false(is.null(s$map_estimates))
  expect_equal(s$map_estimates, fit$par)
})

# --- GS5: Stan emitter translates comparison / modulo operators -----------

test_that("Stan emitter translates comparison and modulo ops (GS5)", {
  # node_to_stan_expr is pure string translation; build a minimal fake DAG.
  mk_data <- function(id) list(node_type = "data", id = id)
  dag <- list(node_1 = mk_data("node_1"), node_2 = mk_data("node_2"))
  stan_names <- list(node_1 = "x", node_2 = "y")
  emit <- function(ot) {
    op <- list(
      node_type = "operation", id = "node_3",
      op_type = ot, parents = c("node_1", "node_2")
    )
    node_to_stan_expr(op, dag, list(), stan_names)
  }

  expect_equal(emit("binary_>"), "(x > y)")
  expect_equal(emit("binary_<"), "(x < y)")
  expect_equal(emit("binary_>="), "(x >= y)")
  expect_equal(emit("binary_<="), "(x <= y)")
  expect_equal(emit("binary_=="), "(x == y)")
  expect_equal(emit("binary_!="), "(x != y)")
  expect_equal(emit("binary_%%"), "fmod(x, y)")
  # Arithmetic still correct (regression guard on the existing branch).
  expect_equal(emit("binary_+"), "(x + y)")
})

test_that("Stan emitter aborts on an untranslatable op (GS5)", {
  mk_data <- function(id) list(node_type = "data", id = id)
  dag <- list(node_1 = mk_data("node_1"), node_2 = mk_data("node_2"))
  stan_names <- list(node_1 = "x", node_2 = "y")

  # Binary: previously fell through to a silent "(x + y)".
  bad_bin <- list(
    node_type = "operation", id = "node_3",
    op_type = "binary_nonsense", parents = c("node_1", "node_2")
  )
  expect_error(
    node_to_stan_expr(bad_bin, dag, list(), stan_names),
    "Cannot translate binary operation"
  )

  # Unary: previously passed through the operand silently.
  bad_un <- list(
    node_type = "operation", id = "node_3",
    op_type = "math_nonsense", parents = "node_1"
  )
  expect_error(
    node_to_stan_expr(bad_un, dag, list(), stan_names),
    "Cannot translate unary operation"
  )
})

# --- GS6: `[.gretaR_array` rejects column indexing ------------------------

test_that("`[.gretaR_array` rejects column indexing instead of mis-indexing (GS6)", {
  skip_if_not_installed("torch")
  reset_gretaR_env()

  x <- as_data(matrix(1:6, nrow = 3, ncol = 2))

  # `beta[i, j]` used to silently row-select and drop j.
  expect_error(x[1L, 2L], "Column indexing")
  # `beta[, j]` used to error obscurely on a missing i.
  expect_error(x[, 2L], "Column indexing")

  # Single-index selection (the hierarchical alpha[group_id] path) still works.
  sel <- x[c(1L, 2L)]
  expect_s3_class(sel, "gretaR_array")
  expect_equal(dim(sel), c(2L, 2L))
})

# --- GS7: fast_grad warns on NaN gradients (parity with the slow path) -----

test_that("fast_grad warns before zeroing NaN gradients (GS7)", {
  skip_if_not_installed("torch")

  # sqrt of a negative input gives a NaN forward value and a NaN gradient.
  compiled_fn <- function(theta_t) torch::torch_sum(torch::torch_sqrt(theta_t))

  # cli_alert_warning() signals a message condition (matching the slow path).
  expect_message(
    g <- fast_grad(compiled_fn, c(-1, -2), torch::torch_float32()),
    "NaN gradient"
  )
  g <- suppressMessages(
    fast_grad(compiled_fn, c(-1, -2), torch::torch_float32())
  )
  expect_false(any(is.nan(g$grad)))
})
