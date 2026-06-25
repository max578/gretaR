# Front-door input validation for the model-consuming and formula entry points.
#
# joint_density() returns a lazy closure, compile_to_stan() iterates over the
# model's blocks, and remove_re_bars() delegates to reformulas::nobars() -- all
# three silently accepted a NULL before they were guarded (the model/formula
# counterpart of the distribution-constructor gap). The NULL-rejection cases
# need no torch backend; only the valid cases build a model.

test_that("joint_density() rejects a non-model input", {
  expect_error(joint_density(NULL), "must be a")
  expect_error(joint_density(list()), "must be a")
})

test_that("compile_to_stan() rejects a non-model input", {
  expect_error(compile_to_stan(NULL), "must be a")
  expect_error(compile_to_stan(42), "must be a")
})

test_that("remove_re_bars() rejects a non-formula input", {
  expect_error(remove_re_bars(NULL), "must be a")
  expect_error(remove_re_bars("y ~ x"), "must be a")
})

test_that("the guarded entry points still accept valid input", {
  skip_if_not_installed("torch")
  y <- as_data(c(1, 2, 1.5, 0.5, 2.5))
  mu <- normal(0, 5)
  distribution(y) <- normal(mu, 1)
  m <- model(mu)
  expect_true(is.function(joint_density(m)))
  expect_true(is.character(compile_to_stan(m)))
  expect_s3_class(remove_re_bars(y ~ x + (1 | g)), "formula")
})
