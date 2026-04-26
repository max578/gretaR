# Fit a Bayesian GLM Using Formula Syntax

A high-level interface for specifying and fitting generalised linear
models (including mixed / hierarchical models) using standard R formula
syntax. Internally translates to the gretaR DSL, constructs the model,
and runs MCMC inference.

lme4-style random effects are supported via regex parsing (lme4 is
**not** required). Recognised patterns:

- `(1|group)` — random intercepts by group

- `(x|group)` — correlated random intercepts + slopes

- `(0 + x|group)` — random slopes only (no intercept)

Multiple random effect terms are permitted, e.g.
`y ~ x + (1|site) + (1|year)`.

A non-centred parameterisation is used by default for superior HMC
geometry.

## Usage

``` r
gretaR_glm(
  formula,
  data,
  family = c("gaussian", "binomial", "poisson"),
  prior = NULL,
  sampler = c("nuts", "hmc", "vi", "map"),
  chains = 4L,
  iter = 2000L,
  warmup = NULL,
  formula_style = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- formula:

  A formula specifying the model (e.g., `y ~ x1 + x2`, or
  `y ~ x + (1|group)` for mixed models).

- data:

  A data frame containing the variables referenced in the formula.

- family:

  Distribution family: `"gaussian"` (default), `"binomial"`, or
  `"poisson"`.

- prior:

  A named list of gretaR distribution objects for parameter priors.
  Recognised names: `"beta"` (regression coefficients), `"intercept"`,
  `"sigma"` (residual SD, gaussian only), `"tau"` (random effect SD).
  Use `NULL` for default priors.

- sampler:

  Sampler: `"nuts"` (default), `"hmc"`, `"vi"`, or `"map"`.

- chains:

  Number of MCMC chains (default 4).

- iter:

  Total iterations per chain (warmup + samples, default 2000).

- warmup:

  Number of warmup iterations (default half of iter).

- formula_style:

  Optional explicit formula style hint: `"base"`, `"lme4"`, `"brms"`,
  `"mgcv"`. If `NULL` (default), auto-detected.

- verbose:

  Logical; print progress (default TRUE).

- ...:

  Additional arguments passed to the sampler.

## Value

A `gretaR_glm_fit` object with components:

- draws:

  Posterior draws (from MCMC or VI).

- model:

  The compiled gretaR_model.

- formula:

  The original formula.

- family:

  The family used.

- data:

  The original data.

- design_matrix:

  The model matrix (fixed effects).

- col_names:

  Named mapping of fixed-effect parameters.

- random_effects:

  List of parsed random effect specifications (NULL for base-style
  formulas).

## Examples

``` r
if (FALSE) { # \dontrun{
# Gaussian linear model
fit <- gretaR_glm(Sepal.Length ~ Sepal.Width + Petal.Length,
                  data = iris, family = "gaussian")
summary(fit$draws)

# Logistic regression
dat <- data.frame(y = rbinom(100, 1, 0.6), x = rnorm(100))
fit <- gretaR_glm(y ~ x, data = dat, family = "binomial")

# Custom priors
fit <- gretaR_glm(y ~ x, data = dat, family = "gaussian",
                  prior = list(beta = normal(0, 1), sigma = half_cauchy(1)))

# Random intercepts model (lme4-style)
sleepstudy <- data.frame(
  Reaction = rnorm(180, 300, 50),
  Days = rep(0:9, each = 18),
  Subject = factor(rep(1:18, times = 10))
)
fit <- gretaR_glm(Reaction ~ Days + (1 | Subject),
                  data = sleepstudy, family = "gaussian")

# Random intercepts + slopes
fit <- gretaR_glm(Reaction ~ Days + (Days | Subject),
                  data = sleepstudy, family = "gaussian")
} # }
```
