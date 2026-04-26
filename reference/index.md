# Package index

## Core DSL

Define data, variables, and models using the gretaR array DSL.

- [`as_data()`](https://max578.github.io/gretaR/reference/as_data.md) :
  Wrap Observed Data as a gretaR Array
- [`variable()`](https://max578.github.io/gretaR/reference/variable.md)
  : Create a Free Variable
- [`distribution()`](https://max578.github.io/gretaR/reference/distribution.md)
  : Get the Distribution of a gretaR Array
- [`` `distribution<-`() ``](https://max578.github.io/gretaR/reference/distribution-set.md)
  : Assign a Distribution (Likelihood) to Observed Data
- [`` `[`( ``*`<gretaR_array>`*`)`](https://max578.github.io/gretaR/reference/sub-.gretaR_array.md)
  : Extract elements from a gretaR_array
- [`model()`](https://max578.github.io/gretaR/reference/model.md) :
  Create a gretaR Model
- [`joint_density()`](https://max578.github.io/gretaR/reference/joint_density.md)
  : Get the Log Joint Density Function

## Distributions

Probability distributions for priors and likelihoods.

- [`normal()`](https://max578.github.io/gretaR/reference/normal.md) :
  Normal Distribution
- [`half_normal()`](https://max578.github.io/gretaR/reference/half_normal.md)
  : Half-Normal Distribution
- [`half_cauchy()`](https://max578.github.io/gretaR/reference/half_cauchy.md)
  : Half-Cauchy Distribution
- [`student_t()`](https://max578.github.io/gretaR/reference/student_t.md)
  : Student-t Distribution
- [`uniform()`](https://max578.github.io/gretaR/reference/uniform.md) :
  Uniform Distribution
- [`bernoulli()`](https://max578.github.io/gretaR/reference/bernoulli.md)
  : Bernoulli Distribution
- [`binomial_dist()`](https://max578.github.io/gretaR/reference/binomial_dist.md)
  : Binomial Distribution
- [`poisson_dist()`](https://max578.github.io/gretaR/reference/poisson_dist.md)
  : Poisson Distribution
- [`gamma_dist()`](https://max578.github.io/gretaR/reference/gamma_dist.md)
  : Gamma Distribution
- [`beta_dist()`](https://max578.github.io/gretaR/reference/beta_dist.md)
  : Beta Distribution
- [`exponential()`](https://max578.github.io/gretaR/reference/exponential.md)
  : Exponential Distribution
- [`multivariate_normal()`](https://max578.github.io/gretaR/reference/multivariate_normal.md)
  : Multivariate Normal Distribution
- [`dirichlet()`](https://max578.github.io/gretaR/reference/dirichlet.md)
  : Dirichlet distribution
- [`negative_binomial()`](https://max578.github.io/gretaR/reference/negative_binomial.md)
  : Negative Binomial distribution
- [`lkj_correlation()`](https://max578.github.io/gretaR/reference/lkj_correlation.md)
  : LKJ Correlation distribution
- [`lognormal()`](https://max578.github.io/gretaR/reference/lognormal.md)
  : Log-Normal Distribution
- [`cauchy()`](https://max578.github.io/gretaR/reference/cauchy.md) :
  Cauchy Distribution
- [`wishart()`](https://max578.github.io/gretaR/reference/wishart.md) :
  Wishart Distribution

## Inference

MCMC samplers, variational inference, and optimisation.

- [`mcmc()`](https://max578.github.io/gretaR/reference/mcmc.md) : Draw
  MCMC Samples from a gretaR Model
- [`hmc()`](https://max578.github.io/gretaR/reference/hmc.md) : Run HMC
  Sampling
- [`nuts()`](https://max578.github.io/gretaR/reference/nuts.md) : Run
  NUTS Sampling
- [`variational()`](https://max578.github.io/gretaR/reference/variational.md)
  : Variational Inference (ADVI)
- [`opt()`](https://max578.github.io/gretaR/reference/opt.md) : Find the
  Maximum A Posteriori (MAP) Estimate
- [`laplace()`](https://max578.github.io/gretaR/reference/laplace.md) :
  Laplace Approximation

## Custom & Mixture

User-defined distributions and mixture models.

- [`custom_distribution()`](https://max578.github.io/gretaR/reference/custom_distribution.md)
  : Custom Distribution
- [`mixture()`](https://max578.github.io/gretaR/reference/mixture.md) :
  Mixture Distribution

## Formula Interface

High-level formula-based model specification.

- [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md)
  : Fit a Bayesian GLM Using Formula Syntax
- [`parse_re_bars()`](https://max578.github.io/gretaR/reference/parse_re_bars.md)
  : Parse random effects bar terms from an lme4-style formula
- [`remove_re_bars()`](https://max578.github.io/gretaR/reference/remove_re_bars.md)
  : Remove random effects bar terms from a formula
- [`process_smooths()`](https://max578.github.io/gretaR/reference/process_smooths.md)
  : Process Smooth Terms from an mgcv-Style Formula

## Stan Backend

Stan code generation and inference.

- [`compile_to_stan()`](https://max578.github.io/gretaR/reference/compile_to_stan.md)
  : Generate Stan Code from a gretaR Model

## Methods and operators

S3 methods and operators on gretaR objects.

- [`` `%*%` ``](https://max578.github.io/gretaR/reference/grapes-times-grapes.md)
  : Matrix multiplication with gretaR_array support
- [`print(`*`<gretaR_fit>`*`)`](https://max578.github.io/gretaR/reference/print.gretaR_fit.md)
  : Print a gretaR Fit Object
- [`summary(`*`<gretaR_fit>`*`)`](https://max578.github.io/gretaR/reference/summary.gretaR_fit.md)
  : Summarise a gretaR Fit Object
- [`coef(`*`<gretaR_fit>`*`)`](https://max578.github.io/gretaR/reference/coef.gretaR_fit.md)
  : Extract Coefficients from a gretaR Fit
- [`plot(`*`<gretaR_fit>`*`)`](https://max578.github.io/gretaR/reference/plot.gretaR_fit.md)
  : Plot Diagnostics for a gretaR Fit

## Utilities

Helper functions for data handling and environment management.

- [`reset_gretaR_env()`](https://max578.github.io/gretaR/reference/reset_gretaR_env.md)
  : Reset the gretaR Global Environment

## Package

- [`gretaR`](https://max578.github.io/gretaR/reference/gretaR-package.md)
  [`gretaR-package`](https://max578.github.io/gretaR/reference/gretaR-package.md)
  : gretaR: Bayesian Statistical Modelling with Torch
