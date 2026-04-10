---
title: 'gretaR: Bayesian Statistical Modelling in R with Torch'
tags:
  - R
  - Bayesian inference
  - probabilistic programming
  - Hamiltonian Monte Carlo
  - torch
authors:
  - name: Max Moldovan
    orcid: 0000-0001-9680-8474
    affiliation: 1
affiliations:
  - name: Adelaide University, Australia
    index: 1
date: 10 April 2026
bibliography: paper.bib
---

# Summary

`gretaR` is an R package for Bayesian statistical modelling that combines an
intuitive, R-native domain-specific language (DSL) with GPU-capable inference
via the `torch` backend. Models are defined interactively using standard R
syntax — arithmetic operators, mathematical functions, and distribution
assignments — then compiled into differentiable computation graphs for
inference via Hamiltonian Monte Carlo (HMC), the No-U-Turn Sampler (NUTS),
Automatic Differentiation Variational Inference (ADVI), maximum a posteriori
(MAP) estimation, or Laplace approximation.

Unlike existing R Bayesian tools that depend on external languages (Stan) or
Python runtimes (TensorFlow via greta), `gretaR` requires no Python
installation and no separate modelling language. The torch backend provides
automatic differentiation and optional GPU acceleration through a pure
R interface.

# Statement of Need

Bayesian statistical modelling is central to modern quantitative research in
ecology, epidemiology, social science, and engineering. The R ecosystem offers
several mature tools — `brms` [@brms] for formula-based models via Stan
[@stan], `NIMBLE` [@nimble] for programmable MCMC, and `INLA` [@inla] for
approximate inference on latent Gaussian models. However, each has limitations
that restrict accessibility or flexibility:

- **Stan/brms** requires learning a separate modelling language and managing
  a C++ toolchain.
- **greta** [@greta] provides an elegant R-native DSL but depends on Python,
  TensorFlow, and reticulate — a fragile installation chain that is the
  primary source of user frustration.
- **NIMBLE** uses BUGS-like syntax that differs from standard R conventions.

`gretaR` addresses these gaps by providing:

1. **Pure R experience** — models are defined using standard R operators and
   functions on `gretaR_array` objects. No new language to learn.
2. **Zero Python dependency** — the `torch` R package links directly to
   libtorch (C++), installed via `torch::install_torch()`.
3. **Multiple inference methods** — NUTS, HMC, ADVI (mean-field and
   full-rank), MAP, and Laplace approximation in a single package.
4. **Ecosystem integration** — posterior draws are returned as
   `posterior::draws_array` objects, directly compatible with `bayesplot`,
   `loo`, and `tidybayes`.

# Key Features

**18 probability distributions** including Normal, Half-Normal, Half-Cauchy,
Student-t, Beta, Gamma, Exponential, Poisson, Binomial, Bernoulli, Dirichlet,
Negative Binomial, LKJ Correlation, Log-Normal, Cauchy, Wishart, Multivariate
Normal, and Uniform — each with differentiable log-probability functions and
appropriate parameter transforms for unconstrained sampling.

**Hierarchical models** via array indexing (`alpha[group_id]`), supporting
multi-level structures with non-centred parameterisations.

**Formula interface** (`gretaR_glm`) for rapid specification of generalised
linear models and mixed-effects models using familiar R formula syntax,
including lme4-style random effects `(1|group)`.

**Custom distributions** via user-supplied torch-differentiable log-probability
functions, and **mixture models** with automatic log-sum-exp marginalisation
over discrete components.

**Sparse matrix support** via the `Matrix` package for efficient handling of
high-dimensional design matrices.

# Validation

`gretaR` was validated against CmdStan [@stan] on 10 benchmark models ranging
from simple normal mean estimation to hierarchical random-intercepts models.
Parameter estimates agree to 2–3 significant figures across all benchmarks.
Performance optimisations (compiled log-probability functions, `autograd_grad`)
reduced the gretaR–Stan timing gap from approximately 50× to 13–16× for
small models (1–5 parameters). The full validation suite is included in
`inst/validation/`.

# Example

```r
library(gretaR)

# Data
x <- as_data(mtcars$wt)
y <- as_data(mtcars$mpg)

# Priors
alpha <- normal(0, 50)
beta <- normal(0, 10)
sigma <- half_cauchy(5)

# Model
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)
m <- model(alpha, beta, sigma)

# Inference
draws <- mcmc(m, n_samples = 1000, warmup = 1000)
summary(draws)
```

# Acknowledgements

`gretaR` builds on the `torch` R package maintained by Daniel Falbel and
the mlverse team at Posit, and is inspired by the design of the original
`greta` package by Nick Golding. The `posterior`, `bayesplot`, and `loo`
packages by the Stan Development Team provide the diagnostic and
visualisation infrastructure.

# References
