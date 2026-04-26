# gretaR

**Bayesian statistical modelling in R, powered by torch – no Python
required.**

gretaR is a probabilistic programming package that lets you define
Bayesian models interactively using native R syntax, then compile them
to torch tensors for GPU-accelerated inference via HMC, NUTS, ADVI, MAP,
and Laplace approximation.

## Installation

``` r
# Install from GitHub with vignettes (requires R >= 4.1.0)
# install.packages("remotes")
remotes::install_github("max578/gretaR", build_vignettes = TRUE)

# Install the torch backend (one-time setup)
torch::install_torch()
```

After installation, browse the bundled vignettes:

``` r
browseVignettes("gretaR")
```

## Quick Start

Bayesian linear regression in 15 lines:

``` r
library(gretaR)

# Observed data
x <- as_data(mtcars$wt)
y <- as_data(mtcars$mpg)

# Priors
alpha <- normal(0, 10)
beta  <- normal(0, 10)
sigma <- half_cauchy(5)

# Likelihood
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)

# Compile and sample
m <- model(alpha, beta, sigma)
draws <- mcmc(m, n_samples = 1000, chains = 4)
summary(draws)
```

## Feature Highlights

- **18 distributions** – Normal, HalfNormal, HalfCauchy, StudentT,
  Uniform, Bernoulli, Binomial, Poisson, Gamma, Beta, Exponential,
  MultivariateNormal, Dirichlet, NegativeBinomial, LKJ, LogNormal,
  Cauchy, Wishart
- **5 inference methods** – HMC, NUTS, ADVI (mean-field and full-rank),
  MAP, Laplace approximation
- **Formula interface** –
  [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md)
  for specifying GLMs with standard R formula syntax (Gaussian,
  Binomial, Poisson families)
- **Hierarchical models** – nested and crossed random effects via the
  core DSL
- **Sparse matrix support** – efficient large design matrices via
  `as_data_sparse()` and `sparse_matmul()`
- **Native R torch backend** – automatic differentiation, GPU
  acceleration, zero Python dependency
- **Ecosystem integration** – posterior draws returned as
  [`posterior::draws_array`](https://mc-stan.org/posterior/reference/draws_array.html)
  for seamless use with bayesplot, loo, and other tidyverse-compatible
  tools

## Comparison with Other Frameworks

| Feature                       | gretaR                                                                    | greta               | brms           | RStan                    |
|-------------------------------|---------------------------------------------------------------------------|---------------------|----------------|--------------------------|
| **Backend**                   | torch (R)                                                                 | TensorFlow (Python) | Stan (C++)     | Stan (C++)               |
| **Python dependency**         | None                                                                      | Required            | None           | None                     |
| **Installation**              | Simple                                                                    | Fragile (TF/Python) | Moderate       | Moderate (C++ toolchain) |
| **Syntax style**              | Interactive R DSL                                                         | Interactive R DSL   | Formula        | Stan language            |
| **GPU support**               | Yes (CUDA/MPS)                                                            | Yes (CUDA)          | No             | Limited (OpenCL)         |
| **Automatic differentiation** | torch autograd                                                            | TF autograd         | Stan autodiff  | Stan autodiff            |
| **HMC/NUTS**                  | Yes                                                                       | Yes                 | Yes (via Stan) | Yes                      |
| **Variational inference**     | ADVI (MF + FR)                                                            | No                  | No             | Yes                      |
| **Formula interface**         | [`gretaR_glm()`](https://max578.github.io/gretaR/reference/gretaR_glm.md) | No                  | `brm()`        | No                       |
| **Maintenance status**        | Active                                                                    | Maintenance-only    | Active         | Active                   |

## Inference Methods

| Method                  | Function                                                                                 | Use Case                                  |
|-------------------------|------------------------------------------------------------------------------------------|-------------------------------------------|
| Hamiltonian Monte Carlo | [`hmc()`](https://max578.github.io/gretaR/reference/hmc.md) / `mcmc(sampler = "hmc")`    | Full posterior, simpler models            |
| No-U-Turn Sampler       | [`nuts()`](https://max578.github.io/gretaR/reference/nuts.md) / `mcmc(sampler = "nuts")` | Full posterior, general purpose (default) |
| ADVI (mean-field)       | `variational(method = "meanfield")`                                                      | Fast approximate posterior                |
| ADVI (full-rank)        | `variational(method = "fullrank")`                                                       | Approximate posterior with correlations   |
| MAP estimation          | [`opt()`](https://max578.github.io/gretaR/reference/opt.md)                              | Point estimate (mode of posterior)        |
| Laplace approximation   | [`laplace()`](https://max578.github.io/gretaR/reference/laplace.md)                      | Gaussian approximation around MAP         |

## Vignettes

- [Getting
  Started](https://max578.github.io/gretaR/vignettes/getting-started.Rmd)
  – core DSL walkthrough with a complete linear regression example
- [Hierarchical
  Models](https://max578.github.io/gretaR/vignettes/hierarchical-models.Rmd)
  – random effects, partial pooling, and nested structures
- [GLM Models](https://max578.github.io/gretaR/vignettes/glm-models.Rmd)
  – formula interface for Gaussian, logistic, and Poisson regression

After installation, browse vignettes with:

``` r
browseVignettes("gretaR")
```

## Citation

If you use gretaR in your research, please cite:

``` r
citation("gretaR")
```

## Contributing

gretaR is currently in active development. Bug reports and feature
requests are welcome via [GitHub
Issues](https://github.com/max578/gretaR/issues).

## Licence

MIT. See [LICENSE](https://max578.github.io/gretaR/LICENSE) for details.
