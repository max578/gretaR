# gretaR: Bayesian Statistical Modelling with Torch

A probabilistic programming package for Bayesian statistical modelling
in R using native R syntax. Models are defined interactively with
`gretaR_array` objects, then compiled to torch tensors for
GPU-accelerated HMC and NUTS inference.

## Core workflow

1.  Wrap observed data with
    [`as_data()`](https://max578.github.io/gretaR/reference/as_data.md)

2.  Define priors using distribution functions (e.g.,
    [`normal()`](https://max578.github.io/gretaR/reference/normal.md),
    [`gamma_dist()`](https://max578.github.io/gretaR/reference/gamma_dist.md))

3.  Define the model structure using standard R operations

4.  Assign a likelihood with
    [`distribution()`](https://max578.github.io/gretaR/reference/distribution.md)

5.  Create a model with
    [`model()`](https://max578.github.io/gretaR/reference/model.md)

6.  Draw samples with
    [`mcmc()`](https://max578.github.io/gretaR/reference/mcmc.md)

## See also

Useful links:

- <https://github.com/max578/gretaR>

- Report bugs at <https://github.com/max578/gretaR/issues>

## Author

**Maintainer**: Max Moldovan <max.moldovan@adelaide.edu.au>
([ORCID](https://orcid.org/0000-0001-9680-8474))
