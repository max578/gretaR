# gretaR 0.1.0

* Initial development release.
* Core DSL: `as_data()`, `variable()`, `distribution()`, `model()`.
* P0 distributions: Normal, HalfNormal, HalfCauchy, StudentT, Uniform,
  Bernoulli, Binomial, Poisson, Gamma, Beta, Exponential, MultivariateNormal.
* Inference: HMC and NUTS samplers with dual averaging adaptation.
* Output as `posterior::draws_array` for ecosystem interoperability.
