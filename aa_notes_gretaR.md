# Development Plan for a Next-Generation R Bayesian Modeling Package, with the reference to https://greta-stats.org and https://github.com/greta-dev/greta. Critically review the draft below, critically assess and come out with practically executable plan. Save the new plan as plan_v1.md

**Project Goal**  
Create a new R package that provides **exactly the same model estimation functionality** as the original **greta** package, but is significantly **more versatile**, **more efficient**, and works **directly with R** (no Python/reticulate dependency for end users). The package will leverage modern tensor frameworks while maintaining full compatibility with greta’s elegant R-native syntax.

## Overview of greta

**greta** is an R package for Bayesian statistical modeling using native R code. Models are defined interactively with `greta_array` objects, then compiled to TensorFlow + TensorFlow Probability for automatic differentiation and MCMC inference (primarily Hamiltonian Monte Carlo).

### Key Features
- Pure R syntax for model definition (no new language to learn)
- `as_data()`, `variable()`, distribution functions, and standard R operations
- Automatic construction of the joint posterior
- HMC sampling via TensorFlow backend
- Good scalability to large data and GPUs
- Extensible through additional packages (e.g., greta.gp)

### Current Limitations
- Heavy reliance on Python/reticulate and specific TensorFlow versions
- Installation fragility (conda environment issues, platform compatibility)
- Limited to static HMC (weaker support for discrete parameters)
- Overhead from reticulate graph translation
- Smaller ecosystem compared to Stan, brms, or NIMBLE

## Proposed New Package Strategy

**Suggested Names**: `torchbayes`, `rtensorbayes`, or `greta2`

### Core Design Principles
- **Pure R experience**: Zero Python/reticulate dependency for users
- **Exact functional parity** with greta: same syntax for data, variables, operations, distributions, and model definition
- **Much higher versatility**: broader inference methods, discrete parameter support, better extensibility
- **Superior efficiency**: better GPU utilization, lower overhead, JIT compilation
- **Modern open-source backend**: primarily **torch** (via the `torch` R package), with optional TensorFlow fallback

### Why torch as Primary Backend?
- Native C++/libtorch integration (no Python)
- Excellent autograd, GPU support, and tensor performance
- `torch::install_torch()` provides seamless installation
- Dynamic execution + tracing/JIT capabilities
- Already proven in R for Bayesian neural networks and probabilistic modeling

### High-Level Architecture

1. **DSL Layer (User-Facing)**
   - `torchbayes_array` (or S7/R6 objects) that behave like greta arrays
   - Full compatibility with greta syntax:
     - `as_data()`, `normal()`, `variable()`, `distribution(y) <- ...`
     - Standard R arithmetic, functions, and subsetting
   - Lazy tracing or torch-based dynamic graph construction

2. **Posterior Builder**
   - Automatically constructs the log joint density (priors + likelihood)
   - Support for variable transforms and constraints

3. **Inference Engine (More Versatile)**
   - Core: Hamiltonian Monte Carlo (HMC) + No-U-Turn Sampler (NUTS)
   - Additional methods: Variational Inference (VI), MAP optimization
   - Future: hybrid MCMC for discrete parameters, Stan backend plugin
   - Vectorized chains, multi-GPU support

4. **Backend Abstraction**
   - Default: `torch` (pure R, high performance)
   - Optional: `tensorflow` + `tfprobability` R packages
   - Pluggable architecture for future backends (JAX, NIMBLE codegen, etc.)

5. **Efficiency Enhancements**
   - JIT compilation of the log-probability function
   - Automatic tensor batching and vectorization
   - GPU-first design with seamless multi-GPU support
   - Targeted 2–5× speedup over original greta on GPU/large models

### Key Improvements Over greta

- **Versatility**
  - Full discrete parameter support (via reparameterization or auxiliary variables)
  - Built-in support for Gaussian Processes, ODEs, neural network layers
  - Tighter integration with tidyverse, tidybayes, bayesplot
  - Easy user-defined functions and custom likelihoods

- **Efficiency**
  - Lower runtime overhead (no reticulate)
  - Better JIT and autograd performance
  - Superior GPU scaling

- **Usability**
  - Simple installation (`torch::install_torch()`)
  - Robust across platforms (Windows, macOS, Linux)
  - Comprehensive diagnostics and visualization tools

### Development Roadmap

1. **Prototype Phase (1–2 months)**
   - Implement core DSL with torch backend
   - Reproduce basic greta examples (linear regression, hierarchical models)
   - Ensure syntax compatibility

2. **Inference Implementation**
   - Static HMC using torch autograd
   - Add NUTS sampler
   - Implement variational inference

3. **Distributions & Operations**
   - Wrap/implement core probability distributions
   - Support all greta mathematical operations

4. **Testing & Validation**
   - Unit tests and integration tests
   - Benchmarking against original greta, brms/cmdstanr, and NIMBLE
   - Numerical equivalence checks

5. **Polish & Release**
   - Documentation (pkgdown site, vignettes, migration guide)
   - CRAN-ready packaging
   - Example gallery and extension package framework

6. **Advanced Features (Phase 2)**
   - Hybrid samplers for discrete parameters
   - Multi-backend support
   - Export to Stan/ONNX for interoperability
   - Community extension packages

### Potential Challenges & Mitigations

- **Implementing NUTS/HMC**: Start with static HMC, then extend; leverage open-source HMC implementations
- **Distribution coverage**: Mirror torch.distributions and community contributions
- **Performance tuning**: Early and continuous benchmarking
- **Adoption**: Maintain 100% greta syntax compatibility + clear migration path

### Expected Benefits

- Dramatically improved installation and reliability
- Faster sampling, especially on GPUs
- Broader range of models and inference methods
- Strong foundation for a thriving R Bayesian ecosystem
- Positioned as the modern, tensor-first Bayesian DSL for R

---

**Next Steps Recommendation**  
Begin with a minimal viable prototype focusing on linear and hierarchical models using the `torch` backend. Once core syntax and HMC sampling are stable, expand to NUTS and additional features.

This plan delivers a drop-in replacement that is **more robust**, **faster**, and **far more versatile** while preserving the elegant R-native modeling experience that makes greta special.


