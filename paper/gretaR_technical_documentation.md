# Introduction

gretaR is a probabilistic programming package for R that enables Bayesian statistical modelling using native R syntax. Models are defined interactively by manipulating `gretaR_array` objects — data wrappers, random variables, and deterministic transformations — which lazily construct a directed acyclic graph (DAG). This graph is then compiled into a differentiable log-joint density function for gradient-based inference.

The package addresses three limitations of existing tools:

1.  **greta** (Golding 2019) provides an elegant R-native DSL but requires Python, TensorFlow, and reticulate — a fragile dependency chain.

2.  **Stan/brms** (Carpenter et al. 2017; Bürkner 2017) requires learning a separate modelling language and managing a C++ toolchain.

3.  **NIMBLE** (Valpine et al. 2017) uses BUGS-like syntax that differs from standard R.

gretaR resolves these by providing pure R syntax with a torch backend (no Python), optional Stan backend (no new language), and five inference methods in a single package.

## Scope of this Document

This document covers every major component of gretaR v0.1.0:

- The directed acyclic graph (DAG) and lazy evaluation system (Section 2)

- The 18 probability distributions and parameter transforms (Section 3)

- Five inference engines: NUTS, HMC, ADVI, MAP, Laplace (Section 4)

- Performance optimisations (Section 5)

- The Stan code generation backend (Section 6)

- The formula interface and random effects (Section 7)

- Custom distributions and mixture models (Section 8)

- Sparse matrix support (Section 9)

- The unified output structure (Section 10)

- Validation results (Section 11)

# Core Architecture: The Directed Acyclic Graph

## Lazy Evaluation and Node Types

gretaR uses lazy evaluation: operations on `gretaR_array` objects do not compute values immediately. Instead, they construct a DAG of interconnected nodes. Each node is an R6 object (`GretaRArray`) with fields:

- **`node_type`**: one of `"data"`, `"variable"`, or `"operation"`

- **`value`**: a torch tensor (fixed for data; managed by the sampler for variables)

- **`parents`**: character vector of parent node identifiers

- **`operation`**: a closure that computes this node’s value from its parents

- **`distribution`**: a prior distribution (for variable nodes)

- **`constraint`**: bounds on the parameter space

- **`transform`**: a bijector mapping between constrained and unconstrained spaces

## Model Compilation

The `model()` function compiles the DAG into a differentiable log-joint density function:

1.  **Identify free variables**: traverse the DAG to find all variable nodes

2.  **Exclude likelihood templates**: variable nodes used on the right-hand side of `distribution(y) <-` are not free parameters

3.  **Build parameter layout**: compute offsets for packing/unpacking a flat parameter vector $`\boldsymbol\theta_{\text{free}} \in \mathbb{R}^d`$

4.  **Construct `log_prob()`**: a function $`f: \mathbb{R}^d \to \mathbb{R}`$ that computes:

``` math
\log p(\boldsymbol\theta, \mathbf{y}) = \underbrace{\sum_{k=1}^{K} \log p(\theta_k \mid \text{prior}_k)}_{\text{priors}} + \underbrace{\sum_{k=1}^{K} \log \left| \frac{\partial T_k^{-1}}{\partial \theta_k^{\text{free}}} \right|}_{\text{Jacobian}} + \underbrace{\sum_{i=1}^{N} \log p(y_i \mid \boldsymbol\theta)}_{\text{likelihood}}
```

where $`T_k`$ is the bijector for parameter $`k`$.

## Gradient Computation

Gradients are computed via torch’s automatic differentiation:

<div class="snugshade">

</div>

The use of `autograd_grad()` (rather than `backward()`) avoids gradient accumulation overhead and eliminates the need for gradient zeroing between evaluations.

# Probability Distributions

gretaR implements 18 probability distributions, each as an R6 class inheriting from `GretaRDistribution`. Every distribution provides:

- `log_prob(x)`: differentiable log-density evaluated at $`x`$ (returns a scalar torch tensor)

- `sample(n)`: draw $`n`$ samples (for prior predictive checks)

- `constraint`: parameter space bounds (used to select the appropriate bijector)

## Continuous Distributions

<table>
<caption>Continuous distributions. *Simplex, correlation, and PD transforms are deferred to a future release.</caption>
<thead>
<tr>
<th style="text-align: left;"><div class="minipage">
<p>Distribution</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Constructor</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Support</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Transform</p>
</div></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><div class="minipage">
<p>Distribution</p>
</div></td>
<td style="text-align: left;"><div class="minipage">
<p>Constructor</p>
</div></td>
<td style="text-align: left;"><div class="minipage">
<p>Support</p>
</div></td>
<td style="text-align: left;"><div class="minipage">
<p>Transform</p>
</div></td>
</tr>
<tr>
<td style="text-align: left;">Normal</td>
<td style="text-align: left;"><code>normal()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ</span></td>
<td style="text-align: left;">Identity</td>
</tr>
<tr>
<td style="text-align: left;">Half-Normal</td>
<td style="text-align: left;"><code>half_normal()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;">Log</td>
</tr>
<tr>
<td style="text-align: left;">Half-Cauchy</td>
<td style="text-align: left;"><code>half_cauchy()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;">Log</td>
</tr>
<tr>
<td style="text-align: left;">Student-<span class="math inline"><em>t</em></span></td>
<td style="text-align: left;"><code>student_t()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ</span></td>
<td style="text-align: left;">Identity</td>
</tr>
<tr>
<td style="text-align: left;">Cauchy</td>
<td style="text-align: left;"><code>cauchy()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ</span></td>
<td style="text-align: left;">Identity</td>
</tr>
<tr>
<td style="text-align: left;">Exponential</td>
<td style="text-align: left;"><code>exponential()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;">Log</td>
</tr>
<tr>
<td style="text-align: left;">Gamma</td>
<td style="text-align: left;"><code>gamma_dist()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;">Log</td>
</tr>
<tr>
<td style="text-align: left;">Beta</td>
<td style="text-align: left;"><code>beta_dist()</code></td>
<td style="text-align: left;"><span class="math inline">[0, 1]</span></td>
<td style="text-align: left;">Logit</td>
</tr>
<tr>
<td style="text-align: left;">Log-Normal</td>
<td style="text-align: left;"><code>lognormal()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;">Log</td>
</tr>
<tr>
<td style="text-align: left;">Uniform</td>
<td style="text-align: left;"><code>uniform()</code></td>
<td style="text-align: left;"><span class="math inline">[<em>a</em>, <em>b</em>]</span></td>
<td style="text-align: left;">Scaled logit</td>
</tr>
<tr>
<td style="text-align: left;">Multivariate Normal</td>
<td style="text-align: left;"><code>multivariate_normal()</code></td>
<td style="text-align: left;"><span class="math inline">ℝ<sup><em>k</em></sup></span></td>
<td style="text-align: left;">Identity</td>
</tr>
<tr>
<td style="text-align: left;">Dirichlet</td>
<td style="text-align: left;"><code>dirichlet()</code></td>
<td style="text-align: left;">Simplex</td>
<td style="text-align: left;">Identity*</td>
</tr>
<tr>
<td style="text-align: left;">LKJ Correlation</td>
<td style="text-align: left;"><code>lkj_correlation()</code></td>
<td style="text-align: left;">Corr. matrix</td>
<td style="text-align: left;">Identity*</td>
</tr>
<tr>
<td style="text-align: left;">Wishart</td>
<td style="text-align: left;"><code>wishart()</code></td>
<td style="text-align: left;">PD matrix</td>
<td style="text-align: left;">Identity*</td>
</tr>
</tbody>
</table>

## Discrete Distributions

<table>
<caption>Discrete distributions (used as likelihoods; not directly sampled via HMC).</caption>
<thead>
<tr>
<th style="text-align: left;"><div class="minipage">
<p>Distribution</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Constructor</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Support</p>
</div></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><div class="minipage">
<p>Distribution</p>
</div></td>
<td style="text-align: left;"><div class="minipage">
<p>Constructor</p>
</div></td>
<td style="text-align: left;"><div class="minipage">
<p>Support</p>
</div></td>
</tr>
<tr>
<td style="text-align: left;">Bernoulli</td>
<td style="text-align: left;"><code>bernoulli()</code></td>
<td style="text-align: left;"><span class="math inline">{0, 1}</span></td>
</tr>
<tr>
<td style="text-align: left;">Binomial</td>
<td style="text-align: left;"><code>binomial()</code></td>
<td style="text-align: left;"><span class="math inline">{0, …, <em>n</em>}</span></td>
</tr>
<tr>
<td style="text-align: left;">Poisson</td>
<td style="text-align: left;"><code>poisson_dist()</code></td>
<td style="text-align: left;"><span class="math inline">ℤ<sub> ≥ 0</sub></span></td>
</tr>
<tr>
<td style="text-align: left;">Negative Binomial</td>
<td style="text-align: left;"><code>negative_binomial()</code></td>
<td style="text-align: left;"><span class="math inline">ℤ<sub> ≥ 0</sub></span></td>
</tr>
</tbody>
</table>

## Parameter Transforms (Bijectors)

Six bijectors map between constrained parameter spaces and the unconstrained real line, as required by gradient-based samplers:

<table>
<thead>
<tr>
<th style="text-align: left;"><div class="minipage">
<p>Bijector</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Domain</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Forward <span class="math inline"><em>T</em>(<em>x</em>)</span></p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Inverse <span class="math inline"><em>T</em><sup>−1</sup>(<em>y</em>)</span></p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p><span class="math inline">log |<em>J</em>|</span></p>
</div></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Identity</td>
<td style="text-align: left;"><span class="math inline">ℝ</span></td>
<td style="text-align: left;"><span class="math inline"><em>y</em> = <em>x</em></span></td>
<td style="text-align: left;"><span class="math inline"><em>x</em> = <em>y</em></span></td>
<td style="text-align: left;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: left;">Log</td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;"><span class="math inline"><em>y</em> = log <em>x</em></span></td>
<td style="text-align: left;"><span class="math inline"><em>x</em> = <em>e</em><sup><em>y</em></sup></span></td>
<td style="text-align: left;"><span class="math inline">∑<em>y</em><sub><em>i</em></sub></span></td>
</tr>
<tr>
<td style="text-align: left;">Logit</td>
<td style="text-align: left;"><span class="math inline">(0, 1)</span></td>
<td style="text-align: left;"><span class="math inline">$y = \log\frac{x}{1-x}$</span></td>
<td style="text-align: left;"><span class="math inline"><em>x</em> = <em>σ</em>(<em>y</em>)</span></td>
<td style="text-align: left;"><span class="math inline">∑[log <em>σ</em>(<em>y</em>) + log (1 − <em>σ</em>(<em>y</em>))]</span></td>
</tr>
<tr>
<td style="text-align: left;">Scaled logit</td>
<td style="text-align: left;"><span class="math inline">(<em>a</em>, <em>b</em>)</span></td>
<td style="text-align: left;">Shifted logit</td>
<td style="text-align: left;">Shifted sigmoid</td>
<td style="text-align: left;">Includes range adjustment</td>
</tr>
<tr>
<td style="text-align: left;">Softplus</td>
<td style="text-align: left;"><span class="math inline">ℝ<sup>+</sup></span></td>
<td style="text-align: left;"><span class="math inline"><em>y</em> = log (<em>e</em><sup><em>x</em></sup> − 1)</span></td>
<td style="text-align: left;"><span class="math inline"><em>x</em> = softplus(<em>y</em>)</span></td>
<td style="text-align: left;"><span class="math inline">∑log <em>σ</em>(<em>y</em>)</span></td>
</tr>
<tr>
<td style="text-align: left;">Lower bound</td>
<td style="text-align: left;"><span class="math inline">(<em>L</em>, ∞)</span></td>
<td style="text-align: left;"><span class="math inline"><em>y</em> = log (<em>x</em> − <em>L</em>)</span></td>
<td style="text-align: left;"><span class="math inline"><em>x</em> = <em>e</em><sup><em>y</em></sup> + <em>L</em></span></td>
<td style="text-align: left;"><span class="math inline">∑<em>y</em><sub><em>i</em></sub></span></td>
</tr>
</tbody>
</table>

The `select_transform()` function automatically chooses the bijector based on the distribution’s constraint specification.

# Inference Engines

## No-U-Turn Sampler (NUTS)

The primary inference method implements the NUTS algorithm (Hoffman and Gelman 2014) with the following components:

**Leapfrog integrator.** Given position $`\boldsymbol\theta`$ and momentum $`\mathbf{p}`$, one step with step size $`\epsilon`$ and mass matrix $`\mathbf{M}`$:

``` math
\begin{align}
\mathbf{p} &\leftarrow \mathbf{p} + \frac{\epsilon}{2} \nabla_\theta \log p(\boldsymbol\theta, \mathbf{y}) \\
\boldsymbol\theta &\leftarrow \boldsymbol\theta + \epsilon \mathbf{M}^{-1} \mathbf{p} \\
\mathbf{p} &\leftarrow \mathbf{p} + \frac{\epsilon}{2} \nabla_\theta \log p(\boldsymbol\theta, \mathbf{y})
\end{align}
```

**Tree doubling.** The trajectory is extended by doubling in a randomly chosen direction. At each depth $`j`$, $`2^j`$ leapfrog steps are evaluated. Growth stops when a U-turn is detected: $`\Delta\boldsymbol\theta \cdot \mathbf{p}_{\text{left}} < 0`$ or $`\Delta\boldsymbol\theta \cdot \mathbf{p}_{\text{right}} < 0`$.

**Windowed warmup.** Adaptation follows a three-phase strategy inspired by Stan (Carpenter et al. 2017):

- Phase 1 (0–15%): step-size adaptation via dual averaging (Nesterov 2009)

- Phase 2 (15–90%): collect samples for diagonal mass matrix estimation

- Phase 3 (90–100%): re-adapt step size with the learned mass matrix

This phased approach prevents the mass matrix update from invalidating the previously adapted step size.

**Implementation note.** The leapfrog operates on plain R numeric vectors (not torch tensors) to avoid scoping issues with torch’s `with_no_grad()` in R. Only the gradient evaluation touches torch.

## Hamiltonian Monte Carlo (HMC)

Static HMC with a fixed number of leapfrog steps per iteration. Uses identical warmup adaptation to NUTS. Recommended for debugging or when NUTS tree depth is a concern.

## Automatic Differentiation Variational Inference (ADVI)

Implements the ADVI framework of Kucukelbir et al. (2017) with two variational families:

**Mean-field:** $`q(\boldsymbol\theta) = \mathcal{N}(\boldsymbol\mu, \text{diag}(\boldsymbol\sigma^2))`$

**Full-rank:** $`q(\boldsymbol\theta) = \mathcal{N}(\boldsymbol\mu, \mathbf{L}\mathbf{L}^\top)`$

The ELBO is optimised via the reparameterisation trick:

``` math
\text{ELBO} = \mathbb{E}_{q}[\log p(\mathbf{y}, \boldsymbol\theta)] + \mathcal{H}[q]
```

where $`\boldsymbol\theta = \boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\epsilon`$ and $`\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$.

Convergence is assessed by monitoring the relative change in the median ELBO over a rolling window.

## MAP Estimation

Maximum a posteriori estimation via the Adam optimiser (Kingma and Ba 2015). The initial point is found by gradient ascent toward the posterior mode (200 steps with learning rate 0.1), providing robust initialisation even for multimodal posteriors.

## Laplace Approximation

Approximates the posterior as a multivariate normal centred at the MAP:

``` math
p(\boldsymbol\theta \mid \mathbf{y}) \approx \mathcal{N}\left(\hat{\boldsymbol\theta}_{\text{MAP}}, \left[-\nabla^2 \log p(\hat{\boldsymbol\theta}_{\text{MAP}}, \mathbf{y})\right]^{-1}\right)
```

The Hessian is computed via symmetric finite differences on the gradient. Eigenvalue clamping ensures positive definiteness. The log marginal likelihood is approximated as:

``` math
\log p(\mathbf{y}) \approx \log p(\hat{\boldsymbol\theta}, \mathbf{y}) + \frac{d}{2}\log(2\pi) + \frac{1}{2}\log|\boldsymbol\Sigma|
```

# Performance Optimisations

Three optimisations reduce the torch backend’s per-gradient-evaluation cost:

## Compiled Log-Probability Function

At model compilation time, `compile_log_prob()` pre-extracts all parameter layout information (offsets, dimensions, transforms, distribution references) into a closure. This eliminates R6 method dispatch and list lookups during the hot path of gradient evaluation.

## `autograd_grad()` vs `backward()`

The standard `backward()` accumulates gradients into the `$grad` attribute, requiring subsequent `$clone()` and `$detach()$cpu()` calls. Using `autograd_grad()` returns gradients directly, reducing per-call overhead by approximately 27%.

## JIT Tracing

The compiled log-probability function is optionally traced via `torch::jit_trace()`, which fuses torch operations into an optimised graph. Tracing succeeds for models without data-dependent control flow; a safe fallback to the untraced compiled function is used otherwise.

**Combined effect:** 3–4$`\times`$ speedup over the initial implementation, reducing the torch–Stan gap from 50–64$`\times`$ to 7–16$`\times`$ for small models.

### Approaches Evaluated and Rejected

<table>
<thead>
<tr>
<th style="text-align: left;"><div class="minipage">
<p>Approach</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Finding</p>
</div></th>
<th style="text-align: left;"><div class="minipage">
<p>Decision</p>
</div></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">GPU/MPS acceleration</td>
<td style="text-align: left;"><span class="math inline">×</span> <em>slower</em> for small tensors due to transfer latency</td>
<td style="text-align: left;">Rejected</td>
</tr>
<tr>
<td style="text-align: left;">C++ leapfrog loop</td>
<td style="text-align: left;">R-level loop overhead is $&lt;$1% of total time</td>
<td style="text-align: left;">Rejected</td>
</tr>
<tr>
<td style="text-align: left;">In-place tensor operations</td>
<td style="text-align: left;">Higher dispatch cost than allocation for small tensors</td>
<td style="text-align: left;">Rejected</td>
</tr>
<tr>
<td style="text-align: left;">Batched chains (vectorised HMC)</td>
<td style="text-align: left;">NUTS trees vary per chain, wasting compute</td>
<td style="text-align: left;">Deferred</td>
</tr>
</tbody>
</table>

# Stan Backend

## Motivation

The torch backend executes the DAG interpretively: each gradient evaluation traverses R-level data structures and dispatches torch operations individually. Stan compiles the entire model to optimised C++ with static automatic differentiation, yielding 32–152$`\times`$ faster sampling on hierarchical models.

gretaR’s Stan backend provides this speed advantage while preserving the R-native DSL:

<div class="snugshade">

<div class="Highlighting">

<span style="color: 0.56,0.35,0.01">*\# Same model definition*</span> mu <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**normal**</span>(<span style="color: 0.00,0.00,0.81">0</span>, <span style="color: 0.00,0.00,0.81">10</span>) sigma <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**half_cauchy**</span>(<span style="color: 0.00,0.00,0.81">2</span>) <span style="color: 0.13,0.29,0.53">**distribution**</span>(y) <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**normal**</span>(mu, sigma) m <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**model**</span>(mu, sigma)

<span style="color: 0.56,0.35,0.01">*\# Choose backend at inference time*</span> fit_torch <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**mcmc**</span>(m, <span style="color: 0.13,0.29,0.53">backend =</span> <span style="color: 0.31,0.60,0.02">"torch"</span>) <span style="color: 0.56,0.35,0.01">*\# native torch NUTS*</span> fit_stan <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**mcmc**</span>(m, <span style="color: 0.13,0.29,0.53">backend =</span> <span style="color: 0.31,0.60,0.02">"stan"</span>) <span style="color: 0.56,0.35,0.01">*\# compiled Stan NUTS*</span>

</div>

</div>

## Code Generation

`compile_to_stan()` walks the compiled model’s DAG and emits valid Stan code:

1.  **Data block**: declares observed data with dimensions inferred from tensor shapes

2.  **Parameters block**: declares free parameters with constraints (e.g., `real<lower=0>`)

3.  **Model block**: emits prior statements and likelihood statements

Each gretaR distribution maps to its Stan equivalent (e.g., `half_cauchy(scale)` $`\to`$ `cauchy(0, scale)` with `<lower=0>` bound).

## Benchmark Results

| Model                        | Torch (s) | Stan (s) | Speedup |
|:-----------------------------|----------:|---------:|:--------|
| Model                        | Torch (s) | Stan (s) | Speedup |
| Intercept-only (2p)          |     351.1 |      3.7 | 95x     |
| Linear regression (3p)       |       2.9 |      3.4 | 1x      |
| Random intercept LMM (23p)   |     466.5 |      4.6 | 101x    |
| Crossed random effects (42p) |    1154.5 |      7.6 | 152x    |
| Logistic GLMM (17p)          |     423.1 |      8.6 | 49x     |
| Poisson GLMM (22p)           |     441.8 |      5.9 | 75x     |
| Random slopes (35p)          |     208.8 |      6.5 | 32x     |

Wall-clock time (500 warmup + 500 samples, 2 chains). Parameter recovery matches to 3 significant figures across backends.

# Formula Interface

## Basic Usage

`gretaR_glm()` provides a high-level interface using standard R formula syntax:

<div class="snugshade">

<div class="Highlighting">

fit <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**gretaR_glm**</span>( Sepal.Length <span style="color: 0.81,0.36,0.00">**~**</span> Sepal.Width <span style="color: 0.81,0.36,0.00">**+**</span> Petal.Length, <span style="color: 0.13,0.29,0.53">data =</span> iris, <span style="color: 0.13,0.29,0.53">family =</span> <span style="color: 0.31,0.60,0.02">"gaussian"</span> )

</div>

</div>

Internally, `model.matrix()` constructs the design matrix; default priors are assigned ($`\beta \sim \mathcal{N}(0, 5)`$, $`\sigma \sim \text{Half-Cauchy}(2)`$); and the model is compiled and sampled via the standard gretaR pipeline.

## Random Effects

lme4-style random effects are parsed via regex (no lme4 dependency required):

<div class="snugshade">

<div class="Highlighting">

fit <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**gretaR_glm**</span>( y <span style="color: 0.81,0.36,0.00">**~**</span> x <span style="color: 0.81,0.36,0.00">**+**</span> (<span style="color: 0.00,0.00,0.81">1</span> <span style="color: 0.81,0.36,0.00">**\|**</span> group), <span style="color: 0.56,0.35,0.01">*\# random intercepts*</span> <span style="color: 0.13,0.29,0.53">data =</span> dat, <span style="color: 0.13,0.29,0.53">family =</span> <span style="color: 0.31,0.60,0.02">"gaussian"</span> )

</div>

</div>

Supported patterns:

- `(1  group)` — random intercepts

- `(x  group)` — correlated random intercepts and slopes

- `(0 + x  group)` — random slopes only

- Multiple terms: `(1  site) + (1  year)`

All random effects use non-centred parameterisation by default: $`\alpha_j = \tau \cdot z_j`$ where $`z_j \sim \mathcal{N}(0, 1)`$ and $`\tau \sim \text{Half-Cauchy}(2)`$. This avoids the funnel geometry that degrades HMC performance in centred parameterisations (Betancourt and Girolami 2015).

## Formula Style Detection

The parser auto-detects formula styles:

| Pattern             | Detected as | Action                            |
|:--------------------|:------------|:----------------------------------|
| `y ~ x1 + x2`       | base R      | Standard GLM via `model.matrix()` |
| `y ~ x + (1\group)` | lme4        | Mixed model with non-centred RE   |
| `y ~ s(x)`          | mgcv        | Error (not yet supported)         |

# Custom Distributions and Mixture Models

## Custom Distributions

Users can define distributions with arbitrary torch-differentiable log-probability functions:

<div class="snugshade">

<div class="Highlighting">

x <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**custom_distribution**</span>( <span style="color: 0.13,0.29,0.53">log_prob_fn =</span> <span style="color: 0.13,0.29,0.53">**function**</span>(x) <span style="color: 0.81,0.36,0.00">**-**</span><span style="color: 0.13,0.29,0.53">**torch_sum**</span>(<span style="color: 0.13,0.29,0.53">**torch_abs**</span>(x)), <span style="color: 0.13,0.29,0.53">constraint =</span> <span style="color: 0.13,0.29,0.53">**list**</span>(<span style="color: 0.13,0.29,0.53">lower =</span> <span style="color: 0.81,0.36,0.00">**-**</span><span style="color: 0.56,0.35,0.01">Inf</span>, <span style="color: 0.13,0.29,0.53">upper =</span> <span style="color: 0.56,0.35,0.01">Inf</span>), <span style="color: 0.13,0.29,0.53">name =</span> <span style="color: 0.31,0.60,0.02">"laplace"</span> )

</div>

</div>

The supplied function must accept a torch tensor and return a scalar torch tensor. It must be differentiable via torch autograd.

## Mixture Models

Finite mixture models are supported via the log-sum-exp trick:

``` math
\log p(x \mid \boldsymbol\pi, \boldsymbol\Theta) = \log \sum_{k=1}^{K} \pi_k \, p_k(x \mid \theta_k) = \text{logsumexp}_k \left[\log \pi_k + \log p_k(x \mid \theta_k)\right]
```

This marginalises over the discrete component indicator, enabling gradient-based inference:

<div class="snugshade">

<div class="Highlighting">

w <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**dirichlet**</span>(<span style="color: 0.13,0.29,0.53">**c**</span>(<span style="color: 0.00,0.00,0.81">1</span>, <span style="color: 0.00,0.00,0.81">1</span>)) mu1 <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**normal**</span>(<span style="color: 0.81,0.36,0.00">**-**</span><span style="color: 0.00,0.00,0.81">2</span>, <span style="color: 0.00,0.00,0.81">1</span>); mu2 <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**normal**</span>(<span style="color: 0.00,0.00,0.81">2</span>, <span style="color: 0.00,0.00,0.81">1</span>) sigma <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**half_cauchy**</span>(<span style="color: 0.00,0.00,0.81">1</span>)

mix <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**mixture**</span>( <span style="color: 0.13,0.29,0.53">distributions =</span> <span style="color: 0.13,0.29,0.53">**list**</span>(<span style="color: 0.13,0.29,0.53">**normal**</span>(mu1, sigma), <span style="color: 0.13,0.29,0.53">**normal**</span>(mu2, sigma)), <span style="color: 0.13,0.29,0.53">weights =</span> w ) <span style="color: 0.13,0.29,0.53">**distribution**</span>(y) <span style="color: 0.56,0.35,0.01">\<-</span> mix

</div>

</div>

# Sparse Matrix Support

Large, sparse design matrices (common in genomics, NLP, and spatial modelling) are handled via the Matrix package (Bates et al. 2024). The `as_data()` function dispatches on `sparseMatrix` objects:

<div class="snugshade">

<div class="Highlighting">

<span style="color: 0.13,0.29,0.53">**library**</span>(Matrix) X_sparse <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**sparseMatrix**</span>(<span style="color: 0.13,0.29,0.53">i =</span> ..., <span style="color: 0.13,0.29,0.53">j =</span> ..., <span style="color: 0.13,0.29,0.53">x =</span> ...) X <span style="color: 0.56,0.35,0.01">\<-</span> <span style="color: 0.13,0.29,0.53">**as_data**</span>(X_sparse) <span style="color: 0.56,0.35,0.01">*\# stored as torch sparse COO tensor*</span> mu <span style="color: 0.56,0.35,0.01">\<-</span> X <span style="color: 0.81,0.36,0.00">**%\*%**</span> beta <span style="color: 0.56,0.35,0.01">*\# sparse-aware matrix multiplication*</span>

</div>

</div>

The Matrix package is a recommended R package shipped with every R installation, imposing zero additional installation burden. Sparse $`\times`$ dense matrix multiplication uses `torch_mm()`, which handles the sparse layout natively.

# Unified Output Structure

All inference functions return a `gretaR_fit` S3 object with consistent structure:

<div class="snugshade">

</div>

S3 methods provide a consistent interface:

- `print(fit)` — concise summary with convergence diagnostics

- `summary(fit)` — full posterior table via `posterior::summarise_draws()`

- `coef(fit)` — named vector of posterior means or MAP estimates

- `plot(fit, type)` — diagnostic plots via bayesplot (`"trace"`, `"density"`, `"pairs"`, `"rhat"`, `"neff"`)

# Validation

gretaR was validated against CmdStan (Carpenter et al. 2017) on 10 benchmark models. Parameter estimates agree to 2–3 significant figures across all benchmarks. The validation suite is included in `inst/validation/` and covers:

- Normal mean estimation (known and unknown variance)

- Linear and multiple regression

- Logistic and Poisson regression

- Gamma GLM and robust regression (Student-$`t`$ errors)

- Beta regression

- Hierarchical random intercepts (non-centred parameterisation)

## Correctness Criteria

A model is considered validated when:

- $`\hat{R} < 1.05`$ for all parameters

- Bulk ESS $`> 400`$ for all parameters

- Posterior means within MCMC uncertainty of Stan reference values

- Zero post-warmup divergent transitions

# Potential for Further Development

## Near-Term (Phase 3)

- **mgcv smooth terms** (`s()`, `te()`) via basis function expansion

- **Simplex and correlation transforms** for Dirichlet and LKJ sampling

- **Wishart/inverse-Wishart sampling** via Bartlett decomposition

- **Improved Stan code generator** with operation-type metadata on DAG nodes

## Medium-Term

- **Gaussian Process module** (`gretaR.gp`) as an extension package

- **ODE-based models** using torch ODE solvers

- **Expectation propagation** for approximate inference

- **S7 class system** migration (when mature)

## Long-Term

- **GPU-accelerated inference** for large latent variable models

- **Batched chain evaluation** for static HMC on GPU

- **Community extension framework** with templates and documentation

- **CRAN submission** and JOSS publication

# References

<span id="refs" label="refs"></span>

<div class="list">

Bates, Douglas, Martin Maechler, and Mikael Jagan. 2024. *Matrix: Sparse and Dense Matrix Classes and Methods*. <https://CRAN.R-project.org/package=Matrix>.

Betancourt, Michael, and Mark Girolami. 2015. “Hamiltonian Monte Carlo for Hierarchical Models.” *Current Trends in Bayesian Methodology with Applications*, 79–101.

Bürkner, Paul-Christian. 2017. “brms: An R Package for Bayesian Multilevel Models Using Stan.” *Journal of Statistical Software* 80 (1): 1–28. <https://doi.org/10.18637/jss.v080.i01>.

Carpenter, Bob, Andrew Gelman, Matthew D. Hoffman, et al. 2017. “Stan: A Probabilistic Programming Language.” *Journal of Statistical Software* 76 (1): 1–32. <https://doi.org/10.18637/jss.v076.i01>.

Golding, Nick. 2019. “Greta: Simple and Scalable Statistical Modelling in R.” *Journal of Open Source Software* 4 (40): 1601. <https://doi.org/10.21105/joss.01601>.

Hoffman, Matthew D., and Andrew Gelman. 2014. “The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.” *Journal of Machine Learning Research* 15: 1593–623.

Kingma, Diederik P., and Jimmy Ba. 2015. “Adam: A Method for Stochastic Optimization.” *arXiv Preprint arXiv:1412.6980*.

Kucukelbir, Alp, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. 2017. “Automatic Differentiation Variational Inference.” *Journal of Machine Learning Research* 18 (14): 1–45.

Nesterov, Yurii. 2009. “Primal-Dual Subgradient Methods for Convex Problems.” *Mathematical Programming* 120 (1): 221–59. <https://doi.org/10.1007/s10107-007-0149-x>.

Valpine, Perry de, Daniel Turek, Christopher J. Paciorek, Clifford Anderson-Bergman, Duncan Temple Lang, and Rastislav Bodik. 2017. “Programming with Models: Writing Statistical Algorithms for General Model Structures with NIMBLE.” *Journal of Computational and Graphical Statistics* 26 (2): 403–13. <https://doi.org/10.1080/10618600.2016.1172487>.

</div>
