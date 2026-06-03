# Partially-Centred Random-Effect Block

Construct a grouped random-effect term with an explicit centring weight
that interpolates between the non-centred and centred parameterisations.
The weight controls only the sampling geometry seen by the HMC/NUTS
sampler – the implied prior on the group effects, \\u_j \sim
N(\mathrm{mean}, \mathrm{sd}^2)\\, is identical for every weight – so it
is a pure efficiency control for hierarchical models whose mixing is
bottlenecked by the funnel coupling between a group-scale parameter and
its latent effects.

## Usage

``` r
random_effect(group, n_groups, sd, centring = 0, mean = 0)
```

## Arguments

- group:

  Integer vector of length \\n\\ mapping each observation to a group in
  `seq_len(n_groups)`, or the equivalent data `gretaR_array`. Used to
  gather the group effects to the observations.

- n_groups:

  Number of groups \\J\\ (a positive integer).

- sd:

  A scalar `gretaR_array` – the random-effect standard deviation
  (commonly a
  [`half_normal`](https://max578.github.io/gretaR/reference/half_normal.md)
  or
  [`half_cauchy`](https://max578.github.io/gretaR/reference/half_cauchy.md)
  group-scale parameter). The funnel partner of the latent block.

- centring:

  The centring weight \\w\\. Either a single number in \\\[0, 1\]\\
  applied to every group, or a numeric vector of length `n_groups`
  giving a per-group weight. Defaults to `0` (non-centred), gretaR's
  historical hierarchical parameterisation.

- mean:

  The random-effect mean, a numeric scalar or scalar `gretaR_array`.
  Defaults to `0`: the standard zero-mean deviation block whose global
  level lives in a separate intercept term.

## Value

A list of class `gretaR_random_effect` with elements

- `latent`:

  The \\J\\-element latent `gretaR_array` to include in the model's
  target list.

- `effect`:

  The \\n\\-element per-observation effect `gretaR_array` for the linear
  predictor.

- `centring`:

  The resolved per-group weight vector.

- `n_groups`:

  The number of groups.

## Details

A centring weight `w = 0` gives the non-centred form \\u_j =
\mathrm{mean} + \mathrm{sd}\\\xi_j\\ with \\\xi_j \sim N(0, 1)\\, which
mixes well when groups are weakly informed. A weight `w = 1` gives the
centred form \\u_j = \mathrm{mean} + \xi_j\\ with \\\xi_j \sim N(0,
\mathrm{sd}^2)\\, which mixes well when groups are strongly informed.
Intermediate and per-group weights interpolate: \$\$\xi_j \sim N(0,
\mathrm{sd}^{w_j}), \qquad u_j = \mathrm{mean} + \mathrm{sd}^{\\1 -
w_j}\\\xi_j,\$\$ which leaves \\u_j \sim N(\mathrm{mean},
\mathrm{sd}^2)\\ unchanged. The informativeness-optimal weight for a
balanced group of \\n_j\\ observations with residual scale \\\sigma\\ is
\\w_j^\star = (n_j/\sigma^2) / (n_j/\sigma^2 + 1/\mathrm{sd}^2)\\; a
data-adaptive sampler estimates it during warmup, but a fixed value may
be supplied directly here.

The returned latent block `$latent` is the free parameter the sampler
moves and must be passed to
[`model_from_arrays`](https://max578.github.io/gretaR/reference/model_from_arrays.md)
(or [`model`](https://max578.github.io/gretaR/reference/model.md)); the
returned `$effect` is the per-observation effect to add into the linear
predictor. The latent block carries the group-level names, not the
observation-level ones.

## See also

[`model_from_arrays`](https://max578.github.io/gretaR/reference/model_from_arrays.md)
for the programmatic model front-end the latent block is designed for.

## Examples

``` r
if (FALSE) { # \dontrun{
# Random-intercept model, near-centred (informative groups).
n <- 1000L
J <- 20L
g <- rep(seq_len(J), length.out = n)
x <- rnorm(n)
y <- as_data(1.5 - 0.8 * x + rnorm(n))

b0 <- normal(0, 5)
b1 <- normal(0, 5)
s <- half_normal(5)
tau <- half_normal(2)

re <- random_effect(g, n_groups = J, sd = tau, centring = 0.9)
eta <- b0 + b1 * as_data(x) + re$effect
distribution(y) <- normal(eta, s)

m <- model_from_arrays(
  list(b0, b1, s, tau, re$latent),
  likelihood = y,
  names = list("b0", "b1", "s", "tau", paste0("u[", seq_len(J), "]"))
)
} # }
```
