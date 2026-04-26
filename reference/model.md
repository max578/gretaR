# Create a gretaR Model

Compile the computation graph defined by the specified target variables
into a differentiable log-joint-density function suitable for HMC/NUTS
inference.

## Usage

``` r
model(..., precision = c("float32", "float64"))
```

## Arguments

- ...:

  `gretaR_array` objects representing the parameters of interest.

- precision:

  Torch dtype: `"float32"` (default) or `"float64"`.

## Value

A `gretaR_model` object with `log_prob()` and `grad_log_prob()` methods.

## Examples

``` r
if (FALSE) { # \dontrun{
alpha <- normal(0, 10)
beta <- normal(0, 5)
sigma <- half_cauchy(1)
y <- as_data(rnorm(100))
x <- as_data(rnorm(100))
mu <- alpha + beta * x
distribution(y) <- normal(mu, sigma)
m <- model(alpha, beta, sigma)
} # }
```
