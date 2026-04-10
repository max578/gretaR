# array.R — gretaR_array: the core user-facing object
#
# gretaR_array wraps a node in the computation DAG. Operations on
# gretaR_arrays build the graph lazily; values are only computed at
# model compilation time.

# =============================================================================
# gretaR_array R6 class
# =============================================================================

#' @title gretaR_array
#' @description The core object in gretaR representing a node in the Bayesian
#'   model computation graph.
#' @noRd
GretaRArray <- R6::R6Class(
  "GretaRArray",

  public = list(
    #' @field id Unique node identifier.
    id = NULL,
    #' @field node_type One of "data", "variable", "operation", "distribution".
    node_type = NULL,
    #' @field value Current torch tensor value (set for data; managed by sampler for variables).
    value = NULL,
    #' @field operation Function that computes this node's value from parents.
    operation = NULL,
    #' @field parents List of parent node IDs.
    parents = NULL,
    #' @field distribution GretaRDistribution object (for variable/distribution nodes).
    distribution = NULL,
    #' @field dim Dimensions as integer vector.
    dim_ = NULL,
    #' @field constraint List with lower/upper bounds.
    constraint = NULL,
    #' @field transform Bijector for this variable.
    transform = NULL,
    #' @field name User-assigned name (set by model()).
    node_name = NULL,
    #' @field is_discrete Whether this is a discrete variable.
    is_discrete = FALSE,
    #' @field is_sparse Whether this node holds a sparse tensor.
    is_sparse = FALSE,
    #' @field op_type Operation type string for Stan code generation.
    op_type = NULL,

    initialize = function(node_type, value = NULL, operation = NULL,
                          parents = NULL, distribution = NULL, dim = NULL,
                          constraint = NULL, is_discrete = FALSE) {
      self$id <- new_node_id()
      self$node_type <- node_type
      self$value <- value
      self$operation <- operation
      self$parents <- parents %||% character(0)
      self$distribution <- distribution
      self$is_discrete <- is_discrete
      self$dim_ <- dim %||% c(1L, 1L)

      # Set up constraints and transforms for variables
      if (node_type == "variable" && !is.null(distribution)) {
        self$constraint <- distribution$constraint
        self$transform <- distribution$get_transform()
      } else if (!is.null(constraint)) {
        self$constraint <- constraint
        self$transform <- select_transform(constraint$lower, constraint$upper)
      }

      # Register in global DAG
      register_node(self)
    },

    # Compute this node's value given current state of the DAG
    compute = function() {
      if (self$node_type == "data") {
        return(self$value)
      }
      if (self$node_type == "variable") {
        return(self$value)  # Set by sampler
      }
      if (self$node_type == "operation" && !is.null(self$operation)) {
        # Get parent values
        parent_values <- lapply(self$parents, function(pid) {
          node <- .gretaR_env$dag$nodes[[pid]]
          node$compute()
        })
        return(self$operation(parent_values))
      }
      self$value
    },

    # Set value from unconstrained space (applies inverse transform)
    set_from_unconstrained = function(y) {
      if (!is.null(self$transform)) {
        self$value <- self$transform$inverse(y)
      } else {
        self$value <- y
      }
    }
  )
)

# =============================================================================
# Wrap as S3 for operator dispatch
# =============================================================================

#' Create a gretaR_array S3 wrapper around the R6 object
#' @noRd
wrap_gretaR_array <- function(r6_obj) {
  obj <- list(.node = r6_obj)
  class(obj) <- "gretaR_array"
  obj
}

# Access the underlying R6 node
#' @noRd
get_node <- function(x) {
  if (inherits(x, "gretaR_array")) return(x$.node)
  NULL
}

# =============================================================================
# as_data() — wrap observed data
# =============================================================================

#' @title Wrap Observed Data as a gretaR Array
#'
#' @description Convert numeric vectors, matrices, or arrays into a
#'   \code{gretaR_array} data node for use in model definitions. Data nodes
#'   are fixed (not sampled) during inference.
#'
#' @param x A numeric vector, matrix, or array of observed data.
#' @return A \code{gretaR_array} representing fixed (observed) data.
#' @export
#' @examples
#' \dontrun{
#' y <- as_data(iris$Sepal.Length)
#' X <- as_data(matrix(rnorm(30), ncol = 3))
#' }
as_data <- function(x) {
  if (inherits(x, "gretaR_array")) return(x)

  # Sparse matrix dispatch (Matrix package)
  if (inherits(x, "sparseMatrix")) {
    return(as_data_sparse(x))
  }

  if (!is.numeric(x)) {
    cli_abort("{.arg x} must be numeric, not {.cls {class(x)}}.")
  }

  if (anyNA(x)) {
    cli_abort(c(
      "Missing values ({.val NA}) detected in data passed to {.fn as_data}.",
      "i" = "gretaR requires complete data. Preprocess with {.pkg mice}, {.pkg missRanger}, or {.fn tidyr::drop_na}."
    ))
  }

  # Convert to matrix
  if (is.vector(x)) {
    x_mat <- matrix(x, ncol = 1L)
  } else if (is.matrix(x)) {
    x_mat <- x
  } else {
    x_mat <- as.matrix(x)
  }

  tensor <- torch_tensor(x_mat, dtype = torch_float32())
  node <- GretaRArray$new(
    node_type = "data",
    value = tensor,
    dim = dim(x_mat)
  )
  wrap_gretaR_array(node)
}

# =============================================================================
# variable() — create a free variable (no distribution)
# =============================================================================

#' @title Create a Free Variable
#'
#' @description Create an unconstrained or constrained variable without a
#'   distributional prior. Useful for deterministic transformations or
#'   parameters that do not require a prior.
#'
#' @param lower Lower bound (default \code{-Inf}, unconstrained).
#' @param upper Upper bound (default \code{Inf}, unconstrained).
#' @param dim Integer vector of dimensions (default \code{c(1, 1)}).
#' @return A \code{gretaR_array} variable node.
#' @export
#' @examples
#' \dontrun{
#' x <- variable()
#' x_pos <- variable(lower = 0)
#' x_bounded <- variable(lower = 0, upper = 1)
#' }
variable <- function(lower = -Inf, upper = Inf, dim = NULL) {
  dim <- dim %||% c(1L, 1L)
  if (length(dim) == 1L) dim <- c(dim, 1L)

  constraint <- list(lower = lower, upper = upper)
  node <- GretaRArray$new(
    node_type = "variable",
    dim = dim,
    constraint = constraint
  )
  # Initialise with a reasonable value
  mid <- if (is.finite(lower) && is.finite(upper)) {
    (lower + upper) / 2
  } else if (is.finite(lower)) {
    lower + 1
  } else if (is.finite(upper)) {
    upper - 1
  } else {
    0
  }
  node$value <- torch_full(dim, mid, dtype = torch_float32())
  wrap_gretaR_array(node)
}

# =============================================================================
# create_variable_node — internal helper for distribution constructors
# =============================================================================

#' @noRd
create_variable_node <- function(distribution, dim = NULL, is_discrete = FALSE) {
  dim <- dim %||% distribution$dim %||% c(1L, 1L)
  if (length(dim) == 1L) dim <- c(dim, 1L)

  node <- GretaRArray$new(
    node_type = "variable",
    distribution = distribution,
    dim = dim,
    is_discrete = is_discrete
  )

  # Initialise with a sample or midpoint
  constraint <- distribution$constraint
  mid <- if (is.finite(constraint$lower) && is.finite(constraint$upper)) {
    (constraint$lower + constraint$upper) / 2
  } else if (is.finite(constraint$lower)) {
    constraint$lower + 1
  } else {
    0
  }
  node$value <- torch_full(dim, mid, dtype = torch_float32())
  wrap_gretaR_array(node)
}

# =============================================================================
# distribution() / distribution<-() — assign likelihoods
# =============================================================================

#' @title Get the Distribution of a gretaR Array
#'
#' @description Retrieve the distribution object associated with a
#'   \code{gretaR_array} variable node, or \code{NULL} if none is set.
#'
#' @param x A \code{gretaR_array}.
#' @return The \code{GretaRDistribution} object, or \code{NULL}.
#' @export
#' @examples
#' \dontrun{
#' mu <- normal(0, 1)
#' distribution(mu)
#' }
distribution <- function(x) {
  node <- get_node(x)
  if (is.null(node)) return(NULL)
  node$distribution
}

#' @title Assign a Distribution (Likelihood) to Observed Data
#'
#' @description Define the likelihood by assigning a distribution to a data
#'   \code{gretaR_array}. This registers the distribution as a likelihood
#'   term in the model's log-joint density.
#'
#' @param x A data \code{gretaR_array} (created with \code{\link{as_data}}).
#' @param value A distribution \code{gretaR_array} (e.g., from \code{\link{normal}}).
#' @return The data \code{gretaR_array} \code{x}, invisibly.
#' @export
#' @examples
#' \dontrun{
#' y <- as_data(rnorm(100))
#' mu <- normal(0, 10)
#' sigma <- half_cauchy(1)
#' distribution(y) <- normal(mu, sigma)
#' }
`distribution<-` <- function(x, value) {
  data_node <- get_node(x)
  dist_node <- get_node(value)

  if (is.null(data_node)) {
    cli_abort("Left-hand side of {.code distribution(x) <- ...} must be a gretaR_array.")
  }
  if (is.null(dist_node) || is.null(dist_node$distribution)) {
    cli_abort("Right-hand side must be a distribution (e.g., {.code normal(mu, sigma)}).")
  }

  # Register this as a likelihood term:
  # The distribution from the RHS evaluates log_prob on the LHS data
  register_distribution(data_node$id, dist_node)

  x
}

# =============================================================================
# Indexing operator [.gretaR_array — critical for hierarchical models
# =============================================================================

#' Extract elements from a gretaR_array
#'
#' Enables indexing into group-level parameters for hierarchical models.
#' Supports integer vector indexing (e.g., `alpha[group_id]`).
#'
#' @param x A gretaR_array.
#' @param i Index: an integer vector or a gretaR_array of integer indices.
#' @param j Optional second index (for 2D arrays).
#' @param ... Additional arguments (ignored).
#' @param drop Logical (ignored; always returns a gretaR_array).
#' @return A new gretaR_array with elements selected by the index.
#' @export
`[.gretaR_array` <- function(x, i, j, ..., drop = TRUE) {
  node <- get_node(x)

  # --- Determine the index vector and create an index data node ---
  if (inherits(i, "gretaR_array")) {
    # i is already a gretaR_array (e.g., from as_data(group_id))
    idx_node <- get_node(i)
  } else if (is.numeric(i) || is.integer(i)) {
    # Plain R integer/numeric vector — wrap as a data node
    # Store as float32 tensor; we convert to long inside the operation
    i <- as.integer(i)
    idx_tensor <- torch_tensor(matrix(i, ncol = 1L), dtype = torch_float32())
    idx_r6 <- GretaRArray$new(
      node_type = "data",
      value = idx_tensor,
      dim = c(length(i), 1L)
    )
    idx_node <- idx_r6
  } else if (is.logical(i)) {
    # Logical indexing: convert to integer positions
    i <- which(i)
    idx_tensor <- torch_tensor(matrix(i, ncol = 1L), dtype = torch_float32())
    idx_r6 <- GretaRArray$new(
      node_type = "data",
      value = idx_tensor,
      dim = c(length(i), 1L)
    )
    idx_node <- idx_r6
  } else {
    cli_abort("Index for {.cls gretaR_array} must be integer, logical, or a gretaR_array.")
  }

  # --- Infer output dimensions ---
  n_idx <- idx_node$dim_[1]
  # Parent shape: rows x cols. Indexing selects rows.
  out_cols <- node$dim_[2]
  out_dim <- c(as.integer(n_idx), as.integer(out_cols))

  # --- Create the operation node ---
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) {
      parent_val <- pvals[[1]]   # [n_groups, cols] tensor
      idx_val    <- pvals[[2]]   # [n_idx, 1] tensor (1-based R indices)

      # Convert to 0-based long index vector for torch_index_select
      idx_long <- idx_val$squeeze(2L)$to(dtype = torch_long()) - 1L

      # torch_index_select uses 0-based indexing when called via $index_select,

      # but the R torch binding for torch_index_select uses 1-based.
      # We need 1-based indices for the R torch API.
      idx_1based <- idx_val$squeeze(2L)$to(dtype = torch_long())

      torch_index_select(parent_val, dim = 1L, index = idx_1based)
    },
    parents = c(node$id, idx_node$id),
    dim = out_dim
  )
  result_node$op_type <- "index_select"

  wrap_gretaR_array(result_node)
}

# =============================================================================
# S3 methods: print, dim, length
# =============================================================================

#' @export
print.gretaR_array <- function(x, ...) {

  node <- get_node(x)
  type_str <- switch(node$node_type,
    data = "data",
    variable = if (!is.null(node$distribution)) {
      paste0("variable (", node$distribution$name, ")")
    } else {
      "free variable"
    },
    operation = "operation",
    "unknown"
  )
  dim_str <- paste(node$dim_, collapse = " x ")
  cat(sprintf("gretaR array (%s)\n", type_str))
  cat(sprintf("  dim: %s\n", dim_str))

  if (!is.null(node$constraint)) {
    cat(sprintf("  constraint: [%s, %s]\n",
                format(node$constraint$lower), format(node$constraint$upper)))
  }
  invisible(x)
}

#' @export
dim.gretaR_array <- function(x) {
  get_node(x)$dim_
}

#' @export
length.gretaR_array <- function(x) {
  prod(get_node(x)$dim_)
}

# =============================================================================
# Operator overloading (Ops group generic)
# =============================================================================

#' @export
Ops.gretaR_array <- function(e1, e2) {
  generic <- .Generic

  # Handle unary operations (e.g., -x)
  if (missing(e2)) {
    node1 <- get_node(e1)
    op_fn <- switch(generic,
      "-" = function(parents) -parents[[1]]$compute(),
      "+" = function(parents) parents[[1]]$compute(),
      cli_abort("Unsupported unary operation: {generic}")
    )
    result_node <- GretaRArray$new(
      node_type = "operation",
      operation = function(pvals) {
        switch(generic,
          "-" = -pvals[[1]],
          "+" = pvals[[1]]
        )
      },
      parents = node1$id,
      dim = node1$dim_
    )
    return(wrap_gretaR_array(result_node))
  }

  # Binary operations
  # Ensure both sides have nodes
  if (!inherits(e1, "gretaR_array")) e1 <- as_data(e1)
  if (!inherits(e2, "gretaR_array")) e2 <- as_data(e2)

  node1 <- get_node(e1)
  node2 <- get_node(e2)

  # Determine output dimensions (broadcasting)
  out_dim <- broadcast_dims(node1$dim_, node2$dim_)

  # Create operation node
  op <- generic  # Capture for closure
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) {
      a <- pvals[[1]]
      b <- pvals[[2]]
      switch(op,
        "+" = a + b,
        "-" = a - b,
        "*" = a * b,
        "/" = a / b,
        "^" = torch_pow(a, b),
        "%%" = torch_fmod(a, b),
        ">" = (a > b)$to(dtype = torch_float32()),
        "<" = (a < b)$to(dtype = torch_float32()),
        ">=" = (a >= b)$to(dtype = torch_float32()),
        "<=" = (a <= b)$to(dtype = torch_float32()),
        "==" = (a == b)$to(dtype = torch_float32()),
        "!=" = (a != b)$to(dtype = torch_float32()),
        cli_abort("Unsupported operation: {op}")
      )
    },
    parents = c(node1$id, node2$id),
    dim = out_dim
  )
  result_node$op_type <- paste0("binary_", op)
  wrap_gretaR_array(result_node)
}

# =============================================================================
# Math group generic (log, exp, sqrt, abs, etc.)
# =============================================================================

#' @export
Math.gretaR_array <- function(x, ...) {
  generic <- .Generic
  node <- get_node(x)

  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) {
      val <- pvals[[1]]
      switch(generic,
        "log" = torch_log(val),
        "exp" = torch_exp(val),
        "sqrt" = torch_sqrt(val),
        "abs" = torch_abs(val),
        "sign" = torch_sign(val),
        "floor" = torch_floor(val),
        "ceiling" = torch_ceil(val),
        "round" = torch_round(val),
        "cos" = torch_cos(val),
        "sin" = torch_sin(val),
        "tan" = torch_tan(val),
        "acos" = torch_acos(val),
        "asin" = torch_asin(val),
        "atan" = torch_atan(val),
        "lgamma" = torch_lgamma(val),
        "digamma" = torch_digamma(val),
        cli_abort("Unsupported math function: {generic}")
      )
    },
    parents = node$id,
    dim = node$dim_
  )
  result_node$op_type <- paste0("math_", generic)
  wrap_gretaR_array(result_node)
}

# =============================================================================
# Matrix operations
# =============================================================================

#' @export
t.gretaR_array <- function(x) {
  node <- get_node(x)
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) torch_t(pvals[[1]]),
    parents = node$id,
    dim = rev(node$dim_)
  )
  result_node$op_type <- "transpose"
  wrap_gretaR_array(result_node)
}

#' Matrix multiplication for gretaR_arrays
#' @param x A gretaR_array.
#' @param y A gretaR_array.
#' @return A gretaR_array.
#' @noRd
gretaR_matmul <- function(x, y) {
  if (!inherits(x, "gretaR_array")) x <- as_data(x)
  if (!inherits(y, "gretaR_array")) y <- as_data(y)
  node1 <- get_node(x)
  node2 <- get_node(y)
  out_dim <- c(node1$dim_[1], node2$dim_[2])

  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) sparse_matmul(pvals[[1]], pvals[[2]]),
    parents = c(node1$id, node2$id),
    dim = out_dim
  )
  result_node$op_type <- "matmul"
  wrap_gretaR_array(result_node)
}

# Register %*% for gretaR_array
#' @noRd
`%*%.gretaR_array` <- function(x, y) gretaR_matmul(x, y)

# =============================================================================
# Reduction operations (sum, mean, etc.)
# =============================================================================

#' Sum of a gretaR_array
#' @param x A gretaR_array.
#' @param ... Ignored.
#' @param na.rm Ignored (no NAs in torch).
#' @return A scalar gretaR_array.
#' @noRd
sum.gretaR_array <- function(x, ..., na.rm = FALSE) {
  node <- get_node(x)
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) torch_sum(pvals[[1]])$unsqueeze(1)$unsqueeze(2),
    parents = node$id,
    dim = c(1L, 1L)
  )
  result_node$op_type <- "sum"
  wrap_gretaR_array(result_node)
}

#' Mean of a gretaR_array
#' @noRd
mean.gretaR_array <- function(x, ...) {
  node <- get_node(x)
  result_node <- GretaRArray$new(
    node_type = "operation",
    operation = function(pvals) torch_mean(pvals[[1]])$unsqueeze(1)$unsqueeze(2),
    parents = node$id,
    dim = c(1L, 1L)
  )
  result_node$op_type <- "mean"
  wrap_gretaR_array(result_node)
}

# =============================================================================
# Utility: dimension broadcasting
# =============================================================================

#' @noRd
broadcast_dims <- function(dim1, dim2) {
  # Simple broadcasting: match R/torch rules
  len <- max(length(dim1), length(dim2))
  d1 <- rev(c(rep(1L, len - length(dim1)), dim1))
  d2 <- rev(c(rep(1L, len - length(dim2)), dim2))
  out <- integer(len)
  for (i in seq_len(len)) {
    if (d1[i] == d2[i]) {
      out[i] <- d1[i]
    } else if (d1[i] == 1L) {
      out[i] <- d2[i]
    } else if (d2[i] == 1L) {
      out[i] <- d1[i]
    } else {
      cli_abort("Incompatible dimensions for broadcasting: {dim1} vs {dim2}")
    }
  }
  rev(out)
}

# Null-coalescing operator (if not already available)
`%||%` <- function(x, y) if (is.null(x)) y else x
