# sparse.R — Sparse matrix support via the Matrix package
#
# Enables efficient handling of large, sparse design matrices (e.g., genomics,
# NLP, spatial models) by converting Matrix::dgCMatrix/dgTMatrix objects to
# torch sparse COO tensors.

#' Convert a sparse Matrix to a torch sparse COO tensor
#'
#' @param x A sparse matrix from the Matrix package (dgCMatrix, dgTMatrix,
#'   or any coercible to TsparseMatrix).
#' @param dtype Torch dtype (default torch_float32).
#' @return A torch sparse COO tensor.
#' @noRd
sparse_to_torch <- function(x, dtype = torch_float32()) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    cli_abort("Package {.pkg Matrix} is required for sparse matrix support.")
  }

  # Convert to triplet (COO) format
  if (!inherits(x, "TsparseMatrix")) {
    x <- methods::as(x, "TsparseMatrix")
  }

  # Build indices matrix (2 x nnz, 1-based for torch)
  indices <- rbind(
    as.integer(x@i + 1L),
    as.integer(x@j + 1L)
  )

  torch_sparse_coo_tensor(
    indices = torch_tensor(indices, dtype = torch_long()),
    values = torch_tensor(x@x, dtype = dtype),
    size = x@Dim
  )
}

#' Wrap sparse observed data as a gretaR_array
#'
#' S3 method for `as_data()` that handles sparse matrices from the
#' Matrix package. The data is stored as a torch sparse COO tensor
#' for memory-efficient matrix operations.
#'
#' @param x A sparse matrix (dgCMatrix, dgTMatrix, or similar).
#' @return A `gretaR_array` representing fixed (observed) sparse data.
#' @noRd
as_data_sparse <- function(x) {
  if (anyNA(x@x)) {
    cli_abort(c(
      "Missing values ({.val NA}) detected in sparse data passed to {.fn as_data}.",
      "i" = "gretaR requires complete data. Preprocess with {.pkg mice}, {.pkg missRanger}, or {.fn tidyr::drop_na}."
    ))
  }

  tensor <- sparse_to_torch(x)
  dims <- dim(x)

  node <- GretaRArray$new(
    node_type = "data",
    value = tensor,
    dim = dims
  )
  # Mark as sparse for dispatch in matmul
  node$is_sparse <- TRUE
  wrap_gretaR_array(node)
}

#' Sparse-aware matrix multiplication
#'
#' Dispatches to `torch_mm()` when either operand is a sparse tensor,
#' falling back to `torch_matmul()` for dense × dense.
#'
#' @param a A torch tensor (sparse or dense).
#' @param b A torch tensor (sparse or dense).
#' @return A torch tensor (result of matrix multiplication).
#' @noRd
sparse_matmul <- function(a, b) {
  a_sparse <- inherits(a, "torch_tensor") && a$is_sparse()
  b_sparse <- inherits(b, "torch_tensor") && b$is_sparse()

  if (a_sparse || b_sparse) {
    # torch_mm handles sparse x dense and dense x sparse
    torch_mm(a, b)
  } else {
    torch_matmul(a, b)
  }
}
