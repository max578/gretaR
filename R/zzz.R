#' @noRd
.onLoad <- function(libname, pkgname) {
  # Reset global state on package load
  reset_gretaR_env()
}

#' @noRd
.onAttach <- function(libname, pkgname) {
  if (!torch::torch_is_installed()) {
    packageStartupMessage(
      "gretaR requires the torch backend. Run torch::install_torch() to set up."
    )
  }
}

#' @title Reset the gretaR Global Environment
#'
#' @description Clear all tracked nodes, distributions, and reset the node
#'   counter. Called automatically on package load; call manually to start a
#'   fresh model definition.
#'
#' @return Invisible \code{NULL}.
#' @export
#' @examples
#' \dontrun{
#' reset_gretaR_env()
#' }
reset_gretaR_env <- function() {
  .gretaR_env$dag <- new.env(parent = emptyenv())
  .gretaR_env$dag$nodes <- list()
  .gretaR_env$distributions <- list()
  .gretaR_env$node_counter <- 0L
}

#' Generate a unique node ID
#' @noRd
new_node_id <- function() {
  .gretaR_env$node_counter <- .gretaR_env$node_counter + 1L
  paste0("node_", .gretaR_env$node_counter)
}

#' Register a node in the global DAG
#' @noRd
register_node <- function(node) {
  .gretaR_env$dag$nodes[[node$id]] <- node
}

#' Register a distribution assignment (likelihood)
#' @noRd
register_distribution <- function(data_node_id, dist_node) {
  .gretaR_env$distributions[[data_node_id]] <- dist_node
}
