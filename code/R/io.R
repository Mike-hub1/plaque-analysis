read_panel_data <- function(panel_letter, data_dir) {
  path <- file.path(data_dir, paste0("fig2", panel_letter, "_data.csv"))
  if (!file.exists(path)) stop("Missing panel data: ", path, call. = FALSE)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

read_fold_data <- function(data_dir) {
  path <- file.path(data_dir, "fig2g_fold_data.csv")
  if (!file.exists(path)) stop("Missing fold data: ", path, call. = FALSE)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

write_csv_utf8 <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(x, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
}
