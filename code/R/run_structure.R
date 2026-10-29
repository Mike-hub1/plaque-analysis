#!/usr/bin/env Rscript
# Portable entry point; paths are resolved relative to this file or the caller.
args_all <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE)[[1]])
root <- dirname(dirname(normalizePath(gsub("~\\+~", " ", script), mustWork = TRUE)))
repository <- dirname(root)
Data <- "your project/dataset"
args <- commandArgs(trailingOnly = TRUE)
value_options <- c("--data-dir", "--output-dir", "--panels", "--dpi", "--python")
args <- unlist(lapply(args, function(arg) {
  option <- sub("=.*$", "", arg)
  if (grepl("=", arg, fixed = TRUE) && option %in% value_options) {
    c(option, substring(arg, nchar(option) + 2L))
  } else arg
}), use.names = FALSE)
usage <- paste(
  "Usage: python code/run.py structure [--data-dir PATH] [--output-dir PATH]",
  "                    [--panels abcdefghi] [--dpi 600] [--python python3]",
  "                    [--layout] [--rebuild-folds]",
  "Set Data, PLAQUE_DATA_DIR, or --data-dir PATH to your dataset directory.",
  "Defaults: your project/dataset, Github/outputs/structure, all panels; --layout adds a reconstructed A4 reference.",
  sep = "\n"
)
if ("--help" %in% args || "-h" %in% args) { cat(usage, "\n"); quit(status = 0L) }
options <- list(data_dir = Sys.getenv("PLAQUE_DATA_DIR", unset = Data),
                output_dir = file.path(repository, "outputs", "structure"),
                panels = "abcdefghi", dpi = 600, python = "python3",
                layout = FALSE, rebuild_folds = FALSE)
i <- 1L
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("--layout", "--no-layout", "--rebuild-folds")) {
    if (arg == "--layout") options$layout <- TRUE
    if (arg == "--no-layout") options$layout <- FALSE
    if (arg == "--rebuild-folds") options$rebuild_folds <- TRUE
  } else if (arg %in% value_options) {
    i <- i + 1L
    if (i > length(args)) stop("Missing value for ", arg)
    options[[gsub("-", "_", substring(arg, 3))]] <- args[[i]]
  } else stop("Unknown argument: ", arg, "\n", usage)
  i <- i + 1L
}
options$dpi <- suppressWarnings(as.numeric(options$dpi))
if (length(options$dpi) != 1L || !is.finite(options$dpi) || options$dpi <= 0) {
  stop("--dpi must be a positive number.")
}
options$panels <- strsplit(options$panels, "", fixed = TRUE)[[1]]
if (!length(options$panels)) stop("--panels cannot be empty.")
if (!nzchar(options$data_dir) || options$data_dir == "your project/dataset") {
  stop("Input path is not configured. Set Data, PLAQUE_DATA_DIR, or --data-dir PATH.", call. = FALSE)
}
if (!dir.exists(options$data_dir)) stop("Input directory does not exist: ", options$data_dir, call. = FALSE)
options$data_dir <- normalizePath(options$data_dir, mustWork = TRUE)
resolve_destination <- function(path) {
  path <- path.expand(path)
  if (file.exists(path) || dir.exists(path)) return(normalizePath(path, mustWork = TRUE))
  file.path(resolve_destination(dirname(path)), basename(path))
}
if (!nzchar(options$output_dir)) stop("--output-dir cannot be empty.", call. = FALSE)
options$output_dir <- resolve_destination(options$output_dir)
protected <- c(options$data_dir, root, file.path(repository, "figures"))
overlaps <- vapply(protected, function(path) {
  path <- resolve_destination(path)
  options$output_dir == path ||
    startsWith(options$output_dir, paste0(path, "/")) ||
    startsWith(path, paste0(options$output_dir, "/"))
}, logical(1))
if (any(overlaps)) {
  stop("Output must be separate from input data, source code, and final artwork.", call. = FALSE)
}
if (file.exists(options$output_dir) && !dir.exists(options$output_dir)) {
  stop("Output path is an existing file: ", options$output_dir, call. = FALSE)
}
dir.create(options$output_dir, recursive = TRUE, showWarnings = FALSE)
options$output_dir <- normalizePath(options$output_dir, mustWork = TRUE)
source(file.path(root, "R", "load.R"))
load_structure(root)
source(file.path(root, "R", "pipeline.R"))
do.call(run_structure, c(list(root = root), options))
message("Figure 2 written to ", options$output_dir)
