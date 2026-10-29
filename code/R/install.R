#!/usr/bin/env Rscript
# Explicit installation only; analysis commands never change the R installation.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[[1]])
root <- dirname(dirname(normalizePath(gsub("~\\+~", " ", script), mustWork = TRUE)))
requirements <- readLines(file.path(root, "requirement.txt"), warn = FALSE)
entries <- requirements[grepl("^#[[:space:]]*R:[[:space:]]*[[:alnum:].]+==", requirements)]
packages <- sub("==.*$", "", sub("^#[[:space:]]*R:[[:space:]]*", "", entries))
if (!length(packages)) stop("No R dependencies found in requirement.txt.")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
