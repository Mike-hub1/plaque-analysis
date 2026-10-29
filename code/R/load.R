# Load definitions without running analysis or exporting figures.
load_structure <- function(root, envir = parent.frame()) {
  packages <- c("dplyr", "tidyr", "ggplot2", "scales", "svglite", "magick",
                "patchwork", "cowplot", "ggforce", "mgcv")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "),
                            ". Run Rscript ", shQuote(file.path(root, "R", "install.R")), ".", call. = FALSE)
  suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(scales)
  })
  files <- c("io.R", "style.R", "export.R",
             list.files(file.path(root, "R", "analysis"), pattern = "[.]R$",
                        full.names = FALSE),
             list.files(file.path(root, "R", "panels"), pattern = "[.]R$",
                        full.names = FALSE))
  for (name in files) {
    candidates <- file.path(root, "R", c("", "analysis", "panels"), name)
    source(candidates[file.exists(candidates)][[1]], local = envir)
  }
  invisible(envir)
}
