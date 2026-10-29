#!/usr/bin/env Rscript
# Render exported final SVGs without modifying vector source files.
.renderer_source <- if (sys.nframe() > 0L) sys.frame(1)$ofile else NULL
if (is.null(.renderer_source)) {
  .renderer_source <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
}
.renderer_code_root <- dirname(dirname(normalizePath(gsub("~\\+~", " ", .renderer_source), mustWork = TRUE)))

render_final <- function(directory, dpi = 120L) {
  directory <- normalizePath(directory, mustWork = TRUE)
  protected <- c(.renderer_code_root, file.path(dirname(.renderer_code_root), "figures"))
  if (any(directory == protected | startsWith(directory, paste0(protected, "/")))) {
    stop("Render exported SVGs in an output directory separate from code and final artwork.")
  }
  if (length(dpi) != 1L || !is.finite(dpi) || dpi <= 0) stop("DPI must be positive.")
  if (!requireNamespace("rsvg", quietly = TRUE)) stop("Install R package rsvg for PNG/PDF rendering.")
  if (!requireNamespace("xml2", quietly = TRUE)) stop("Install R package xml2 for SVG dimensions.")
  files <- list.files(directory, pattern = "^Figure6([a-f])?[.]svg$", full.names = TRUE)
  if (length(files) != 7L) stop("Expected final A4 SVG and six panel SVGs.")
  for (file in files) {
    root <- xml2::xml_root(xml2::read_xml(file))
    width <- as.numeric(sub("mm$", "", xml2::xml_attr(root, "width")))
    height <- as.numeric(sub("mm$", "", xml2::xml_attr(root, "height")))
    if (!all(is.finite(c(width, height))) || min(width, height) <= 0) stop("SVG dimensions must be positive millimetres.")
    stem <- sub("\\.svg$", "", file)
    rsvg::rsvg_png(file, paste0(stem, ".png"), width = round(width/25.4*dpi), height = round(height/25.4*dpi))
    rsvg::rsvg_pdf(file, paste0(stem, ".pdf"))
  }
  invisible(files)
}
args <- commandArgs(trailingOnly = TRUE)
if (sys.nframe() == 0L) {
  if (length(args) != 2L) stop("Usage: Rscript render_artwork.R FINAL_DIRECTORY DPI")
  dpi <- as.integer(args[[2]])
  if (is.na(dpi) || dpi <= 0L) stop("DPI must be positive.")
  render_final(args[[1]], dpi)
}
