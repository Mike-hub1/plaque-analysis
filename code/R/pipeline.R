panel_dimensions <- data.frame(
  panel = letters[1:9],
  width_mm = c(61, 183, 183, 61, 112, 61, 82, 158, 112),
  height_mm = c(58, 95, 108, 58, 100, 58, 72, 112, 72)
)

run_structure <- function(root, data_dir, output_dir, panels = letters[1:9],
                     dpi = 600, rebuild_folds = FALSE, layout = FALSE,
                     python = "python3") {
  if (any(!panels %in% letters[1:9]) || anyDuplicated(panels)) {
    stop("--panels must contain unique letters from a to i.", call. = FALSE)
  }
  if (layout && !all(letters[1:9] %in% panels)) {
    stop("Final layout requires --panels abcdefghi; use --no-layout for a subset.")
  }
  fold_data <- NULL
  if (rebuild_folds) {
    fold_data <- rebuild_fig2g_folds(read_panel_data("d", data_dir),
                                   read_panel_data("g", data_dir))
    write_csv_utf8(fold_data, file.path(output_dir, "data", "fig2g_fold_data.csv"))
  }
  for (letter in panels) {
    message("Rendering Figure 2", letter)
    data <- read_panel_data(letter, data_dir)
    plot <- if (letter == "g") {
      if (is.null(fold_data)) fold_data <- read_fold_data(data_dir)
      build_fig2g(data, fold_data)
    } else {
      get(paste0("build_fig2", letter), mode = "function")(data)
    }
    size <- panel_dimensions[panel_dimensions$panel == letter, ]
    export_panel(plot, letter, size$width_mm, size$height_mm,
                 output_dir = file.path(output_dir, "panels"), dpi = dpi)
  }
  if (layout) {
    executable <- Sys.which(python)
    if (!nzchar(executable)) stop("Python not found: ", python)
    svg_path <- file.path(output_dir, "Figure2_reconstructed_A4.svg")
    status <- system2(executable, c(shQuote(file.path(root, "run.py")), "assemble-structure",
                     "--panel-dir", shQuote(file.path(output_dir, "panels")),
                     "--output", shQuote(svg_path)))
    if (status != 0L) stop("Figure 2 A4 composition failed.")
    raster <- magick::image_read(svg_path, density = paste0(dpi, "x", dpi))
    magick::image_write(raster, file.path(output_dir, "Figure2_reconstructed_A4.png"), format = "png")
  }
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
  invisible(output_dir)
}
