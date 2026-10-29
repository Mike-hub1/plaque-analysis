export_panel <- function(plot, panel_letter, width_mm, height_mm, output_dir, dpi = 600) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  base_path <- file.path(output_dir, paste0("fig2", panel_letter))

  svglite::svglite(
    paste0(base_path, ".svg"),
    width = width_mm / 25.4,
    height = height_mm / 25.4,
    bg = "white"
  )
  print(plot)
  grDevices::dev.off()

  # Rasterize the final SVG instead of drawing the ggplot a second time on a
  # different graphics device.  svglite and ragg can use different font
  # metrics, which previously caused some PNG panels to be clipped or laid out
  # differently from their SVG counterparts.
  svg_path <- paste0(base_path, ".svg")
  png_path <- paste0(base_path, ".png")
  expected_width_px <- round(width_mm / 25.4 * dpi)
  expected_height_px <- round(height_mm / 25.4 * dpi)

  raster <- magick::image_read(
    svg_path,
    density = paste0(dpi, "x", dpi)
  )
  raster_info <- magick::image_info(raster)
  if (raster_info$width != expected_width_px ||
      raster_info$height != expected_height_px) {
    raster <- magick::image_resize(
      raster,
      sprintf("%dx%d!", expected_width_px, expected_height_px)
    )
  }
  raster <- magick::image_background(raster, "white", flatten = TRUE)

  temporary_png <- file.path(
    dirname(png_path),
    paste0(".", basename(png_path), ".new")
  )
  magick::image_write(
    raster,
    path = temporary_png,
    format = "png",
    depth = 8,
    density = paste0(dpi, "x", dpi)
  )

  if (file.exists(png_path)) unlink(png_path)
  if (!file.rename(temporary_png, png_path)) {
    unlink(temporary_png)
    stop("Could not install synchronized PNG for Fig. 2", panel_letter)
  }

  invisible(c(svg = paste0(base_path, ".svg"), png = paste0(base_path, ".png")))
}
