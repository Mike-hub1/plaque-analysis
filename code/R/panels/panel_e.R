# Figure 2e; builder returns a plot and does not write files.
build_fig2e <- function(panel_data) {
  data <- panel_data
  stopifnot(nrow(data) == 100L, sum(data$n) == 853L)
  nonzero <- data |>
    filter(n > 0) |>
    mutate(rank_delta = pathology_decile - load_decile)
  band <- expand.grid(load_decile = 1:10, pathology_decile = 1:10) |>
    filter(abs(load_decile - pathology_decile) <= 1)
  same <- sum(data$n[data$load_decile == data$pathology_decile]) / sum(data$n)
  within_one <- sum(data$n[abs(data$load_decile - data$pathology_decile) <= 1]) / sum(data$n)

  p <- ggplot(data, aes(load_decile, pathology_decile)) +
    geom_tile(fill = "#FBFBFB", colour = "#E5E5E5", linewidth = 0.20) +
    geom_tile(
      data = band, inherit.aes = FALSE,
      aes(load_decile, pathology_decile),
      fill = "#F3F3F4", colour = "#E5E5E5", linewidth = 0.20
    ) +
    geom_abline(slope = 1, intercept = 0, colour = "#6B7280", linetype = "22", linewidth = 0.4) +
    geom_point(
      data = nonzero,
      aes(size = n, fill = rank_delta),
      shape = 21, colour = "#595959", stroke = 0.34, alpha = 0.96
    ) +
    scale_fill_gradient2(
      low = "#75ACD3", mid = "#FCFCFC", high = "#DF86A4",
      midpoint = 0, limits = c(-9, 9), breaks = c(-9, 0, 9),
      labels = c("−9", "0", "+9"), oob = squish,
      name = "Decile difference\n(pathology − Aβ)",
      guide = guide_colourbar(
        order = 1, direction = "horizontal", title.position = "top",
        title.hjust = 0, barwidth = grid::unit(44, "pt"), barheight = grid::unit(4.2, "pt")
      )
    ) +
    scale_size_area(
      max_size = 6.3, breaks = c(5, 20, 60), limits = c(0, max(data$n)),
      name = "Plaque count (n)",
      guide = guide_legend(order = 2, nrow = 1, title.position = "left")
    ) +
    scale_x_continuous(breaks = 1:10, labels = paste0("D", 1:10), expand = expansion(add = 0.75)) +
    scale_y_continuous(breaks = 1:10, labels = paste0("D", 1:10), expand = expansion(add = 0.75)) +
    coord_fixed(clip = "off") +
    labs(
      title = "Aβ plaque area and LAMP1-related\npathology area show partial rank concordance",
      subtitle = sprintf("Exact match %.1f%%; within ±1 decile %.1f%%", 100 * same, 100 * within_one),
      x = "Aβ plaque area (decile)",
      y = "LAMP1-related pathology area (decile)"
    ) +
    theme_fig2() +
    theme(
      legend.position = "top", legend.box = "vertical", legend.box.just = "left",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5.4)
    )

    p
}
