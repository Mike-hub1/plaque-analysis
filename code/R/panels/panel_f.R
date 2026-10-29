# Figure 2f; builder returns a plot and does not write files.
build_fig2f <- function(panel_data) {
  data <- panel_data
  stopifnot(nrow(data) == 853L)
  data$region_factor <- factor(data$region_raw, levels = region_levels)
  region_n <- table(data$region_factor)
  axis_labels <- setNames(
    paste0(region_labels[region_levels], "\n(n=", as.integer(region_n[region_levels]), ")"),
    region_levels
  )
  box_data <- data |>
    group_by(region_raw) |>
    filter(n() >= 5L) |>
    ungroup()

  p <- ggplot(data, aes(region_factor, m1_full_residual_log1p, colour = region_factor)) +
    geom_hline(yintercept = 0, colour = "#767676", linetype = "22", linewidth = 0.45) +
    geom_jitter(
      size = 0.65, alpha = 0.18, stroke = 0,
      position = position_jitter(width = 0.18, height = 0, seed = 20260808)
    ) +
    geom_boxplot(
      data = box_data, width = 0.52, outlier.shape = NA,
      fill = "white", linewidth = 0.45, alpha = 0.9
    ) +
    scale_colour_manual(values = region_palette, drop = FALSE) +
    scale_x_discrete(labels = axis_labels) +
    labs(
      title = "Load-adjusted residuals retain\nregional heterogeneity",
      x = "Recorded brain region",
      y = "GAM residual (log1p scale)"
    ) +
    theme_fig2() +
    theme(legend.position = "none", axis.text.x = element_text(size = 5.7, lineheight = 0.9))

    p
}
