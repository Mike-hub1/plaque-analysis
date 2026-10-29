# Figure 2d; builder returns a plot and does not write files.
build_fig2d <- function(panel_data) {
  data <- panel_data
  stopifnot(nrow(data) == 853L)
  data$region_factor <- factor(data$region_raw, levels = region_levels)
  r2 <- pooled_r2(data$observed_log1p, data$predicted_log1p)

  p <- ggplot(data, aes(observed_lamp1_um2, predicted_lamp1_um2, colour = region_factor)) +
    geom_abline(slope = 1, intercept = 0, colour = "#767676", linetype = "22", linewidth = 0.45) +
    geom_point(size = 0.85, alpha = 0.48, stroke = 0) +
    scale_colour_manual(values = region_palette, drop = FALSE) +
    scale_x_log10(labels = label_number(accuracy = 1, big.mark = ",")) +
    scale_y_log10(labels = label_number(accuracy = 1, big.mark = ",")) +
    labs(
      title = "Aβ-only predictions leave\nsubstantial variation",
      subtitle = sprintf(
        "10-fold CV R² = %.2f; unexplained fraction = %.0f%%",
        r2, 100 * (1 - r2)
      ),
      x = "Observed LAMP1-related area (µm²)",
      y = "Cross-validated prediction (µm²)"
    ) +
    theme_fig2() +
    theme(legend.position = "none")

    p
}
