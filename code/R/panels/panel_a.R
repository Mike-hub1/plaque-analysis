# Figure 2a; builder returns a plot and does not write files.
build_fig2a <- function(panel_data) {
  source_data <- panel_data
  plaque_data <- source_data[source_data$record_type == "plaque", , drop = FALSE]
  curve_data <- source_data[source_data$record_type == "gam_curve", , drop = FALSE]
  stopifnot(nrow(plaque_data) == 853L, nrow(curve_data) > 2L)

  plaque_data$region_factor <- factor(plaque_data$region_raw, levels = region_levels)
  correlation <- stratified_bootstrap_correlation(
    plaque_data,
    plaque_data$abeta_area_um2,
    plaque_data$lamp1_pathology_area_um2
  )

  p <- ggplot(plaque_data, aes(abeta_area_um2, lamp1_pathology_area_um2)) +
    geom_ribbon(
      data = curve_data,
      aes(x = abeta_area_um2, ymin = ci_low_lamp1_um2, ymax = ci_high_lamp1_um2),
      inherit.aes = FALSE, fill = "#9DB7D5", alpha = 0.32
    ) +
    geom_line(
      data = curve_data,
      aes(x = abeta_area_um2, y = predicted_lamp1_um2),
      inherit.aes = FALSE, colour = "#0F4D92", linewidth = 0.8
    ) +
    geom_point(aes(colour = region_factor), size = 0.85, alpha = 0.48, stroke = 0) +
    scale_colour_manual(
      values = region_palette, breaks = region_levels,
      labels = region_labels[region_levels], drop = FALSE
    ) +
    scale_x_log10(labels = label_number(accuracy = 1, big.mark = ",")) +
    scale_y_log10(labels = label_number(accuracy = 1, big.mark = ",")) +
    labs(
      title = "Aβ load is associated with\nLAMP1-related pathology",
      subtitle = sprintf(
        "Spearman ρ = %.2f (95%% CI %.2f–%.2f); n = %d",
        correlation[["estimate"]], correlation[["low"]], correlation[["high"]],
        nrow(plaque_data)
      ),
      x = "Aβ plaque area (µm²)",
      y = "LAMP1-related pathology area (µm²)",
      colour = NULL
    ) +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(alpha = 1, size = 1.9))) +
    theme_fig2() +
    theme(legend.position = "top", legend.title = element_blank())

    p
}
