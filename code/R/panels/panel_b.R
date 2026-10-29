# Figure 2b; builder returns a plot and does not write files.
build_fig2b <- function(panel_data) {
  data <- panel_data
  stopifnot(nrow(data) == 853L, all(data$load_decile %in% 1:10))
  data$load_decile_f <- factor(data$load_decile, levels = 1:10, labels = paste0("D", 1:10))

  summary_data <- data |>
    group_by(load_decile) |>
    summarise(
      p10 = unname(quantile(lamp1_pathology_area_um2, 0.10)),
      p90 = unname(quantile(lamp1_pathology_area_um2, 0.90)),
      p90_p10 = p90 / p10,
      .groups = "drop"
    )

  decile_palette <- setNames(
    grDevices::colorRampPalette(c("#E88C83", "#80C7D4", "#5CB8A5", "#7E8DAE"))(10),
    paste0("D", 1:10)
  )

  make_half_violin <- function(group_data) {
    decile <- unique(group_data$load_decile)
    density <- stats::density(group_data$pathology_log10, n = 384, cut = 0, adjust = 0.95)
    baseline <- decile + 0.12
    curve <- data.frame(
      load_decile_f = factor(paste0("D", decile), levels = paste0("D", 1:10)),
      y_log10 = density$x,
      baseline_x = baseline,
      outer_x = baseline + 0.36 * density$y / max(density$y)
    )
    polygon <- bind_rows(
      transform(curve[1, ], outer_x = baseline),
      curve,
      transform(curve[nrow(curve), ], outer_x = baseline)
    )
    list(curve = curve, polygon = polygon)
  }

  parts <- lapply(split(data, data$load_decile), make_half_violin)
  violin_curve <- bind_rows(lapply(parts, `[[`, "curve"))
  violin_polygon <- bind_rows(lapply(parts, `[[`, "polygon"))
  y_breaks <- c(20, 50, 100, 200, 500, 1000, 2000)

  p <- ggplot() +
    geom_point(
      data = data,
      aes(dot_x, pathology_log10, colour = load_decile_f),
      size = 0.82, alpha = 0.78, stroke = 0
    ) +
    geom_polygon(
      data = violin_polygon,
      aes(outer_x, y_log10, group = load_decile_f, fill = load_decile_f),
      colour = NA, alpha = 0.88
    ) +
    geom_line(
      data = violin_curve,
      aes(outer_x, y_log10, group = load_decile_f, colour = load_decile_f),
      linewidth = 0.28
    ) +
    geom_boxplot(
      data = data,
      aes(load_decile, pathology_log10, group = load_decile),
      width = 0.13, fill = "white", colour = "#4A4A4A",
      linewidth = 0.40, outlier.shape = NA
    ) +
    scale_colour_manual(values = decile_palette, guide = "none") +
    scale_fill_manual(values = decile_palette, guide = "none") +
    scale_x_continuous(
      breaks = 1:10, labels = paste0("D", 1:10),
      limits = c(0.55, 10.62), expand = expansion(add = 0)
    ) +
    scale_y_continuous(
      breaks = log10(y_breaks), labels = label_number(big.mark = ",", accuracy = 1)(y_breaks),
      limits = log10(c(10, 2600)), expand = expansion(add = 0)
    ) +
    labs(
      title = "LAMP1-related pathology remains heterogeneous within Aβ plaque-area deciles",
      subtitle = sprintf(
        "n = %d plaques; P90/P10 = %.2f–%.2f× across deciles",
        nrow(data), min(summary_data$p90_p10), max(summary_data$p90_p10)
      ),
      x = "Aβ plaque area (decile)",
      y = "LAMP1-related pathology area (µm², log scale)"
    ) +
    theme_classic(base_size = 9, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.48, colour = "#272727"),
      axis.ticks = element_line(linewidth = 0.42, colour = "#272727"),
      axis.text = element_text(size = 7.6, colour = "#272727"),
      axis.title = element_text(size = 8.8, colour = "#272727"),
      panel.grid.major.y = element_line(colour = "#E2E2E2", linewidth = 0.34, linetype = "22"),
      panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
      plot.title = element_text(size = 10.2, face = "bold"),
      plot.subtitle = element_text(size = 8, colour = "#555555", margin = margin(b = 5)),
      plot.margin = margin(6, 7, 6, 6)
    )

    p
}
