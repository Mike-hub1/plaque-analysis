build_fig2g <- function(pooled_data, fold_data) {
  prepared <- prepare_fig2g(pooled_data, fold_data)
  fold_data <- prepared$fold_data
  summary_data <- prepared$summary_data
  delta_summary <- prepared$delta_summary
  bracket_path <- prepared$bracket_path
  cloud_colours <- prepared$cloud_colours
  center_colours <- prepared$center_colours
  p <- ggplot() +
    geom_hline(
      yintercept = 0, colour = "#555555",
      linewidth = 0.30, linetype = "dashed"
    ) +
    geom_point(
      data = fold_data,
      aes(x_plot, fold_r2, colour = model),
      size = 0.68, alpha = 0.46, stroke = 0,
      show.legend = FALSE
    ) +
    geom_errorbar(
      data = summary_data,
      aes(x = x, ymin = fold_q025, ymax = fold_q975),
      width = 0.075, linewidth = 0.44, colour = "#111111"
    ) +
    geom_point(
      data = summary_data,
      aes(x, center, fill = model),
      shape = 21, size = 3.25, stroke = 0.64,
      colour = "#111111", show.legend = FALSE
    ) +
    geom_text(
      data = summary_data,
      aes(x, label_y, label = center_label),
      family = "Arial", fontface = "bold", size = 3.35,
      colour = "#111111",
      show.legend = FALSE
    ) +
    geom_path(
      data = bracket_path,
      aes(x = x, y = y, group = comparison),
      inherit.aes = FALSE, linewidth = 0.34,
      colour = "#555555", lineend = "round", linejoin = "round"
    ) +
    geom_text(
      data = delta_summary,
      aes(x = (x1 + x2) / 2, y = delta_y, label = delta_label),
      inherit.aes = FALSE, family = "Arial", fontface = "bold",
      size = 3.15, colour = "#111111"
    ) +
    geom_text(
      data = delta_summary,
      aes(x = (x1 + x2) / 2, y = interval_y, label = interval_label),
      inherit.aes = FALSE, family = "Arial",
      size = 2.95, colour = "#111111"
    ) +
    scale_colour_manual(values = cloud_colours) +
    scale_fill_manual(values = center_colours) +
    scale_x_continuous(
      breaks = 1:3,
      labels = c(
        "Mean only\n(intercept)",
        "Aβ load\nonly",
        "Aβ load\n+ brain region"
      ),
      limits = c(0.48, 3.42),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_continuous(
      breaks = seq(-0.2, 0.8, by = 0.2),
      labels = sprintf("%.1f", seq(-0.2, 0.8, by = 0.2)),
      expand = expansion(mult = c(0, 0))
    ) +
    # Reserve a narrow, unticked band above 0.8 so the comparison annotations
    # remain inside the data panel and cannot collide with the new title.
    coord_cartesian(ylim = c(-0.2, 0.90), clip = "off") +
    labs(
      title = "Aβ load and brain region increase cross-validated R²",
      x = NULL,
      y = "Cross-validated R²"
    ) +
    theme_classic(base_size = 10.0, base_family = "Arial") +
    theme(
      legend.position = "none",
      text = element_text(colour = "#111111"),
      plot.title.position = "plot",
      plot.title = element_text(
        size = 8.8, face = "bold", colour = "#111111",
        lineheight = 0.96, hjust = 0, margin = margin(b = 3.0)
      ),
      axis.title.y = element_text(
        size = 10.5, colour = "#111111", margin = margin(r = 7)
      ),
      axis.text.y = element_text(size = 8.9, colour = "#111111"),
      axis.text.x = element_text(
        size = 9.0, colour = "#111111", lineheight = 0.98,
        margin = margin(t = 5)
      ),
      axis.line = element_line(colour = "#111111", linewidth = 0.38),
      axis.ticks = element_line(colour = "#111111", linewidth = 0.34),
      axis.ticks.length = grid::unit(1.7, "mm"),
      plot.margin = margin(t = 5.0, r = 4.0, b = 3.5, l = 4.5)
    )
  
  p
}
