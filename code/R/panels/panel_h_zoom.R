build_fig2h_zoom <- function(raw_data, summary_data) {
  zoom_raw <- raw_data |>
    filter(brainstem_status == "Brainstem absent") |>
    mutate(
      zoom_y = if_else(scenario == "Exclude bs", 2, 1),
      zoom_y_jitter = zoom_y + (y_jitter - y) * 0.82
    )
  zoom_summary <- summary_data |>
    filter(brainstem_status == "Brainstem absent") |>
    mutate(
      zoom_y = if_else(scenario == "Exclude bs", 2, 1)
    )

  p_zoom <- ggplot() +
    annotate(
      "rect",
      xmin = -Inf,
      xmax = Inf,
      ymin = -Inf,
      ymax = Inf,
      fill = "#FFF9F6",
      alpha = 1
    ) +
    geom_hline(
      yintercept = c(1, 2),
      linewidth = 0.20,
      colour = "#E8E1DE"
    ) +
    geom_vline(
      xintercept = c(-0.004, -0.002),
      linewidth = 0.20,
      colour = "#E3E8EB"
    ) +
    geom_vline(
      xintercept = 0,
      linewidth = 0.52,
      colour = "#303030"
    ) +
    geom_point(
      data = zoom_raw,
      aes(x = delta_r2_region, y = zoom_y_jitter),
      colour = unname(fig2h_status_colours[["Brainstem absent"]]),
      size = 0.60,
      alpha = 0.26,
      stroke = 0
    ) +
    geom_segment(
      data = zoom_summary,
      aes(x = q025, xend = q975, y = zoom_y, yend = zoom_y),
      colour = unname(fig2h_status_colours[["Brainstem absent"]]),
      linewidth = 0.68,
      lineend = "round"
    ) +
    geom_segment(
      data = zoom_summary,
      aes(x = q25, xend = q75, y = zoom_y, yend = zoom_y),
      colour = unname(fig2h_status_colours[["Brainstem absent"]]),
      linewidth = 2.25,
      alpha = 0.50,
      lineend = "round"
    ) +
    geom_point(
      data = zoom_summary,
      aes(x = median_delta_r2, y = zoom_y),
      shape = 21,
      fill = "white",
      colour = unname(fig2h_status_colours[["Brainstem absent"]]),
      size = 2.35,
      stroke = 0.72
    ) +
    scale_x_continuous(
      limits = c(-0.0062, 0.00045),
      breaks = c(-0.006, -0.004, -0.002, 0),
      labels = c("\u2212.006", "\u2212.004", "\u2212.002", "0"),
      expand = expansion(mult = 0)
    ) +
    scale_y_continuous(
      limits = c(0.55, 2.45),
      breaks = c(1, 2),
      # Use the explicit region composition rather than the ambiguous
      # shorthand "Major only". This row is the n = 832 scenario.
      labels = c("CTX + HPF + STR", "\u2212 BS"),
      expand = expansion(mult = 0)
    ) +
    coord_cartesian(clip = "off") +
    labs(
      title = "Near-zero \u0394R\u00b2",
      subtitle = "Expanded raw x-axis",
      x = "\u0394R\u00b2",
      y = NULL
    ) +
    theme_classic(base_family = "Arial", base_size = 6.3) +
    theme(
      plot.background = element_rect(fill = NA, colour = NA),
      panel.background = element_rect(fill = NA, colour = NA),
      axis.line.x = element_line(colour = "#303030", linewidth = 0.34),
      axis.line.y = element_blank(),
      axis.ticks.x = element_line(colour = "#303030", linewidth = 0.30),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = 5.0, colour = "#303030"),
      axis.text.y = element_text(size = 5.0, colour = "#303030", margin = margin(r = 2.2)),
      axis.title.x = element_text(size = 5.4, colour = "#303030", margin = margin(t = 1.8)),
      plot.title = element_text(
        size = 6.2,
        face = "bold",
        hjust = 0,
        colour = "#303030",
        margin = margin(b = 0.8)
      ),
      plot.subtitle = element_text(
        size = 5.0,
        hjust = 0,
        colour = "#6A6A6A",
        margin = margin(b = 2.6)
      ),
      plot.margin = margin(2.8, 4.0, 2.2, 4.0)
    )
  p_zoom
}
