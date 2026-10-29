build_fig2h_main <- function(prepared, compact = FALSE) {
  raw_data <- prepared$raw
  summary_data <- prepared$summary
  scenario_meta <- prepared$meta

  point_size <- if (compact) 0.43 else 0.78
  point_alpha <- if (compact) 0.21 else 0.24
  outer_linewidth <- if (compact) 0.42 else 0.78
  inner_linewidth <- if (compact) 1.35 else 3.20
  cap_height <- if (compact) 0.080 else 0.115
  median_size <- if (compact) 1.72 else 3.35

  row_guides <- data.frame(y = 1:6)
  x_guides <- data.frame(x = c(0.05, 0.10, 0.15))

  p_main <- ggplot() +
    annotate(
      "rect",
      xmin = -Inf,
      xmax = Inf,
      ymin = 2.5,
      ymax = 6.48,
      fill = "#DDECF4",
      alpha = 0.56
    ) +
    annotate(
      "rect",
      xmin = -Inf,
      xmax = Inf,
      ymin = 0.52,
      ymax = 2.5,
      fill = "#F8E5DD",
      alpha = 0.64
    ) +
    geom_hline(
      data = row_guides,
      aes(yintercept = y),
      linewidth = if (compact) 0.18 else 0.25,
      colour = "#E7EBEE"
    ) +
    geom_vline(
      data = x_guides,
      aes(xintercept = x),
      linewidth = if (compact) 0.18 else 0.25,
      colour = "#E1E6EA"
    ) +
    geom_hline(
      yintercept = 2.5,
      linewidth = if (compact) 0.30 else 0.42,
      colour = "#BFC7CD"
    ) +
    geom_vline(
      xintercept = 0,
      linewidth = if (compact) 0.46 else 0.72,
      colour = "#303030"
    ) +
    geom_point(
      data = raw_data,
      aes(
        x = delta_r2_region,
        y = y_jitter,
        colour = brainstem_status
      ),
      size = point_size,
      alpha = point_alpha,
      stroke = 0,
      show.legend = FALSE
    ) +
    geom_segment(
      data = summary_data,
      aes(
        x = q025,
        xend = q975,
        y = y,
        yend = y,
        colour = brainstem_status
      ),
      linewidth = outer_linewidth,
      lineend = "round",
      show.legend = FALSE
    ) +
    geom_segment(
      data = summary_data,
      aes(
        x = q025,
        xend = q025,
        y = y - cap_height,
        yend = y + cap_height,
        colour = brainstem_status
      ),
      linewidth = outer_linewidth,
      show.legend = FALSE
    ) +
    geom_segment(
      data = summary_data,
      aes(
        x = q975,
        xend = q975,
        y = y - cap_height,
        yend = y + cap_height,
        colour = brainstem_status
      ),
      linewidth = outer_linewidth,
      show.legend = FALSE
    ) +
    geom_segment(
      data = summary_data,
      aes(
        x = q25,
        xend = q75,
        y = y,
        yend = y,
        colour = brainstem_status
      ),
      linewidth = inner_linewidth,
      alpha = 0.48,
      lineend = "round",
      show.legend = FALSE
    ) +
    geom_point(
      data = summary_data,
      aes(
        x = median_delta_r2,
        y = y,
        colour = brainstem_status
      ),
      shape = 21,
      fill = "white",
      size = median_size,
      stroke = if (compact) 0.55 else 0.90,
      show.legend = FALSE
    ) +
    scale_colour_manual(
      values = fig2h_status_colours,
      breaks = c("Brainstem retained", "Brainstem absent"),
      labels = c(
        "BS retained: 100/100 \u0394R\u00b2 > 0 per scenario",
        "BS absent: 100/100 \u0394R\u00b2 < 0 per scenario"
      ),
      drop = FALSE
    ) +
    scale_x_continuous(
      limits = c(-0.009, 0.170),
      breaks = c(0, 0.05, 0.10, 0.15),
      labels = if (compact) {
        c("0", ".05", ".10", ".15")
      } else {
        c("0", "0.05", "0.10", "0.15")
      },
      expand = expansion(mult = 0)
    ) +
    scale_y_continuous(
      limits = c(0.52, 6.48),
      breaks = scenario_meta$y,
      labels = if (compact) scenario_meta$label_compact else scenario_meta$label_full,
      expand = expansion(mult = 0)
    ) +
    coord_cartesian(clip = "off") +
    labs(
      title = if (compact) {
        "Recorded-region gain collapses\nafter brainstem exclusion"
      } else {
        "Recorded-region gain collapses after brainstem exclusion"
      },
      subtitle = if (compact) {
        "100 matched splits; median and 2.5\u201397.5% range"
      } else {
        paste0(
          "\u0394R\u00b2 across 100 matched repeated splits\n",
          "Points: splits; circle: median; thick/thin bars: IQR/2.5\u201397.5% repeated-split interval"
        )
      },
      x = "\u0394R\u00b2 = R\u00b2(A\u03b2 + region) \u2212 R\u00b2(A\u03b2 only)",
      y = NULL
    ) +
    theme_classic(
      base_family = "Arial",
      base_size = if (compact) 7.0 else 9.0
    ) +
    theme(
      axis.line.x = element_line(
        linewidth = if (compact) 0.32 else 0.48,
        colour = "#303030"
      ),
      axis.line.y = element_blank(),
      axis.ticks.x = element_line(
        linewidth = if (compact) 0.30 else 0.42,
        colour = "#303030"
      ),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(
        size = if (compact) 4.8 else 7.0,
        colour = "#303030"
      ),
      axis.text.y = element_text(
        size = if (compact) 5.0 else 7.3,
        colour = "#303030",
        hjust = 1,
        margin = margin(r = if (compact) 2.0 else 5.0)
      ),
      axis.title.x = element_text(
        size = if (compact) 6.0 else 8.2,
        face = if (compact) "plain" else "bold",
        colour = "#252525",
        margin = margin(t = if (compact) 2.5 else 5.0)
      ),
      panel.grid = element_blank(),
      plot.title = element_text(
        size = if (compact) 8.8 else 12.0,
        face = "bold",
        colour = "#252525",
        lineheight = 0.96,
        margin = margin(b = if (compact) 2.0 else 3.0)
      ),
      plot.subtitle = element_text(
        size = if (compact) 5.7 else 7.6,
        colour = "#626262",
        lineheight = 1.05,
        margin = margin(b = if (compact) 3.0 else 6.0)
      ),
      plot.title.position = "plot",
      legend.position = "none",
      legend.justification = "left",
      legend.direction = "horizontal",
      legend.title = element_blank(),
      legend.text = element_text(
        size = 6.7,
        colour = "#404040",
        margin = margin(r = 8.0)
      ),
      legend.key.width = grid::unit(10, "pt"),
      legend.key.height = grid::unit(8, "pt"),
      legend.spacing.x = grid::unit(3, "pt"),
      legend.margin = margin(-2.0, 0, 1.0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      plot.margin = margin(
        if (compact) 3.0 else 5.0,
        if (compact) 3.0 else 7.0,
        if (compact) 3.0 else 5.0,
        if (compact) 3.0 else 6.0
      )
    )
  p_main
}
