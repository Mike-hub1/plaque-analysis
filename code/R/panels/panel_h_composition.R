compose_fig2h <- function(p_main, p_zoom, summary_data) {
  source_box <- data.frame(
    x = c(-0.0076, 0.0038, 0.0038, -0.0076),
    y = c(0.67, 0.67, 2.33, 2.33),
    group = 1L
  )

  p_main <- p_main +
    ggforce::geom_shape(
      data = source_box,
      aes(x = x, y = y, group = group),
      inherit.aes = FALSE,
      fill = NA,
      colour = "#C96F55",
      linewidth = 0.54,
      radius = grid::unit(1.7, "mm"),
      show.legend = FALSE
    ) +
    geom_text(
      data = summary_data,
      aes(
        x = main_label_x,
        y = y,
        label = median_label,
        colour = brainstem_status,
        hjust = main_label_hjust
      ),
      family = "Arial",
      fontface = "bold",
      size = 2.55,
      show.legend = FALSE
    ) +
    labs(title = NULL, subtitle = NULL)

  legend_data <- data.frame(
    brainstem_status = factor(
      c("Brainstem retained", "Brainstem absent"),
      levels = c("Brainstem retained", "Brainstem absent")
    ),
    x = c(0, 0.61),
    y = 1,
    label = c("BS retained", "BS absent")
  )

  p_status_key <- ggplot(
    legend_data,
    aes(y = y, colour = brainstem_status)
  ) +
    geom_point(
      aes(x = x),
      shape = 21,
      fill = "white",
      size = 2.70,
      stroke = 0.75,
      show.legend = FALSE
    ) +
    geom_text(
      aes(x = x + 0.11, label = label),
      hjust = 0,
      family = "Arial",
      size = 2.30,
      colour = "#343434",
      show.legend = FALSE
    ) +
    scale_colour_manual(values = fig2h_status_colours, drop = FALSE) +
    scale_x_continuous(limits = c(-0.05, 1.28), expand = expansion(mult = 0)) +
    scale_y_continuous(limits = c(0.72, 1.28), expand = expansion(mult = 0)) +
    labs(title = "Brainstem status") +
    theme_void(base_family = "Arial") +
    theme(
      plot.title = element_text(
        size = 6.1,
        face = "bold",
        colour = "#303030",
        hjust = 0,
        margin = margin(b = 2.0)
      ),
      plot.margin = margin(1.0, 2.0, 1.0, 2.0)
    )

  # Keep the status key in a narrow side column so the physical 0-0.15 span
  # remains compact. The enlarged near-zero panel is then overlaid across the
  # empty lower-right evidence field, with an explicit source box and arrow.
  side_column <- p_status_key /
    patchwork::plot_spacer() +
    patchwork::plot_layout(heights = c(0.30, 1.88))

  p_base <- p_main + side_column +
    patchwork::plot_layout(
      widths = c(3.18, 1.20)
    ) +
    patchwork::plot_annotation(
      title = "Recorded-region gain collapses after brainstem exclusion",
      subtitle = paste0(
        "\u0394R\u00b2 across 100 matched repeated splits\n",
        "Points: splits; circle: median; thick/thin bars: IQR/2.5\u201397.5% repeated-split interval"
      ),
      theme = theme(
        plot.title = element_text(
          family = "Arial",
          size = 12.0,
          face = "bold",
          colour = "#252525",
          hjust = 0,
          margin = margin(b = 3.0)
        ),
        plot.subtitle = element_text(
          family = "Arial",
          size = 7.6,
          colour = "#626262",
          lineheight = 1.05,
          hjust = 0,
          margin = margin(b = 4.0)
        )
      )
    )

  inset_x <- 0.590
  inset_y <- 0.145
  inset_w <- 0.380
  inset_h <- 0.215

  zoom_arrow <- grid::curveGrob(
    x1 = 0.265,
    y1 = 0.235,
    # Stop immediately before the inset border so the closed arrowhead
    # remains visible instead of being covered by the inset frame.
    x2 = inset_x - 0.006,
    y2 = inset_y + 0.100,
    default.units = "npc",
    curvature = -0.08,
    angle = 85,
    ncp = 6,
    shape = 0.55,
    square = FALSE,
    arrow = grid::arrow(
      type = "closed",
      length = grid::unit(2.4, "mm")
    ),
    gp = grid::gpar(
      col = "#C96F55",
      fill = "#C96F55",
      lwd = 1.35,
      lineend = "round"
    )
  )

  inset_shadow <- grid::roundrectGrob(
    r = grid::unit(2.8, "mm"),
    gp = grid::gpar(fill = "#D8D8D8", col = NA, alpha = 0.32)
  )
  inset_frame <- grid::roundrectGrob(
    r = grid::unit(2.8, "mm"),
    gp = grid::gpar(fill = "#FFF9F6", col = "#C96F55", lwd = 1.45)
  )

  p_full <- cowplot::ggdraw() +
    cowplot::draw_plot(p_base, x = 0, y = 0, width = 1, height = 1) +
    cowplot::draw_grob(zoom_arrow, x = 0, y = 0, width = 1, height = 1) +
    cowplot::draw_grob(
      inset_shadow,
      x = inset_x + 0.005,
      y = inset_y - 0.006,
      width = inset_w,
      height = inset_h
    ) +
    cowplot::draw_grob(
      inset_frame,
      x = inset_x,
      y = inset_y,
      width = inset_w,
      height = inset_h
    ) +
    cowplot::draw_plot(
      p_zoom,
      x = inset_x + 0.010,
      y = inset_y + 0.012,
      width = inset_w - 0.020,
      height = inset_h - 0.024
    )
  p_full
}
