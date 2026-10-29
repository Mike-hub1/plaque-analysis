# Figure 2i; builder returns a plot and does not write files.
build_fig2i <- function(panel_data) {
  data <- panel_data
  stopifnot(
    nrow(data) == 1706L,
    all(data$n_oof_predictions == 100L),
    all(is.finite(data$fold_error)),
    all(data$fold_error >= 1)
  )
  models <- c("Aβ load", "Aβ + region")
  stopifnot(all(table(factor(data$model, levels = models)) == 853L))

  model_colours <- c(
    "Aβ load" = "#1F77B4",
    "Aβ + region" = "#FF7F0E"
  )

  # Exact empirical exceedance curves evaluated on a dense 1–4× display grid.
  # Every ordinate is computed from all 853 real plaque-level errors; the grid
  # only controls line rendering and does not smooth, fit or simulate the data.
  display_thresholds <- seq(1, 4, length.out = 601L)
  curve <- bind_rows(lapply(models, function(model_name) {
    errors <- data$fold_error[data$model == model_name]
    data.frame(
      model = model_name,
      threshold = display_thresholds,
      plaques_ge_pct = vapply(
        display_thresholds,
        function(x) 100 * mean(errors >= x),
        numeric(1)
      )
    )
  }))
  curve$model <- factor(curve$model, levels = models)

  stats <- data |>
    group_by(model) |>
    summarise(
      n = n(),
      median_error = median(fold_error),
      fraction_ge_1_5 = mean(fold_error >= 1.5),
      fraction_ge_2 = mean(fold_error >= 2),
      maximum_error = max(fold_error),
      .groups = "drop"
    )
  stats$model <- factor(stats$model, levels = models)
  m1 <- stats[stats$model == "Aβ load", ]
  m2 <- stats[stats$model == "Aβ + region", ]

  threshold_annotations <- data.frame(
    threshold = c(1.5, 2),
    baseline_pct = c(100 * m1$fraction_ge_1_5, 100 * m1$fraction_ge_2),
    adjusted_pct = c(100 * m2$fraction_ge_1_5, 100 * m2$fraction_ge_2),
    text_x = c(1.52, 2.03),
    text_y = c(47.0, 18.6),
    stringsAsFactors = FALSE
  ) |>
    mutate(label = sprintf(
      "≥%g×\n%.1f%% → %.1f%%",
      threshold, baseline_pct, adjusted_pct
    ))

  threshold_points <- bind_rows(
    data.frame(
      model = "Aβ load",
      threshold = threshold_annotations$threshold,
      plaques_ge_pct = threshold_annotations$baseline_pct
    ),
    data.frame(
      model = "Aβ + region",
      threshold = threshold_annotations$threshold,
      plaques_ge_pct = threshold_annotations$adjusted_pct
    )
  )
  threshold_points$model <- factor(threshold_points$model, levels = models)

  maximum_row <- data |>
    slice_max(fold_error, n = 1, with_ties = FALSE)
  maximum_x <- maximum_row$fold_error

  p <- ggplot(
    curve,
    aes(threshold, plaques_ge_pct, colour = model, group = model)
  ) +
    geom_vline(
      data = threshold_annotations,
      aes(xintercept = threshold),
      inherit.aes = FALSE, colour = "#76ADD2",
      linewidth = 0.34, linetype = "dashed", alpha = 0.88
    ) +
    geom_line(linewidth = 0.50, lineend = "round") +
    geom_point(
      data = threshold_points,
      aes(threshold, plaques_ge_pct, colour = model),
      inherit.aes = FALSE, size = 2.55, stroke = 0,
      show.legend = FALSE
    ) +
    geom_text(
      data = threshold_annotations,
      aes(x = text_x, y = text_y, label = label),
      inherit.aes = FALSE, hjust = 0, vjust = 0.5,
      family = "Arial", size = 3.05, lineheight = 0.92,
      colour = "#171717"
    ) +
    annotate(
      "text", x = 3.98, y = 9.2,
      label = sprintf(
        "Tail extends beyond 4×\n(max ≈ %.1f×)",
        maximum_x
      ),
      hjust = 1, vjust = 0.5, family = "Arial",
      size = 2.85, lineheight = 0.92, colour = "#171717"
    ) +
    scale_colour_manual(
      values = model_colours,
      breaks = models,
      labels = c("Aβ load", "Aβ + region"),
      name = NULL
    ) +
    scale_x_continuous(
      breaks = seq(1, 4, by = 0.5),
      labels = sprintf("%.1f", seq(1, 4, by = 0.5)),
      limits = c(1, 4),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_continuous(
      breaks = seq(0, 100, by = 20),
      limits = c(0, 100), expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = "Regional adjustment barely changes\nthe plaque-level error tail",
      subtitle = sprintf(
        "Median %.2f× → %.2f×;  ≥2× %.1f%% → %.1f%%  (n = %d)",
        m1$median_error, m2$median_error,
        100 * m1$fraction_ge_2, 100 * m2$fraction_ge_2, m1$n
      ),
      x = "Symmetric fold error",
      y = "Plaques with error ≥ x (%)"
    ) +
    theme_fig2(base_size = 9.7) +
    theme(
      plot.title = element_text(size = 11.2, lineheight = 0.96),
      plot.subtitle = element_text(size = 8.8, margin = margin(b = 4.5)),
      legend.position = "inside",
      legend.position.inside = c(0.82, 0.89),
      legend.justification = c(0.5, 0.5),
      legend.direction = "vertical",
      legend.background = element_rect(fill = alpha("white", 0.80), colour = NA),
      legend.key.width = grid::unit(10, "mm"),
      legend.key.height = grid::unit(3.7, "mm"),
      legend.text = element_text(size = 8.9, colour = "#171717"),
      axis.title.x = element_text(size = 9.6, margin = margin(t = 6)),
      axis.title.y = element_text(size = 9.6, margin = margin(r = 6)),
      axis.text.x = element_text(size = 8.3),
      axis.text.y = element_text(size = 8.5),
      axis.line = element_line(colour = "#171717", linewidth = 0.62),
      axis.ticks = element_line(colour = "#171717", linewidth = 0.52),
      axis.ticks.length = grid::unit(1.9, "mm"),
      panel.grid.major.y = element_line(colour = "#ECECEC", linewidth = 0.34),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4.2, 4.2, 4.0, 4.7)
    )

    p
}
