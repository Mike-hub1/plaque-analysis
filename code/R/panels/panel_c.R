# Figure 2c; builder returns a plot and does not write files.
build_fig2c <- function(panel_data) {
  data <- panel_data |>
    transmute(
      brain_region = region_raw,
      matched_ab_area_um2 = abeta_area_um2,
      group_n = n_group,
      lamp1_range_fold = fold_range_max_min
    ) |>
    arrange(desc(group_n))
  stopifnot(
    nrow(data) == 72L,
    sum(data$group_n) == 634L,
    all(is.finite(data$matched_ab_area_um2)),
    all(is.finite(data$lamp1_range_fold)),
    all(data$matched_ab_area_um2 > 0),
    all(data$lamp1_range_fold > 0),
    all(data$group_n >= 3L)
  )

  median_spread <- median(data$lamp1_range_fold)
  fraction_ge_2 <- mean(data$lamp1_range_fold >= 2)
  right_limit <- max(data$lamp1_range_fold) * 1.06
  bottom_limit <- min(data$matched_ab_area_um2) * 0.68
  top_limit <- max(data$matched_ab_area_um2) * 1.55
  data$region_factor <- factor(data$brain_region, levels = region_levels)

  # Panel-specific pastel palette supplied by the author. The mapping keeps CTX
  # in blue and BS in yellow, while purple and pink distinguish HPF and STR.
  fig2c_region_palette <- c(
    ctx = "#ABC8E5",
    bs = "#D0D08A",
    hpf = "#9F8DB8",
    str = "#D8A0C1"
  )
  present_region_levels <- region_levels[
    region_levels %in% unique(data$brain_region)
  ]

  p <- ggplot(data, aes(lamp1_range_fold, matched_ab_area_um2)) +
    annotate("rect", xmin = 1, xmax = 2, ymin = bottom_limit, ymax = top_limit, fill = "#F3F4F6") +
    annotate("rect", xmin = 2, xmax = median_spread, ymin = bottom_limit, ymax = top_limit, fill = "#FBF8F2") +
    annotate("rect", xmin = median_spread, xmax = right_limit, ymin = bottom_limit, ymax = top_limit, fill = "#EAF3F8") +
    geom_vline(xintercept = 2, colour = "#8F8F8F", linetype = "22", linewidth = 0.42) +
    geom_vline(xintercept = median_spread, colour = "#2F73AE", linewidth = 0.62) +
    geom_point(
      aes(size = group_n, fill = region_factor),
      shape = 21, colour = "white", stroke = 0.42, alpha = 0.96
    ) +
    annotate(
      "text", x = 2, y = max(data$matched_ab_area_um2) * 1.27,
      label = "2×", hjust = 1.18, vjust = 1, size = 2.75, colour = "#8F8F8F"
    ) +
    annotate(
      "text", x = median_spread, y = max(data$matched_ab_area_um2) * 1.27,
      label = sprintf("median %.2f×", median_spread),
      hjust = -0.08, vjust = 1, size = 2.75, colour = "#2F73AE"
    ) +
    scale_fill_manual(
      values = fig2c_region_palette, breaks = present_region_levels,
      labels = region_labels[present_region_levels],
      name = "Recorded region", drop = TRUE
    ) +
    scale_size_area(
      max_size = 8, breaks = c(3, 10, 30, 60), name = "Plaques per group"
    ) +
    scale_x_continuous(
      trans = "log2", breaks = c(1, 2, 3, 5, 10, 30, 50),
      labels = function(x) paste0(x, "×"), limits = c(1, right_limit),
      expand = expansion(mult = 0)
    ) +
    scale_y_log10(
      breaks = c(10, 20, 30, 50, 100, 200, 300),
      labels = label_number(accuracy = 1), limits = c(bottom_limit, top_limit),
      expand = expansion(mult = 0)
    ) +
    labs(
      title = "Matched Aβ loads retain substantial plaque-level LAMP1 variability",
      subtitle = sprintf(
        "%d groups; median spread %.2f×; %.0f%% ≥2×",
        nrow(data), median_spread, 100 * fraction_ge_2
      ),
      x = "Within-group LAMP1 spread (max/min)",
      y = "Matched Aβ plaque area (µm²)"
    ) +
    theme_classic(base_size = 9, base_family = "Arial") +
    theme(
      # Keep the two encodings in one compact legend strip. This removes the
      # large two-row gap that previously separated the subtitle from the plot.
      legend.position = "top",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.box.just = "left",
      legend.justification = "left",
      legend.title = element_text(size = 7.8, margin = margin(l = 5.0, r = 2.5)),
      legend.text = element_text(size = 7.4),
      legend.key.width = grid::unit(15, "pt"),
      legend.key.height = grid::unit(18, "pt"),
      legend.spacing.x = grid::unit(2.2, "pt"),
      legend.spacing.y = grid::unit(0, "pt"),
      legend.box.spacing = grid::unit(1.2, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      axis.text = element_text(size = 7.6, colour = "#2B2B2B"),
      axis.title = element_text(size = 9, colour = "#2B2B2B"),
      plot.title = element_text(size = 11, face = "bold", margin = margin(b = 1)),
      plot.subtitle = element_text(size = 9, colour = "#666666", margin = margin(b = 1.5)),
      panel.grid = element_blank(),
      plot.margin = margin(4.0, 9.5, 5.0, 6.5)
    ) +
    guides(
      fill = guide_legend(
        order = 1, nrow = 1, byrow = TRUE, title.position = "left",
        override.aes = list(shape = 21, size = 3.4, colour = "white", alpha = 1)
      ),
      size = guide_legend(
        order = 2, nrow = 1, byrow = TRUE, title.position = "left",
        override.aes = list(
          shape = 21, fill = NA, colour = "#596670", stroke = 0.65, alpha = 1
        )
      )
    )

    p
}
