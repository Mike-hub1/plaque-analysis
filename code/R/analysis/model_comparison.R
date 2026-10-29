# Summaries retain all repeated splits and real test-fold values.
prepare_fig2g <- function(pooled_data, fold_data) {
  model_order <- c("Mean only", "Aβ load", "Aβ + region")
  stopifnot(
    nrow(pooled_data) == 300L,
    nrow(fold_data) == 3000L,
    all(is.finite(pooled_data$r2)),
    all(is.finite(fold_data$fold_r2)),
    all(table(factor(pooled_data$model, levels = model_order)) == 100L),
    all(table(factor(fold_data$model, levels = model_order)) == 1000L)
  )
  
  pooled_data$model <- factor(pooled_data$model, levels = model_order)
  fold_data$model <- factor(fold_data$model, levels = model_order)
  
  # A deterministic, data-independent horizontal offset spreads all 1,000 real
  # fold-level results per model without changing their R² coordinates.
  fold_data <- fold_data |>
    mutate(
      model_x = as.numeric(model),
      point_index = (repeat_id - 1L) * 10L + fold_id - 1L,
      jitter_unit = ((point_index * 37L) %% 1001L) / 1000 - 0.5,
      x_plot = model_x + 0.30 * jitter_unit
    )
  
  pooled_summary <- pooled_data |>
    group_by(model) |>
    summarise(
      center = median(r2),
      pooled_q025 = unname(quantile(r2, 0.025)),
      pooled_q975 = unname(quantile(r2, 0.975)),
      .groups = "drop"
    )
  
  fold_summary <- fold_data |>
    group_by(model) |>
    summarise(
      fold_q025 = unname(quantile(fold_r2, 0.025)),
      fold_q975 = unname(quantile(fold_r2, 0.975)),
      .groups = "drop"
    )
  
  summary_data <- pooled_summary |>
    left_join(fold_summary, by = "model") |>
    mutate(
      x = as.numeric(model),
      center_label = sprintf("%.3f", center),
      label_y = c(0.155, 0.525, 0.685)
    )
  
  paired <- pooled_data |>
    select(repeat_id, model, r2) |>
    pivot_wider(names_from = model, values_from = r2) |>
    mutate(
      load_minus_mean = `Aβ load` - `Mean only`,
      region_minus_load = `Aβ + region` - `Aβ load`
    )
  
  delta_summary <- data.frame(
    comparison = c("load_minus_mean", "region_minus_load"),
    x1 = c(1, 2),
    x2 = c(2, 3),
    bracket_y = c(0.555, 0.735),
    bracket_tip_y = c(0.532, 0.712),
    delta_y = c(0.655, 0.842),
    interval_y = c(0.602, 0.790),
    stringsAsFactors = FALSE
  )
  
  delta_values <- lapply(delta_summary$comparison, function(column) {
    values <- paired[[column]]
    c(
      estimate = median(values),
      low = unname(quantile(values, 0.025)),
      high = unname(quantile(values, 0.975))
    )
  })
  delta_values <- do.call(rbind, delta_values)
  delta_summary$estimate <- delta_values[, "estimate"]
  delta_summary$low <- delta_values[, "low"]
  delta_summary$high <- delta_values[, "high"]
  delta_summary$delta_label <- sprintf(
    "ΔR² = %+.3f", delta_summary$estimate
  )
  delta_summary$interval_label <- sprintf(
    "[%.3f, %.3f]", delta_summary$low, delta_summary$high
  )
  
  # Draw each comparison bracket as one continuous path. Quarter-ellipse bends
  # replace the former right-angle corners while preserving the same endpoints.
  make_bracket_path <- function(row_index, radius_x = 0.060,
                                radius_y = 0.012, n_curve = 18L) {
    row <- delta_summary[row_index, , drop = FALSE]
    left_theta <- seq(pi, pi / 2, length.out = n_curve)
    right_theta <- seq(pi / 2, 0, length.out = n_curve)
  
    data.frame(
      comparison = row$comparison,
      x = c(
        row$x1,
        row$x1,
        row$x1 + radius_x + radius_x * cos(left_theta[-1]),
        row$x2 - radius_x,
        row$x2 - radius_x + radius_x * cos(right_theta[-1]),
        row$x2
      ),
      y = c(
        row$bracket_tip_y,
        row$bracket_y - radius_y,
        row$bracket_y - radius_y + radius_y * sin(left_theta[-1]),
        row$bracket_y,
        row$bracket_y - radius_y + radius_y * sin(right_theta[-1]),
        row$bracket_tip_y
      ),
      stringsAsFactors = FALSE
    )
  }
  
  bracket_path <- do.call(
    rbind,
    lapply(seq_len(nrow(delta_summary)), make_bracket_path)
  )
  
  cloud_colours <- c(
    "Mean only" = "#F3E49C",
    "Aβ load" = "#E7B0C5",
    "Aβ + region" = "#A1D9DE"
  )
  center_colours <- c(
    "Mean only" = "#F3E49C",
    "Aβ load" = "#E7B0C5",
    "Aβ + region" = "#A1D9DE"
  )
  
  list(fold_data = fold_data, summary_data = summary_data, delta_summary = delta_summary, bracket_path = bracket_path, cloud_colours = cloud_colours, center_colours = center_colours)
}
