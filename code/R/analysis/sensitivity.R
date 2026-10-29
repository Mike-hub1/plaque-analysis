# Fig. 2h distribution-enhanced forest plot builder.
#
# Figure contract
#   Core conclusion: incremental predictive information from recorded brain
#   region remains positive when brainstem plaques are retained, but collapses
#   to a small negative value when the brainstem contrast is absent.
#   Evidence: all 600 raw delta-R-squared values from 100 matched repeated
#   splits across six prespecified sensitivity scenarios.
#   Archetype: quantitative robustness panel.
#   Summary marks: median, IQR and the 2.5th-97.5th percentiles of the repeated
#   split distribution. These intervals describe resampling stability, not
#   biological confidence intervals or independent biological replication.
#   Scale: one common raw linear x-axis. A connected rectangular inset expands
#   only the two near-zero scenarios and never replaces the common-scale
#   comparison.

format_fig2h_delta <- function(x, compact = FALSE) {
  digits <- ifelse(abs(x) < 0.01, 4L, 3L)
  out <- mapply(
    function(value, n_digits) sprintf(paste0("%+.", n_digits, "f"), value),
    x,
    digits,
    USE.NAMES = FALSE
  )
  out <- sub("^-", "\u2212", out)
  if (compact) out <- sub("^([+\u2212])0\\.", "\\1.", out)
  out
}

prepare_fig2h_distribution <- function(panel_h_data) {
  required_columns <- c(
    "repeat_id", "seed", "scenario", "n", "A\u03b2 load",
    "A\u03b2 + region", "delta_r2_region"
  )
  stopifnot(
    all(required_columns %in% names(panel_h_data)),
    nrow(panel_h_data) == 600L,
    length(unique(panel_h_data$repeat_id)) == 100L,
    !anyNA(panel_h_data[, required_columns]),
    all(is.finite(panel_h_data$delta_r2_region)),
    all(abs(
      panel_h_data$delta_r2_region -
        (panel_h_data[["A\u03b2 + region"]] - panel_h_data[["A\u03b2 load"]])
    ) < 1e-12)
  )

  scenario_meta <- data.frame(
    scenario = c(
      "All regions",
      "Exclude hpf",
      "Exclude str",
      "Exclude Cing",
      "Exclude bs",
      "Major regions only"
    ),
    y = 6:1,
    label_full = c(
      "All regions (n = 853)",
      "\u2212 HPF (n = 756)",
      "\u2212 STR (n = 808)",
      "\u2212 Cing (n = 851)",
      "\u2212 BS (n = 834)",
      "CTX + HPF + STR (n = 832)"
    ),
    label_compact = c(
      "All", "\u2212HPF", "\u2212STR", "\u2212Cing", "\u2212BS", "CTX+HPF+STR"
    ),
    expected_n = c(853L, 756L, 808L, 851L, 834L, 832L),
    brainstem_status = c(
      rep("Brainstem retained", 4),
      rep("Brainstem absent", 2)
    ),
    stringsAsFactors = FALSE
  )

  observed_counts <- panel_h_data |>
    count(scenario, n, name = "repeated_splits")
  stopifnot(
    nrow(observed_counts) == 6L,
    all(observed_counts$repeated_splits == 100L),
    setequal(observed_counts$scenario, scenario_meta$scenario)
  )

  plot_data <- panel_h_data |>
    left_join(scenario_meta, by = "scenario") |>
    mutate(
      brainstem_status = factor(
        brainstem_status,
        levels = c("Brainstem retained", "Brainstem absent")
      )
    ) |>
    arrange(y, repeat_id)

  stopifnot(
    !anyNA(plot_data$y),
    all(plot_data$n == plot_data$expected_n)
  )

  set.seed(20260808L)
  plot_data$y_jitter <- plot_data$y + stats::runif(nrow(plot_data), -0.145, 0.145)

  summary_data <- plot_data |>
    group_by(
      scenario, y, label_full, label_compact, n,
      brainstem_status
    ) |>
    summarise(
      repeated_splits = n(),
      minimum = min(delta_r2_region),
      q025 = unname(quantile(delta_r2_region, 0.025)),
      q25 = unname(quantile(delta_r2_region, 0.25)),
      median_delta_r2 = median(delta_r2_region),
      mean_delta_r2 = mean(delta_r2_region),
      q75 = unname(quantile(delta_r2_region, 0.75)),
      q975 = unname(quantile(delta_r2_region, 0.975)),
      maximum = max(delta_r2_region),
      sd_delta_r2 = sd(delta_r2_region),
      positive_splits = sum(delta_r2_region > 0),
      negative_splits = sum(delta_r2_region < 0),
      .groups = "drop"
    ) |>
    mutate(
      median_label = format_fig2h_delta(median_delta_r2),
      main_label_x = if_else(
        median_delta_r2 >= 0,
        q975 + 0.0030,
        0.0045
      ),
      main_label_hjust = 0
    ) |>
    arrange(y)

  list(raw = plot_data, summary = summary_data, meta = scenario_meta)
}

fig2h_status_colours <- c(
  "Brainstem retained" = "#3F7FA6",
  "Brainstem absent" = "#D97A60"
)
