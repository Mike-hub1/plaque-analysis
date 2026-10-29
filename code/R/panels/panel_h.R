build_fig2h_distribution <- function(panel_h_data, compact = FALSE) {
  prepared <- prepare_fig2h_distribution(panel_h_data)
  main <- build_fig2h_main(prepared, compact)
  zoom <- if (compact) NULL else build_fig2h_zoom(prepared$raw, prepared$summary)
  plot <- if (compact) main else compose_fig2h(main, zoom, prepared$summary)
  list(plot = plot, main_plot = main, zoom_plot = zoom,
       raw = prepared$raw, summary = prepared$summary)
}

build_fig2h <- function(panel_data) {
  build_fig2h_distribution(panel_data)$plot
}
