region_levels <- c("ctx", "bs", "hpf", "cing", "str")
region_labels <- c(
  ctx = "CTX", bs = "BS", hpf = "HPF", cing = "Cing", str = "STR"
)
region_palette <- c(
  ctx = "#4C78A8", bs = "#E3B341", hpf = "#68AD65",
  cing = "#9C6BA4", str = "#DB7272"
)
model_palette <- c(
  "Mean only" = "#CFCECE", "Aβ load" = "#4C78A8", "Aβ + region" = "#59A14F"
)

theme_fig2 <- function(base_size = 7.2) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#272727"),
      axis.ticks = element_line(linewidth = 0.35, colour = "#272727"),
      axis.text = element_text(size = 6.1, colour = "#272727"),
      axis.title = element_text(size = 6.8, colour = "#272727"),
      legend.title = element_text(size = 5.9, colour = "#272727"),
      legend.text = element_text(size = 5.8, colour = "#272727"),
      plot.title = element_text(size = 7.5, face = "bold", lineheight = 0.96),
      plot.subtitle = element_text(
        size = 6.2, colour = "#555555", lineheight = 0.98,
        margin = margin(b = 3)
      ),
      panel.grid = element_blank(),
      plot.margin = margin(3.0, 3.3, 3.0, 3.0)
    )
}

