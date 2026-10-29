pooled_r2 <- function(observed, predicted) {
  1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
}

stratified_bootstrap_correlation <- function(data, x, y, replicates = 2000L, seed = 20260808L) {
  set.seed(seed)
  strata <- split(seq_len(nrow(data)), data$region_raw)
  estimates <- vapply(seq_len(replicates), function(i) {
    index <- unlist(
      lapply(strata, function(z) sample(z, length(z), replace = TRUE)),
      use.names = FALSE
    )
    suppressWarnings(cor(x[index], y[index], method = "spearman"))
  }, numeric(1))
  c(
    estimate = suppressWarnings(cor(x, y, method = "spearman")),
    low = unname(quantile(estimates, 0.025, na.rm = TRUE)),
    high = unname(quantile(estimates, 0.975, na.rm = TRUE))
  )
}

