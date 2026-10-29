assign_stratified_folds <- function(data, folds = 10L, seed) {
  set.seed(seed)
  fold_id <- integer(nrow(data))
  for (region in levels(droplevels(data$region_raw))) {
    index <- which(as.character(data$region_raw) == region)
    shuffled <- sample.int(length(index))
    fold_id[index] <-
      ((rank(shuffled, ties.method = "first") - 1L) %% folds) + 1L
  }
  fold_id
}

r2_score <- function(observed, predicted) {
  denominator <- sum((observed - mean(observed))^2)
  if (!is.finite(denominator) || denominator <= 0) return(NA_real_)
  1 - sum((observed - predicted)^2) / denominator
}


rebuild_fig2g_folds <- function(model_data, pooled_reference, repeat_ids = 1:100) {
  if (!requireNamespace("mgcv", quietly = TRUE)) stop("Install mgcv to rebuild folds.")
  stopifnot(length(repeat_ids) > 0L, !anyDuplicated(repeat_ids), all(repeat_ids %in% 1:100))
  stopifnot(
    nrow(model_data) == 853L,
    identical(sort(unique(model_data$repeat_id)), 1L),
    all(model_data$model == "Aβ load"),
    nrow(pooled_reference) == 300L,
    length(unique(pooled_reference$repeat_id)) == 100L
  )
  
  region_order <- c("ctx", "bs", "hpf", "cing", "str")
  model_order <- c("Mean only", "Aβ load", "Aβ + region")
  model_data$region_raw <- factor(model_data$region_raw, levels = region_order)
  model_data$x_load <- log1p(model_data$abeta_area_um2)
  model_data$y_path <- model_data$observed_log1p
  
  fold_records <- vector("list", length(repeat_ids) * 10L * length(model_order))
  record_index <- 0L
  
  for (repeat_id in repeat_ids) {
    repeat_rows <- pooled_reference[
      pooled_reference$repeat_id == repeat_id,
      , drop = FALSE
    ]
    seed <- unique(repeat_rows$seed)
    stopifnot(length(seed) == 1L)
  
    fold_id <- assign_stratified_folds(model_data, folds = 10L, seed = seed)
  
    predictions <- matrix(
      NA_real_, nrow = nrow(model_data), ncol = length(model_order),
      dimnames = list(NULL, model_order)
    )
  
    for (fold in seq_len(10L)) {
      test_index <- which(fold_id == fold)
      train_data <- model_data[fold_id != fold, , drop = FALSE]
      test_data <- model_data[test_index, , drop = FALSE]
  
      fit_mean <- lm(y_path ~ 1, data = train_data)
      fit_load <- mgcv::gam(
        y_path ~ s(x_load, bs = "cs", k = 8),
        data = train_data, method = "REML"
      )
      fit_region <- mgcv::gam(
        y_path ~ s(x_load, bs = "cs", k = 8) + region_raw,
        data = train_data, method = "REML"
      )
  
      fold_predictions <- list(
        "Mean only" = as.numeric(predict(fit_mean, newdata = test_data)),
        "Aβ load" = as.numeric(predict(fit_load, newdata = test_data)),
        "Aβ + region" = as.numeric(predict(fit_region, newdata = test_data))
      )
  
      for (model_name in model_order) {
        predictions[test_index, model_name] <- fold_predictions[[model_name]]
        record_index <- record_index + 1L
        fold_records[[record_index]] <- data.frame(
          repeat_id = repeat_id,
          seed = seed,
          fold_id = fold,
          n_fold = length(test_index),
          model = model_name,
          fold_r2 = r2_score(test_data$y_path, fold_predictions[[model_name]]),
          is_canonical_split = repeat_id == 1L,
          stringsAsFactors = FALSE
        )
      }
    }
  
    stopifnot(all(is.finite(predictions)))
  }
  
  fold_data <- bind_rows(fold_records)
  fold_data$model <- factor(fold_data$model, levels = model_order)
  fold_data <- fold_data |>
    left_join(
      pooled_reference |>
        select(repeat_id, seed, model, pooled_r2 = r2),
      by = c("repeat_id", "seed", "model")
    ) |>
    arrange(repeat_id, fold_id, model)
  
  stopifnot(
    nrow(fold_data) == length(repeat_ids) * 30L,
    all(is.finite(fold_data$fold_r2)),
    all(is.finite(fold_data$pooled_r2)),
    all(table(fold_data$model) == length(repeat_ids) * 10L)
  )
  
  fold_data
}
