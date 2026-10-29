"""Nested animal-isolated prediction with inner-model ensembles."""
from pathlib import Path
import numpy as np
import pandas as pd
from plaque_analysis.io.modeling import targets, write_json
from plaque_analysis.models.metrics import evaluate_predictions, paired_improvements
from plaque_analysis.models.splits import inner_splits, outer_splits
from plaque_analysis.models.training import predict_regressor, train_regressor


def fit_ensemble(frame, train_rows, prediction_rows, features, config, *, seed, model_directory=None):
    """Fit training-side inner folds. Prediction-side outcomes are never read."""
    training = frame.iloc[train_rows]
    test_x = frame.iloc[prediction_rows][features].to_numpy(dtype=float)
    x, y = training[features].to_numpy(dtype=float), targets(training, config)
    groups = training[config.animal_column].astype(str).to_numpy()
    predictions, audits = [], []
    for fold, (fit, validation) in enumerate(inner_splits(groups, config.inner_folds, seed)):
        fitted = train_regressor(x[fit], y[fit], x[validation], y[validation], features, config, seed=seed + fold)
        predictions.append(predict_regressor(fitted, test_x))
        audit = {**fitted[3], "inner_fold": fold,
                 "training_animals": sorted(set(groups[fit])), "validation_animals": sorted(set(groups[validation]))}
        if model_directory is not None:
            import joblib
            import torch
            destination = Path(model_directory) / f"inner_{fold:02d}"
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(fitted[0].state_dict(), destination / "network.pt")
            joblib.dump({"features": fitted[1], "target": fitted[2]}, destination / "transforms.joblib")
        audits.append(audit)
    # Each model has its own training-side target scale. Average only inverse-transformed predictions.
    return np.mean(predictions, axis=0), audits


def run_prediction(frame, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = frame[config.animal_column].astype(str).to_numpy()
    if len(set(groups)) - 1 < config.inner_folds:
        raise ValueError("The outer training animals cannot support the requested inner_folds")
    y = targets(frame, config)
    prediction_rows, audit_rows = [], []
    models = config.model_inputs()
    for outer, (animal, train, test) in enumerate(outer_splits(groups)):
        print(f"Prediction: outer animal {outer + 1}/{len(set(groups))}: {animal}", flush=True)
        for name in ("M0", *models):
            if name == "M0":
                predicted = np.full(len(test), y[train].mean())
                audits = []
            else:
                weights = output / "models" / name / f"outer_{outer:02d}" if config.save_network_weights else None
                predicted, audits = fit_ensemble(frame, train, test, models[name], config,
                                                 seed=config.seed + 1000 * outer, model_directory=weights)
            prediction_rows.extend({"plaque_id": frame.iloc[i][config.id_column], "animal_id": animal,
                                    "model": name, "observed": y[i], "predicted": value, "outer_fold": outer}
                                   for i, value in zip(test, predicted))
            audit_rows.append({"model": name, "outer_fold": outer, "test_animals": [animal],
                               "outer_training_animals": sorted(set(groups[train])), "inner_models": audits})
    predictions = pd.DataFrame(prediction_rows)
    errors, scores = evaluate_predictions(predictions, config)
    individual, gains = paired_improvements(errors, config)
    for name, table in (("predictions", predictions), ("animal_mae", errors), ("model_performance", scores),
                        ("paired_animal_gains", individual), ("paired_gain_summary", gains)):
        table.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "fold_audit.json", audit_rows)
    return predictions
