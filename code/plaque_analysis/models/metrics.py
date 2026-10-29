"""Pooled R², paired animal MAE and conditional animal-bootstrap intervals."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import r2_score


def evaluate_predictions(predictions, config):
    rows, scores = [], []
    for model, part in predictions.groupby("model", sort=True):
        scores.append({"model": model, "pooled_r2": r2_score(part.observed, part.predicted), "n_plaques": len(part)})
        for animal, animal_part in part.groupby("animal_id", sort=True):
            rows.append({"model": model, "animal_id": animal,
                         "mae": np.mean(np.abs(animal_part.observed - animal_part.predicted)),
                         "n_plaques": len(animal_part)})
    errors = pd.DataFrame(rows)
    summaries = pd.DataFrame(scores).merge(errors.groupby("model").mae.mean().rename("animal_mean_mae"), on="model")
    return errors, summaries


def paired_improvements(errors, config):
    wide = errors.pivot(index="animal_id", columns="model", values="mae")
    if "M5" not in wide or wide.isna().any().any():
        raise ValueError("Complete paired animal MAE for M5 and its comparators is required")
    rng = np.random.default_rng(config.seed)
    samples = rng.integers(0, len(wide), size=(config.bootstrap_repeats, len(wide)))
    individual, summary = [], []
    for model in ("M1", "M2", "M3", "M4"):
        if model not in wide:
            continue
        differences = (wide[model] - wide.M5).to_numpy()
        ci = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975])
        nonzero = differences[differences != 0]
        p = binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue if len(nonzero) else 1.0
        baseline_mean = float(wide[model].mean())
        summary.append({"comparator": model, "n_animals": len(wide), "mean_delta_mae": differences.mean(),
                        "ci_low": ci[0], "ci_high": ci[1],
                        "relative_reduction_percent": 100 * differences.mean() / baseline_mean if baseline_mean > 0 else np.nan,
                        "sign_test_p": p, "sign_test_nonzero_animals": len(nonzero)})
        individual.extend({"animal_id": animal, "comparator": model, "delta_mae": value,
                           "comparator_mae": wide.loc[animal, model]} for animal, value in zip(wide.index, differences))
    return pd.DataFrame(individual), pd.DataFrame(summary)


def animal_bootstrap_line(part, repeats, seed, points=100):
    """Refit only the display line; animal repeats retain all duplicate plaques."""
    animals = sorted(part.animal_id.unique())
    x = np.linspace(part.observed.min(), part.observed.max(), points)
    rng, curves = np.random.default_rng(seed), []
    for _ in range(repeats):
        drawn = rng.choice(animals, len(animals), replace=True)
        sample = pd.concat([part[part.animal_id == a] for a in drawn], ignore_index=True)
        if sample.observed.nunique() < 2:
            raise ValueError("Cannot fit a display line to constant observed targets")
        slope, intercept = np.polyfit(sample.observed, sample.predicted, 1)
        curves.append(slope * x + intercept)
    low, high = np.quantile(curves, [0.025, 0.975], axis=0)
    slope, intercept = np.polyfit(part.observed, part.predicted, 1)
    return pd.DataFrame({"observed": x, "fitted": slope * x + intercept, "ci_low": low, "ci_high": high})
