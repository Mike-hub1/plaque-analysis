"""Repeated age-stratified animal sampling with reference-label alignment."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from plaque_analysis.models.atlas import fit_atlas
from plaque_analysis.io.modeling import write_json
from plaque_analysis.models.splits import stratified_animal_sets


def align_components(reference_labels, fitted_labels, k):
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (fitted_labels, reference_labels), 1)
    fitted, reference = linear_sum_assignment(-counts)
    mapping = np.empty(k, dtype=int)
    mapping[fitted] = reference
    return mapping


def aligned_probabilities(probabilities, mapping):
    aligned = np.zeros_like(probabilities)
    aligned[:, mapping] = probabilities
    return aligned


def run_stability(frame, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    combinations = {**config.feature_blocks, "F+Popt+PPA": config.imaging_features()}
    # These all-data references match the manuscript's descriptive stability comparison.
    references = {name: fit_atlas(frame, columns, config) for name, columns in combinations.items()}
    reference_labels = {name: model.labels(frame) for name, model in references.items()}
    samples = stratified_animal_sets(frame, config)
    rows, posterior_rows, bic_rows = [], [], []
    joint = references["F+Popt+PPA"]
    posterior_sum = np.zeros((joint.k, joint.k))
    posterior_count = np.zeros(joint.k, dtype=int)
    for repeat, animals in enumerate(samples):
        train = frame[config.animal_column].astype(str).isin(animals).to_numpy()
        print(f"Stability: animal sample {repeat + 1}/{len(samples)}", flush=True)
        for name, columns in combinations.items():
            ref = references[name]
            fitted = fit_atlas(frame[train], columns, config, fixed_k=ref.k, seed=config.seed + repeat + 1)
            mapping = align_components(reference_labels[name][train], fitted.labels(frame[train]), ref.k)
            posterior = aligned_probabilities(fitted.probabilities(frame[~train]), mapping)
            target = reference_labels[name][~train]
            rows.append({"repeat": repeat, "feature_set": name, "reference_k": ref.k,
                         "heldout_ari": adjusted_rand_score(target, posterior.argmax(axis=1)),
                         "training_animals": "|".join(animals),
                         "test_animals": "|".join(sorted(set(frame.loc[~train, config.animal_column].astype(str))))})
            selected = fit_atlas(frame[train], columns, config, seed=config.seed + repeat + 1)
            bic_rows.append({"repeat": repeat, "feature_set": name, "selected_k": selected.k})
            if name == "F+Popt+PPA":
                for reference_class in range(ref.k):
                    subset = posterior[target == reference_class]
                    if len(subset):
                        posterior_sum[:, reference_class] += subset.sum(axis=0)
                        posterior_count[reference_class] += len(subset)
                    for inferred_class in range(ref.k):
                        posterior_rows.append({"repeat": repeat, "reference_class": f"S{reference_class + 1}",
                                               "inferred_class": f"S{inferred_class + 1}", "n_heldout_plaques": len(subset),
                                               "mean_posterior": subset[:, inferred_class].mean() if len(subset) else np.nan})
    pd.DataFrame(rows).to_csv(output / "heldout_stability.csv", index=False)
    pd.DataFrame(bic_rows).to_csv(output / "reselected_class_counts.csv", index=False)
    pd.DataFrame(posterior_rows).to_csv(output / "posterior_by_repeat.csv", index=False)
    means = np.divide(posterior_sum, posterior_count[None, :],
                      out=np.full_like(posterior_sum, np.nan), where=posterior_count[None, :] > 0)
    pd.DataFrame(means, index=[f"S{i + 1}" for i in range(joint.k)],
                 columns=[f"S{i + 1}" for i in range(joint.k)]).rename_axis("inferred_class").to_csv(output / "mean_posterior.csv")
    write_json(output / "stability_audit.json", {
        "animal_sets": samples, "sampling": "without replacement within each age; unique construction sets",
        "reference_k": {name: model.k for name, model in references.items()},
        "reference_bic": {name: model.bic_table for name, model in references.items()},
        "alignment": "Hungarian overlap on construction plaques only; no LAMP1 input",
        "posterior_occurrences_per_reference_class": posterior_count,
        "reference_scope": "All-input imaging references; ARI describes reproducibility relative to each reference, not external validation",
    })
