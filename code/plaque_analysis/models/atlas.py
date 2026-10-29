"""Outcome-blind Gaussian mixture reference classes on standardized imaging features."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from plaque_analysis.io.modeling import construction_mask, targets, write_json
from plaque_analysis.models.prediction import fit_ensemble
from plaque_analysis.models.preprocessing import FeatureTransform


@dataclass
class Atlas:
    features: list
    transform: FeatureTransform
    mixture: GaussianMixture
    component_order: np.ndarray
    bic_table: list

    @property
    def k(self):
        return self.mixture.n_components

    def probabilities(self, frame):
        x = self.transform.transform(frame[self.features].to_numpy(dtype=float))
        return self.mixture.predict_proba(x)[:, self.component_order]

    def labels(self, frame):
        return self.probabilities(frame).argmax(axis=1)


def fit_atlas(frame, features, config, *, fixed_k=None, seed=None):
    """Accepts imaging columns explicitly; never reads target or baseline fields."""
    if not set(features) <= set(config.imaging_features()):
        raise ValueError("Atlas features must come only from declared imaging blocks")
    seed = int(config.seed if seed is None else seed) % (2**32 - 1)
    x = frame[features].to_numpy(dtype=float)
    transform = FeatureTransform("standard", seed=seed).fit(x, features)
    z = transform.transform(x)
    candidates = [fixed_k] if fixed_k is not None else config.k_candidates
    table, fits = [], []
    for k in candidates:
        if k >= len(z):
            raise ValueError(f"Class count {k} requires more than {k} construction plaques")
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=config.gmm_initializations,
                                max_iter=config.gmm_max_iterations, reg_covar=config.covariance_regularization,
                                random_state=seed).fit(z)
        table.append({"k": k, "bic": model.bic(z), "converged": bool(model.converged_)})
        if model.converged_:
            fits.append((table[-1]["bic"], model))
    if not fits:
        raise RuntimeError("No GMM candidate converged")
    mixture = min(fits, key=lambda item: item[0])[1]
    # Stable numerical IDs use component centres, without any outcome-based relabeling.
    order = np.lexsort(mixture.means_[:, ::-1].T)
    return Atlas(list(features), transform, mixture, order, table)


def atlas_assignments(frame, atlas, config, construction=None):
    result = frame[[config.id_column, config.animal_column]].rename(
        columns={config.id_column: "plaque_id", config.animal_column: "animal_id"}).copy()
    probabilities = atlas.probabilities(frame)
    result["class"] = [f"S{i + 1}" for i in probabilities.argmax(axis=1)]
    result["maximum_posterior"] = probabilities.max(axis=1)
    for k in range(atlas.k): result[f"probability_S{k + 1}"] = probabilities[:, k]
    if construction is not None:
        result["subset"] = np.where(construction, "construction", "validation")
    return result


def run_atlas(frame, config, output, *, pathology=True):
    import joblib
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    construction = construction_mask(frame, config)
    reference = fit_atlas(frame[construction], config.imaging_features(), config)
    assignments = atlas_assignments(frame, reference, config, construction)
    assignments.to_csv(output / "reference_assignments.csv", index=False)
    pd.DataFrame(reference.bic_table).to_csv(output / "reference_bic.csv", index=False)
    joblib.dump(reference, output / "reference_atlas.joblib")
    z_train = reference.transform.transform(frame.loc[construction, reference.features].to_numpy(dtype=float))
    pca = PCA(n_components=min(2, z_train.shape[1])).fit(z_train)
    projection = pca.transform(reference.transform.transform(frame[reference.features].to_numpy(dtype=float)))
    projected = assignments.copy()
    projected["PC1"] = projection[:, 0]
    projected["PC2"] = projection[:, 1] if projection.shape[1] > 1 else 0.0
    projected.to_csv(output / "display_pca.csv", index=False)
    audit = {"construction_animals": list(config.construction_animals), "selected_k": reference.k,
             "features": reference.transform.audit(), "pca_used_for_clustering": False,
             "class_order": "lexicographic standardized imaging component centres; no outcome information",
             "pca_explained_variance_ratio": pca.explained_variance_ratio_}
    if pathology:
        add_pathology_association(frame, config, output, construction, assignments, audit)
    write_json(output / "atlas_audit.json", audit)
    return reference


def add_pathology_association(frame, config, output, construction, assignments, audit):
    predicted, fitted = fit_ensemble(frame, np.flatnonzero(construction), np.flatnonzero(~construction),
                                     list(config.baseline), config, seed=config.seed)
    residuals = assignments.loc[~construction, ["plaque_id", "animal_id", "class"]].copy()
    residuals["observed_log_area"] = targets(frame[~construction], config)
    residuals["baseline_prediction"] = predicted
    residuals["residual"] = residuals.observed_log_area - residuals.baseline_prediction
    residuals.to_csv(Path(output) / "validation_residuals.csv", index=False)
    residuals.groupby(["animal_id", "class"], as_index=False).agg(
        mean_residual=("residual", "mean"), n_plaques=("residual", "size")).to_csv(
        Path(output) / "animal_class_residuals.csv", index=False)
    audit["baseline_inner_models"] = fitted
