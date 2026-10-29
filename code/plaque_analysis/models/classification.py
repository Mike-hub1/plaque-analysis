"""Apply a fixed Figure 5 imaging atlas without fitting or RNA relabeling."""

import numpy as np
from plaque_analysis.models.persistence import load_atlas


def atlas_parts(atlas):
    names = ("features", "transform", "mixture", "component_order")
    try:
        parts = [atlas[name] for name in names] if isinstance(atlas, dict) else [getattr(atlas, name) for name in names]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Fixed atlas must supply {', '.join(names)}") from exc
    features, transform, mixture, order = parts
    features = list(features)
    order = np.asarray(order)
    if len(features) != 60 or len(features) != len(set(features)):
        raise ValueError("The manuscript atlas requires exactly 60 unique imaging feature names")
    if order.shape != (3,) or not np.issubdtype(order.dtype, np.integer) or set(order.tolist()) != {0, 1, 2}:
        raise ValueError("S1-S3 require a fixed three-component atlas and its component order")
    if not callable(getattr(transform, "transform", None)) or not callable(getattr(mixture, "predict_proba", None)):
        raise ValueError("Fixed atlas must provide fitted transform and predict_proba methods")
    return features, transform, mixture, order


def assign_fixed_classes(features, model_path):
    if "plaque_id" not in features or features.plaque_id.isna().any() or features.plaque_id.duplicated().any():
        raise ValueError("Imaging features require unique plaque_id values")
    names, transform, mixture, order = atlas_parts(load_atlas(model_path))
    missing = set(names) - set(features)
    if missing:
        raise ValueError(f"Missing fixed-atlas imaging features: {sorted(missing)}")
    values = features[names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fixed-atlas imaging features must be finite")
    probability = np.asarray(mixture.predict_proba(transform.transform(values)), dtype=float)
    if probability.shape != (len(features), 3) or not np.isfinite(probability).all():
        raise ValueError("The fixed atlas must return three finite posterior probabilities per plaque")
    if (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError("GMM posterior probabilities must be in [0, 1] and sum to one")
    probability = probability[:, order]
    result = features[["plaque_id"]].copy()
    result["class"] = [f"S{index + 1}" for index in probability.argmax(axis=1)]
    for index in range(3):
        result[f"probability_S{index + 1}"] = probability[:, index]
    result["maximum_posterior"] = probability.max(axis=1)
    return result
