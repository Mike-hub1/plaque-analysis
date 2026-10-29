"""Infer descriptive regions from reviewed regional-indicator neurons."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def infer_reviewed_regions(points, cells, *, radius_um=150.0, minimum_neurons=5, majority=.6):
    """Use eligible mapped neurons in physical micrometres; never relabel NonTel.

    Both points and the cells' x_um/y_um columns use the same image coordinate
    system. Region names are supplied by the reviewed annotation table.
    """
    from plaque_analysis.analysis.rna_annotation import boolean_values

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Region query coordinates must be finite with shape (n, 2)")
    if not np.isfinite(radius_um) or radius_um <= 0 or minimum_neurons < 1 or not .5 < majority <= 1:
        raise ValueError("Invalid region neighborhood parameters")
    required = {"x_um", "y_um", "rna_qc_pass", "valid_for_spatial_sampling", "annotation_eligible", "subtype_eligible",
                "broad_celltype", "reviewed_region"}
    if required - set(cells):
        raise ValueError(f"Missing region inference fields: {sorted(required - set(cells))}")
    eligible = np.ones(len(cells), dtype=bool)
    for flag in ("rna_qc_pass", "valid_for_spatial_sampling", "annotation_eligible", "subtype_eligible"):
        eligible &= boolean_values(cells[flag], flag)
    regions = cells.reviewed_region.fillna("").astype(str).str.strip()
    eligible &= cells.broad_celltype.isin(["Excitatory neuron", "Inhibitory neuron"]).to_numpy()
    eligible &= regions.ne("").to_numpy()
    indicators = cells.loc[eligible, ["x_um", "y_um"]].to_numpy(dtype=float)
    if not np.isfinite(indicators).all():
        raise ValueError("Reviewed regional-indicator coordinates must be finite")
    return region_votes(points, indicators, regions.loc[eligible].to_numpy(), radius_um, minimum_neurons, majority)


def region_votes(points, indicators, labels, radius_um, minimum_neurons, majority):
    neighbors = cKDTree(indicators).query_ball_point(points, r=radius_um, workers=1)
    rows = []
    for indices in neighbors:
        count = len(indices)
        label, fraction = "Transition/undetermined", np.nan
        if count:
            counts = pd.Series(labels[indices]).value_counts()
            fraction = float(counts.iloc[0] / count)
            if count >= minimum_neurons and fraction >= majority:
                label = str(counts.index[0])
        rows.append((label, fraction, count))
    return pd.DataFrame(rows, columns=["inferred_region_code", "inferred_region_support_fraction",
                                       "n_region_indicative_neurons_150um"])
