"""Load supplied measurements, annotations and a fixed imaging model.

counts.npz is a sparse cell-by-gene raw-count matrix. Its axes follow cells.csv
and genes.csv exactly. Cells and landmarks use the platform-aligned DAPI pixel
coordinates x_dapi/y_dapi. Target tissue and plaque masks share the target
image pixel grid; calibration.json supplies target_pixel_size_um=[x, y].
Its optional display_transform is the rigid physical-coordinate orientation
used to put the ventral side at the top of panels a/c/e.
Manual annotations and candidate-marker/UCell evidence are supplied inputs.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def read_table(root, name, required, identifiers=()):
    frame = pd.read_csv(root / name, dtype={column: str for column in identifiers})
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for column in identifiers:
        if frame[column].isna().any() or frame[column].eq("").any() or frame[column].duplicated().any():
            raise ValueError(f"{name}: {column} must contain unique, nonempty identifiers")
    return frame


def read_boolean(values, name):
    normalized = values.astype(str).str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError(f"{name}: expected explicit True/False values")
    return normalized.eq("true")


def load_inputs(data_dir, config_file):
    root = Path(data_dir)
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    if "target_pixel_size_um" not in calibration:
        raise ValueError("calibration.json requires measured target_pixel_size_um=[x, y]")
    config["registration"]["target_pixel_size_um"] = calibration["target_pixel_size_um"]
    if "display_transform" in calibration:
        config["display"]["coordinate_transform"] = calibration["display_transform"]
    cells = read_table(root, "cells.csv", ["cell_id", "sample_id", "x_dapi", "y_dapi"], ["cell_id"])
    genes = read_table(root, "genes.csv", ["gene", "is_mitochondrial"], ["gene"])
    genes["is_mitochondrial"] = read_boolean(genes.is_mitochondrial, "is_mitochondrial")
    annotations = read_table(root, "annotations.csv",
        ["cell_id", "broad_celltype", "reviewed_subtype", "confidence", "annotation_eligible", "reviewed_region"], ["cell_id"])
    annotations["annotation_eligible"] = read_boolean(annotations.annotation_eligible, "annotation_eligible")
    landmarks = read_table(root, "landmarks.csv",
        ["landmark_id", "x_dapi", "y_dapi", "x_target", "y_target"], ["landmark_id"])
    plaques = read_table(root, "plaques.csv", ["plaque_id", "mask_label", "x_um", "y_um"], ["plaque_id"])
    features = read_table(root, "imaging_features.csv", ["plaque_id"], ["plaque_id"])
    membership = read_table(root, "gene_programs.csv", ["gene", "program", "selection_program"])
    membership["selection_program"] = read_boolean(membership.selection_program, "selection_program")
    counts = sparse.load_npz(root / "counts.npz").tocsr()
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("counts.npz axes must match cells.csv rows and genes.csv rows")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.equal(counts.data, np.floor(counts.data)).all():
        raise ValueError("counts.npz must contain finite, nonnegative raw integer counts")
    tissue = np.load(root / "tissue_mask.npy", allow_pickle=False)
    plaque_mask = np.load(root / "plaque_mask.npy", allow_pickle=False)
    if tissue.ndim != 2 or not np.isin(tissue, [0, 1]).all() or plaque_mask.shape != tissue.shape:
        raise ValueError("tissue_mask.npy must be a 2-D binary mask on the same grid as plaque_mask.npy")
    if cells.sample_id.isna().any() or cells.sample_id.nunique() != 1:
        raise ValueError("Process one section/sample at a time; do not combine coordinate systems")
    if set(features.plaque_id) != set(plaques.plaque_id):
        raise ValueError("Every plaque must have one row of imaging features")
    model = root / "reference_atlas.joblib"
    if not model.is_file():
        raise FileNotFoundError("Supply the fixed Figure 5 model as reference_atlas.joblib")
    return {"config": config, "counts": counts, "genes": genes, "cells": cells,
            "annotations": annotations, "landmarks": landmarks, "plaques": plaques,
            "features": features, "membership": membership, "tissue_mask": tissue.astype(bool),
            "plaque_mask": plaque_mask, "model_path": model}
