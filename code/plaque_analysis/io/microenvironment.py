"""Read supplied measurements and upstream Figure 6 metadata.

mouse/counts.npz is a cell-by-gene raw-count matrix ordered like mouse/cells.csv
and mouse/genes.csv. Cells supply the Figure 6 QC/annotation flags and target
image coordinates. Plaques supply its fixed class and neighborhood_eligible
flag. Masks retain all plaques, so dropping a neighbourhood never reassigns
its cells to a different plaque. mouse/observation_mask.npy describes common
measured tissue/registration support for the density denominator.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from plaque_analysis.spatial.observed_windows import assign_spatial_windows


def read_table(path, required, *, boolean_columns=(), identifiers=()):
    frame = pd.read_csv(path, dtype={name: str for name in identifiers})
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {sorted(missing)}")
    for name in identifiers:
        if frame[name].isna().any() or frame[name].eq("").any() or frame[name].duplicated().any():
            raise ValueError(f"{name} must contain unique, nonempty identifiers")
    for name in boolean_columns:
        values = frame[name].astype(str).str.strip().str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"{name} must contain explicit True/False values")
        frame[name] = values.eq("true")
    return frame


def load_inputs(data_dir, config_file):
    root = Path(data_dir)
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    calibration = json.loads((root / "mouse/calibration.json").read_text(encoding="utf-8"))
    if "target_pixel_size_um" not in calibration:
        raise ValueError("mouse/calibration.json requires target_pixel_size_um=[x, y]")
    spacing = calibration["target_pixel_size_um"]
    if "display_transform" in calibration:
        config["display"]["coordinate_transform"] = calibration["display_transform"]
    flags = ["rna_qc_pass", "valid_for_spatial_sampling", "annotation_eligible"]
    cells = read_table(root / "mouse/cells.csv",
        ["cell_id", "target_x", "target_y", "broad_celltype", *flags],
        boolean_columns=flags, identifiers=["cell_id"])
    plaques = read_table(root / "mouse/plaques.csv",
        ["plaque_id", "mask_label", "class", "x_um", "y_um", "neighborhood_eligible"],
        boolean_columns=["neighborhood_eligible"], identifiers=["plaque_id"])
    if not plaques["class"].isin(["S1", "S2", "S3"]).all():
        raise ValueError("Plaques require the fixed Figure 5/Figure 6 class in S1–S3")
    if not np.isfinite(plaques[["x_um", "y_um"]].to_numpy(float)).all():
        raise ValueError("Plaque locations must be finite physical coordinates")
    genes = read_table(root / "mouse/genes.csv", ["gene"], identifiers=["gene"])
    memberships = read_table(root / "mouse/gene_sets.csv", ["program", "gene"])
    counts = sparse.load_npz(root / "mouse/counts.npz").tocsr()
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (counts.shape != (len(cells), len(genes)) or not np.isfinite(counts.data).all()
            or (counts.data < 0).any() or (counts.data != np.floor(counts.data)).any()):
        raise ValueError("Raw integer counts must match the cell and gene table order")
    mask = np.load(root / "mouse/plaque_mask.npy", allow_pickle=False)
    support = np.load(root / "mouse/observation_mask.npy", allow_pickle=False)
    cells, areas = assign_spatial_windows(cells, plaques, mask, support, spacing)
    return {"cells": cells, "plaques": plaques, "counts": counts,
            "genes": genes.gene.tolist(), "memberships": memberships,
            "areas": areas, "config": config, "data_dir": root}


def write_tables(tables, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(root / f"{name}.csv", index=False, float_format="%.15g")
