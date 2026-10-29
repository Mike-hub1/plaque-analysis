"""Assign registered nuclei to plaque boundaries and aggregate raw RNA counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from plaque_analysis._native import neighborhoods as _core


QC_COLUMNS = ("rna_qc_pass", "valid_for_spatial_sampling", "annotation_eligible")


def _boolean_values(series: pd.Series) -> np.ndarray:
    if series.isna().any() or not series.isin([True, False, 0, 1]).all():
        raise ValueError(f"{series.name} must contain nonmissing Boolean values")
    return series.to_numpy(dtype=bool)


def assign_neighborhoods(cells, plaque_mask, plaques, pixel_size_um,
                         radius_um=150, min_cells=20):
    """Return all cells with assignment flags and all plaques with eligible counts.

    cells: cell_id, target_x, target_y (target pixels), x_um, y_um and the
    QC_COLUMNS flags. plaques: plaque_id, mask_label, class, x_um, y_um.
    pixel_size_um is (x, y). Each cell has at most one assigned plaque.
    """
    return _core.assign_neighborhoods(cells, plaque_mask, plaques, pixel_size_um, radius_um, min_cells)


def aggregate_counts(counts, cell_ids, gene_names, assignments, neighborhood_meta):
    """Sum cell-by-gene raw counts over included cells, retaining all genes.

    Returns a gene-by-plaque DataFrame and all-gene library totals indexed by
    plaque_id. Count rows are matched to cells by cell_id.
    """
    return _core.aggregate_counts(counts, cell_ids, gene_names, assignments, neighborhood_meta)
