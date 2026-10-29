"""Unique plaque ownership and observed annular areas on the target image grid."""

import numpy as np
from plaque_analysis._native import observed_windows as _core


def _validate_grid(plaques, plaque_mask, observation_mask, pixel_size_um):
    """Validate corresponding plaque labels, shared support and physical spacing."""
    labels = np.asarray(plaque_mask)
    observed = np.asarray(observation_mask)
    spacing = np.asarray(pixel_size_um, dtype=float)
    if (labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or observed.shape != labels.shape
            or not np.isin(observed, [0, 1]).all()):
        raise ValueError("Plaque labels and binary observation support must share a 2-D grid")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("target_pixel_size_um must be positive [x, y] spacing")
    values = plaques.mask_label.to_numpy(float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or (values != np.floor(values)).any() or len(set(values)) != len(values)):
        raise ValueError("mask_label must contain unique positive integers")
    if set(np.unique(labels)) - {0} != set(values.astype(int)):
        raise ValueError("Every labelled plaque must occur once in plaques.csv")
    if not len(plaques):
        raise ValueError("At least one plaque is required")
    return labels, observed.astype(bool), spacing, values


def _annular_areas(plaques, labels, observed, spacing, values, distance, owner):
    return _core._annular_areas(plaques, labels, observed, spacing, values, distance, owner)


def _assign_cells(cells, plaques, labels, observed, spacing, values, distance, owner):
    return _core._assign_cells(cells, plaques, labels, observed, spacing, values, distance, owner)


def assign_spatial_windows(cells, plaques, plaque_mask, observation_mask, pixel_size_um):
    """Return (cells, areas): plaque ownership per cell and observed annular areas per plaque.

    cells: target_x, target_y (target pixels), valid_for_spatial_sampling.
    plaques: plaque_id, mask_label. observation_mask is the common measured
    support; pixel_size_um is (x, y). Cells gain plaque_id, boundary_distance_um
    (negative inside a plaque), x_um and y_um; areas hold area_0_50_um2 and
    area_50_150_um2.
    """
    return _core.assign_spatial_windows(cells, plaques, plaque_mask, observation_mask, pixel_size_um)
