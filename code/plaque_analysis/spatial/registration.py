"""Landmark-based registration and spatial error summaries.

Coordinates are zero-based pixel centres; target pixel sizes convert
coordinates and errors to µm. Registration errors are landmark-based
estimates, not cell-wise ground truth or independent external validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from plaque_analysis._native import registration as _core


def _require_columns(frame, columns, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _require_identifiers(frame, column):
    identifiers = frame[column].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError(f"{column} must contain unique, nonempty identifiers")


def _require_one_section(*frames):
    for column in ("section_id", "sample_id"):
        identities = set()
        for frame in frames:
            if column not in frame:
                continue
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or values.eq("").any() or values.nunique() != 1:
                raise ValueError(f"Process one section at a time; {column} must have one nonempty value")
            identities.update(values.unique())
        if len(identities) > 1:
            raise ValueError(f"Cell and landmark {column} values must refer to the same section")


def _coordinates(frame, columns):
    values = frame[list(columns)].to_numpy()
    if np.iscomplexobj(values):
        raise ValueError("Coordinates must be real numbers")
    try:
        return frame[list(columns)].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("Coordinates must be numeric pixel positions") from error


def _registration_settings(config, tissue_mask):
    if not isinstance(config, dict) or "target_pixel_size_um" not in config:
        raise ValueError("Set target_pixel_size_um=[x_size_um, y_size_um] explicitly")
    if np.iscomplexobj(config["target_pixel_size_um"]):
        raise ValueError("Target pixel sizes must be real numbers")
    spacing = np.asarray(config["target_pixel_size_um"], dtype=float)
    smoothing = float(config.get("tps_smoothing", 0))
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("target_pixel_size_um must contain two finite positive values")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("tps_smoothing must be finite and nonnegative")
    mask = np.asarray(tissue_mask)
    if mask.ndim != 2 or 0 in mask.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("tissue_mask must be a nonempty two-dimensional boolean or 0/1 array")
    return spacing, smoothing, mask.astype(bool)


def _require_landmark_geometry(source, target):
    if len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("At least four finite corresponding landmarks are required")
    for name, coordinates in (("source", source), ("target", target)):
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError(f"Duplicate {name} landmark coordinates are not permitted")
        for omitted in range(len(coordinates)):
            training = np.delete(coordinates, omitted, axis=0)
            if np.linalg.matrix_rank(training - training.mean(axis=0)) < 2:
                raise ValueError(f"{name} landmarks must remain noncollinear in every leave-one-out fold")


def _fit_affine_tps(source, target, smoothing):
    return _core._fit_affine_tps(source, target, smoothing)


def _map_points(model, coordinates):
    return _core._map_points(model, coordinates)


def _landmark_loo_errors(landmarks, source, target, spacing, smoothing):
    return _core._landmark_loo_errors(landmarks, source, target, spacing, smoothing)


def _mapping_status(source, target, error, mask):
    return _core._mapping_status(source, target, error, mask)


def register_cells(cells: pd.DataFrame, landmarks: pd.DataFrame,
                   tissue_mask: np.ndarray, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map cells into the target image and attach registration errors.

    Required cells: cell_id, x_dapi, y_dapi, broad_celltype. Required landmarks:
    landmark_id, x_dapi, y_dapi, x_target, y_target. Each call covers one section.
    If section_id/sample_id columns are provided, they must agree across inputs.
    Returns (cells with target_x, target_y, x_um, y_um, registration_error_um and
    valid_for_spatial_sampling, landmark leave-one-out errors). x_um/y_um use
    target pixel-centre (0, 0) as their physical origin.
    """
    return _core.register_cells(cells, landmarks, tissue_mask, config)


def wilson_interval(excluded, total, z=1.959963984540054):
    """Wilson score bounds in percent for finite integer counts; scalar or 1D input."""
    if np.iscomplexobj(excluded) or np.iscomplexobj(total):
        raise ValueError("Wilson counts must be real integers")
    excluded, total = np.asarray(excluded, dtype=float), np.asarray(total, dtype=float)
    if excluded.ndim > 1 or total.ndim > 1:
        raise ValueError("Wilson counts must be scalars or one-dimensional arrays")
    excluded, total = np.broadcast_arrays(excluded, total)
    if (not np.isfinite(excluded).all() or not np.isfinite(total).all()
            or not np.equal(excluded, np.floor(excluded)).all()
            or not np.equal(total, np.floor(total)).all()
            or (total <= 0).any() or (excluded < 0).any() or (excluded > total).any()):
        raise ValueError("Require finite integer counts with 0 <= excluded <= total and total > 0")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Wilson z must be finite and positive")
    proportion = excluded / total
    denominator = 1 + z*z/total
    centre = (proportion + z*z/(2*total))/denominator
    half = z*np.sqrt(proportion*(1-proportion)/total + z*z/(4*total**2))/denominator
    low = np.where(excluded == 0, 0, np.clip(centre-half, 0, 1))
    high = np.where(excluded == total, 1, np.clip(centre+half, 0, 1))
    return np.column_stack((low, high)) * 100


def summarize_registration(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize all supplied cells; error quantiles use only available mappings."""
    _require_columns(cells, ("broad_celltype", "valid_for_spatial_sampling", "registration_error_um"), "cells")
    _require_one_section(cells)
    classes = cells.broad_celltype.astype("string").str.strip()
    flags = cells.valid_for_spatial_sampling.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    if classes.isna().any() or classes.eq("").any() or flags.isna().any():
        raise ValueError("Cell classes must be nonempty and availability flags must be explicit booleans")
    errors = pd.to_numeric(cells.registration_error_um, errors="raise").to_numpy(dtype=float)
    available = flags.to_numpy(dtype=bool)
    if not np.isfinite(errors[available]).all() or (errors[available] < 0).any():
        raise ValueError("Available mappings must have finite nonnegative registration errors")
    rows = []
    for label in classes.unique():
        group = classes.eq(label).to_numpy(dtype=bool)
        selected = errors[group & available]
        total, valid = int(group.sum()), len(selected)
        low, high = wilson_interval(total-valid, total)[0]
        quantiles = np.quantile(selected, [.05, .25, .5, .75, .95], method="linear") if valid else np.full(5, np.nan)
        rows.append({"broad_celltype": label, "total_n": total, "valid_n": valid,
                     "excluded_n": total-valid, "invalid_percent": 100*(total-valid)/total,
                     "wilson95_low_percent": low, "wilson95_high_percent": high,
                     **{f"registration_error_um_{key}": value for key, value in
                        zip(("p05", "p25", "median", "p75", "p95"), quantiles)}})
    return pd.DataFrame(rows).sort_values(["total_n", "broad_celltype"], ascending=[False, True]).reset_index(drop=True)
