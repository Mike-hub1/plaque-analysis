"""Equal-plaque paired contrasts with physical spatial-block bootstrap intervals."""

import numpy as np
import pandas as pd


GROUP_COLUMNS = ("program", "S_label", "far_window")
REQUIRED_COLUMNS = (*GROUP_COLUMNS, "plaque_id", "x_um", "y_um", "score_near", "score_far")


def _validate_pairs(paired_scores):
    if not isinstance(paired_scores, pd.DataFrame):
        raise ValueError("paired_scores must be a DataFrame")
    if not paired_scores.columns.is_unique:
        raise ValueError("Paired-score column names must be unique")
    if set(REQUIRED_COLUMNS) - set(paired_scores):
        raise ValueError(f"Missing paired-score columns: {sorted(set(REQUIRED_COLUMNS) - set(paired_scores))}")
    data = paired_scores[list(REQUIRED_COLUMNS)].copy()
    names = [*GROUP_COLUMNS, "plaque_id"]
    if data[names].isna().any().any() or data.duplicated(names).any():
        raise ValueError("Each program/class/far-window needs one row per nonmissing plaque ID")
    if any(data[name].astype(str).str.strip().eq("").any() for name in names):
        raise ValueError("Program, class, window and plaque identifiers must be nonempty")
    if not data.S_label.isin(["S1", "S2", "S3"]).all():
        raise ValueError("S_label must contain S1, S2 or S3")
    if not data.far_window.isin(["150-300", ">300"]).all():
        raise ValueError("far_window must contain '150-300' or '>300' in micrometres")
    numeric_columns = ["x_um", "y_um", "score_near", "score_far"]
    if np.iscomplexobj(data[numeric_columns].to_numpy()):
        raise ValueError("Paired scores and physical coordinates must be real")
    values = data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Paired scores and physical coordinates must be finite")
    data[numeric_columns] = values
    consistency = data.groupby("plaque_id")[["S_label", "x_um", "y_um"]].nunique()
    if consistency.gt(1).any().any():
        raise ValueError("Each plaque must have one class and physical location")
    for column in ("sample_id", "section_id"):
        if column in paired_scores and (paired_scores[column].isna().any() or paired_scores[column].nunique() > 1):
            raise ValueError("Compute spatial-block intervals for one section at a time")
    data["delta"] = data.score_far - data.score_near
    return data


def _bootstrap_interval(group, generator, n_bootstrap):
    blocks = group.groupby(["block_x", "block_y"], sort=True).delta.agg(["sum", "count"])
    if len(blocks) < 2:
        return np.nan, np.nan, len(blocks), "insufficient_spatial_blocks"
    sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    estimates = np.empty(n_bootstrap)
    # Batch block draws to keep temporary memory independent of total repeats.
    for start in range(0, n_bootstrap, 256):
        stop = min(start + 256, n_bootstrap)
        sampled = generator.integers(0, len(blocks), size=(stop - start, len(blocks)))
        estimates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.quantile(estimates, [.025, .975], method="linear")
    return low, high, len(blocks), "ok"


def bootstrap_spatial_contrasts(paired_scores, *, block_size_um=1000.0,
                                n_bootstrap=5000, seed=42):
    """Summarize far-minus-near scores with a 95% spatial-block percentile CI.

    Supply already eligible paired plaque scores: both windows must satisfy
    the study's cell-count/QC rules before this call. Locations are in target
    physical micrometres. Nonoverlapping square blocks use floor(x/block_size),
    floor(y/block_size), anchored at the image origin (0, 0). Within each
    program/class/far-window stratum, sample its occupied blocks with replacement
    and retain every plaque in each selected block, including repeated blocks.
    Each plaque has equal weight; blocks are not first reduced to equal-weight
    means. Fewer than two occupied blocks yield a mean but an unavailable CI.
    This describes within-section spatial variation, not independent animals.
    """
    if not np.isfinite(block_size_um) or block_size_um <= 0:
        raise ValueError("block_size_um must be finite and positive")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    data = _validate_pairs(paired_scores)
    data[["block_x", "block_y"]] = np.floor(data[["x_um", "y_um"]] / block_size_um)
    generator = np.random.default_rng(seed)
    rows = []
    for key, group in data.groupby(list(GROUP_COLUMNS), sort=True):
        low, high, n_blocks, status = _bootstrap_interval(group, generator, n_bootstrap)
        rows.append({**dict(zip(GROUP_COLUMNS, key)), "effect": group.delta.mean(),
            "low": low, "high": high, "n": len(group), "n_blocks": n_blocks,
            "block_size_um": float(block_size_um), "n_bootstrap": int(n_bootstrap),
            "seed": int(seed), "status": status})
    columns = [*GROUP_COLUMNS, "effect", "low", "high", "n", "n_blocks",
               "block_size_um", "n_bootstrap", "seed", "status"]
    return pd.DataFrame(rows, columns=columns)
