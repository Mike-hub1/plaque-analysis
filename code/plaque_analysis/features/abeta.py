"""Figure 3: abeta features."""
from __future__ import annotations

from plaque_analysis._native import abeta as _core

from scipy import stats
import numpy as np


def compute_advanced_stats(signed_intensity, mask):
    """Histogram descriptors of an EXPLICIT mask, retaining signed values."""
    a = np.asarray(signed_intensity, float)[mask]
    keys = ("mean_intensity", "std_intensity", "median_intensity", "iqr_over_median",
            "cv", "skewness", "kurtosis_excess", "integrated_signal")
    if a.size < 2:
        return dict.fromkeys(keys, np.nan), dict.fromkeys(keys, "too_few_mask_pixels")
    mean, median, sd = float(a.mean()), float(np.median(a)), float(a.std(ddof=1))
    variation = np.ptp(a) > 0
    values = {
        "mean_intensity": mean, "std_intensity": sd, "median_intensity": median,
        "iqr_over_median": float(stats.iqr(a)/median) if median > 0 else np.nan,
        "cv": sd/mean if mean > 0 else np.nan,
        "skewness": float(stats.skew(a, bias=False)) if variation and a.size >= 3 else np.nan,
        "kurtosis_excess": float(stats.kurtosis(a, bias=False)) if variation and a.size >= 4 else np.nan,
        "integrated_signal": float(a.sum()),
    }
    return values, {k: "undefined_denominator_or_constant_intensity"
                    for k, v in values.items() if not np.isfinite(v)}


def extended_abeta_geometry(mask, spacing, config):
    """Pixel-cell morphology/topology; no inferred fibrils, volume, or subpixel detail."""
    return _core.extended_abeta_geometry(mask, spacing, config)


def extended_abeta_intensity(signed, mask, spacing, config):
    """Additional histogram and masked spatial organization descriptors, Aβ only."""
    return _core.extended_abeta_intensity(signed, mask, spacing, config)


def compute_abeta_features(raw_a, signed_a, mask_a, spacing, config, *,
                           effective_bits=None, metadata_bits=None, truncated=False):
    """Return (features, feature_qc rows, GLCM direction rows, saturation) for X_A.

    Only Aβ arrays enter this function; LAMP1 cannot alter X_A.
    """
    return _core.compute_abeta_features(raw_a, signed_a, mask_a, spacing, config, effective_bits=effective_bits, metadata_bits=metadata_bits, truncated=truncated)

