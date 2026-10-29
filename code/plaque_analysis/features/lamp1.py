"""Figure 3: lamp1 features."""
from __future__ import annotations

from plaque_analysis._native import lamp1 as _core


def boundary_profiles(signed_l, mask_l, valid, geometry, spacing, ring_width, radius):
    """Return one row per boundary-distance ring of the analysis window."""
    return _core.boundary_profiles(signed_l, mask_l, valid, geometry, spacing, ring_width, radius)


def angular_profiles(mask_a, mask_l, valid, geometry, spacing, sectors):
    """Return (sector rows, angular coverage, angular asymmetry)."""
    return _core.angular_profiles(mask_a, mask_l, valid, geometry, spacing, sectors)


def radial_angular_cells(mask_a, mask_l, window, geometry, profiles, spacing, sectors):
    """Return one row per ring × sector cell, using the rings in profiles."""
    if not hasattr(_core, "radial_angular_cells"):
        raise RuntimeError("Figure 3 radial × angular cells require a plaque-analysis-core release "
                           "that provides radial_angular_cells; request the updated core package.")
    return _core.radial_angular_cells(mask_a, mask_l, window, geometry, profiles, spacing, sectors)


def extended_lamp1_features(signed, mask_l, mask_a, window, geometry, spacing, config, angular):
    """Return (values, reasons) for fixed-window LAMP1 readouts; patches are image patches, NOT organelles."""
    return _core.extended_lamp1_features(signed, mask_l, mask_a, window, geometry, spacing, config, angular)


def compute_lamp1_outcomes(signed_l, mask_l, mask_a, valid, geometry, spacing, config):
    """Return (Y_L values, feature_qc rows, ring profiles, sector rows, observed window)."""
    return _core.compute_lamp1_outcomes(signed_l, mask_l, mask_a, valid, geometry, spacing, config)
