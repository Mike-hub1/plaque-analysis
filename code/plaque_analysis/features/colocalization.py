"""Figure 3: relations."""
from __future__ import annotations

from plaque_analysis._native import colocalization as _core

from scipy import stats
import numpy as np


def safe_pearson(a, b, domain):
    a, b = np.asarray(a)[domain], np.asarray(b)[domain]
    if len(a) < 3:
        return np.nan, "too_few_window_pixels"
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, "zero_variance"
    return float(stats.pearsonr(a, b).statistic), ""


def compute_colocalization(signed_a, signed_l, mask_a, mask_l, window):
    """Return (values, reasons) for the Z_AL window relations of one object."""
    return _core.compute_colocalization(signed_a, signed_l, mask_a, mask_l, window)


def extended_joint_features(signed_a, signed_l, mask_a, mask_l, window, spacing):
    """Return (values, reasons); joint descriptors, not predictors of the same LAMP1 readout and not tests."""
    return _core.extended_joint_features(signed_a, signed_l, mask_a, mask_l, window, spacing)


def neighborhood_composition(ranks_a, ranks_l, window, n_l):
    """Return LAMP1 state composition rows around each Aβ intensity stratum; state 0 is LAMP1-negative."""
    return _core.neighborhood_composition(ranks_a, ranks_l, window, n_l)

