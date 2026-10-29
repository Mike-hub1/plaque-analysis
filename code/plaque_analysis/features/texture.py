"""Figure 3: texture."""
from __future__ import annotations

from plaque_analysis._native import texture as _core


def rank_intensity_labels(labels, intensity, mask=None):
    """Return intensity ranks per label; rank 1 is the brightest."""
    return _core.rank_intensity_labels(labels, intensity, mask)


def intensity_strata(intensity, mask, n_clusters=5, random_state=42):
    """Return (ranks, strata rows) for at most n_clusters intensity levels; rank 1 is brightest."""
    return _core.intensity_strata(intensity, mask, n_clusters, random_state)


def compute_glcm_texture(intensity, mask, spacing, *, levels=8, bin_edges=None,
                         offsets=((1, 0), (0, 1), (1, 1), (-1, 1)), min_pairs=16):
    """Return one-pixel-lag GLCM features of one object with per-direction rows.

    offsets are (dx, dy) in native pixels; spacing is (y_um, x_um).
    """
    return _core.compute_glcm_texture(intensity, mask, spacing, levels=levels, bin_edges=bin_edges, offsets=offsets, min_pairs=min_pairs)


def multiscale_abeta_texture(signed, mask, spacing, config):
    """Return two- and three-pixel-lag GLCM features with per-direction rows."""
    return _core.multiscale_abeta_texture(signed, mask, spacing, config)
