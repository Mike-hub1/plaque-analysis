"""Figure 3: geometry."""
from __future__ import annotations

from plaque_analysis._native import fluorescence_geometry as _core


def signed_boundary_distance(mask, spacing):
    """Signed distance (μm) of each pixel centre to the object boundary.

    spacing is (y_um, x_um). Inside < 0, outside > 0. An empty mask returns NaN.
    """
    return _core.signed_boundary_distance(mask, spacing)


def touches_unobserved(mask, valid):
    """True when the object touches an unobserved pixel or the image edge."""
    return _core.touches_unobserved(mask, valid)


def build_windows(instance_labels, valid_a, spacing, radius_um):
    """Return {object_id: geometry} for the fixed analysis window of each Aβ object.

    geometry holds distance, distance_extended, expected, expected_extended, pad,
    crop, object_truncated and ambiguous_pixels_near_object.
    """
    return _core.build_windows(instance_labels, valid_a, spacing, radius_um)


def object_morphology(mask, spacing, *, min_pixels=9, truncated=False):
    """Return (values, reasons) for the shape features of one object."""
    return _core.object_morphology(mask, spacing, min_pixels=min_pixels, truncated=truncated)


def compute_spatial_statistics(*_args, **_kwargs):
    """Legacy entry; disabled."""
    return _core.compute_spatial_statistics(*_args, **_kwargs)


def compute_fractal_and_morphology(*_args, **_kwargs):
    """Legacy entry; disabled."""
    return _core.compute_fractal_and_morphology(*_args, **_kwargs)
