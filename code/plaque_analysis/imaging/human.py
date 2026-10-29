"""Optical reconstruction and boundary-based LAMP1 intensity measurements."""

from plaque_analysis.imaging.reconstruction import human_optical_parameters as _optical_parameters
from plaque_analysis._native import human_measurements as _core


def optical_parameters(stack, angles_deg, *, phi_sign=1, phi_offset_deg=0, max_se_deg=10):
    """Return Figure 4 optical maps (m2, phi, phi_valid) for one ROI stack.

    phi is in degrees modulo 180° under the instrument convention; phi_valid
    marks directions whose standard error is at most max_se_deg.
    """
    return _optical_parameters(stack, angles_deg, phi_sign=phi_sign,
                               phi_offset_deg=phi_offset_deg, max_se_deg=max_se_deg)


def mask_touches_field(mask):
    """True when the plaque mask touches the ROI edge."""
    return _core.mask_touches_field(mask)


def exterior_distance(mask, spacing_xy):
    """Return (filled envelope, distance in μm to its boundary); spacing is (x_um, y_um)."""
    return _core.exterior_distance(mask, spacing_xy)


def lamp1_measurements(lamp1, mask, valid, spacing_xy):
    """Return inside and exterior LAMP1 intensity rows relative to a plaque-local reference.

    lamp1 must already be background corrected; spacing is (x_um, y_um).
    """
    return _core.lamp1_measurements(lamp1, mask, valid, spacing_xy)
