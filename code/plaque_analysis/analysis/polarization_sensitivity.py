"""Figure 4: sensitivity."""
from __future__ import annotations

from scipy import ndimage
import numpy as np
from plaque_analysis.analysis.polarization import analyse_optical, orientation_mask
from plaque_analysis.imaging.orientation import axial_summary, boundary_depth_profile, distribution, local_axial
from plaque_analysis.imaging.reconstruction import fit_harmonics, odd_window


def leave_angle_sensitivity(stack, angles, tissue, config, baseline, progress=None):
    """Each frame is omitted once; ranges describe technical perturbation, not CIs."""
    rows, curves, scale_curves = [], [], []
    for k, angle in enumerate(angles):
        keep = np.arange(len(angles)) != k
        maps, _, _ = fit_harmonics(stack[keep], np.asarray(angles)[keep],
                                   phi_sign=config["phi_sign"], phi_offset_deg=config["phi_offset_deg"])
        fold = analyse_optical(maps, tissue, config,
                               fixed_orientation=baseline["valid_orientation"])
        rows.append({"omitted_file": f"{k+1}.tif", "omitted_angle_deg": float(angle),
                     **fold["values"]})
        curves.append(fold["correlation"]["C2"])
        scale_curves.append([r["median"] for r in fold["scale_rows"]])
        if progress:
            progress(k+1, len(angles))
    return rows, np.asarray(curves), np.asarray(scale_curves)


def threshold_sensitivity(maps, tissue, config):
    rows = []
    for threshold in config["orientation_sensitivity_se_deg"]:
        valid = orientation_mask(maps, tissue, threshold)
        order, _, _ = local_axial(maps["phi_deg"], valid,
                                  odd_window(config["local_window_um"], config["pixel_size_um"]),
                                  config["minimum_window_coverage"])
        stats = axial_summary(maps["phi_deg"], valid, config["orientation_bins"])
        rows.append({"formal_se_threshold_deg": threshold, "valid_pixel_n": int(valid.sum()),
                     "valid_fraction": float(valid.sum()/tissue.sum()),
                     "S_global": stats["S_global"], "angular_entropy": stats["angular_entropy"],
                     "local_S_median": float(np.nanmedian(order)) if np.isfinite(order).any() else np.nan})
    return rows


def boundary_sensitivity(maps, tissue, config, radius_px=3):
    """Declared mask perturbations; primary results always use the unperturbed mask."""
    yy, xx = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    disk = yy*yy+xx*xx <= radius_px**2
    variants = [("eroded", ndimage.binary_erosion(tissue, structure=disk)),
                ("primary", tissue), ("dilated", ndimage.binary_dilation(tissue, structure=disk))]
    rows = []
    for name, mask in variants:
        valid = orientation_mask(maps, mask, config["orientation_max_formal_se_deg"])
        stats = distribution(maps["M2"][mask])
        angular = axial_summary(maps["phi_deg"], valid, config["orientation_bins"])
        depth, _, _ = boundary_depth_profile(maps["M2"], mask, config["pixel_size_um"],
                                             config["depth_quantiles"])
        rows.append({"variant": name, "boundary_radius_px": 0 if name == "primary" else radius_px,
                     "boundary_radius_um": 0 if name == "primary" else radius_px*config["pixel_size_um"],
                     "tissue_pixel_n": int(mask.sum()), "orientation_pixel_n": int(valid.sum()),
                     "M2_median": stats["median"], "M2_quartile_dispersion": stats["quartile_dispersion"],
                     "S_global": angular["S_global"],
                     "core_rim_ratio": depth[-1]["median"]/depth[0]["median"] if depth[0]["median"] > 0 else np.nan})
    return rows


def angle_validation(heldout2, heldout4, a0, mask, angles):
    rows = []
    valid = mask & np.isfinite(a0) & (a0 > 0)
    for k, angle in enumerate(angles):
        row = {"file": f"{k+1}.tif", "angle_deg": angle, "tissue_n": int(valid.sum())}
        for key, residual in [("second", heldout2), ("second_fourth", heldout4)]:
            errors = np.abs(residual[k][valid])/a0[valid]*100
            row.update({key+"_median_pct_A0": float(np.median(errors)),
                        key+"_q25_pct_A0": float(np.percentile(errors, 25)),
                        key+"_q75_pct_A0": float(np.percentile(errors, 75))})
        rows.append(row)
    return rows
