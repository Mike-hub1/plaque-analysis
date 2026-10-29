"""Figure 4: analysis."""
from __future__ import annotations

from skimage.measure import perimeter_crofton
from skimage.measure import regionprops
import numpy as np
from plaque_analysis.imaging.orientation import axial_summary, boundary_depth_profile, centroid_texture_coherence, distribution, first_half_decay, local_axial, nematic_correlation, scale_profile
from plaque_analysis.imaging.photoacoustic import pa_grid_statistics
from plaque_analysis.imaging.reconstruction import odd_window


def orientation_mask(maps, tissue, max_se):
    return (tissue & np.isfinite(maps["phi_deg"]) & np.isfinite(maps["phi_formal_se_deg"])
            & (maps["phi_formal_se_deg"] <= max_se) & (maps["M2"] > 0))


def interpolate_curve(correlation, key, at_um):
    r, y = correlation["distance_um"], correlation[key]
    good = np.isfinite(r) & np.isfinite(y)
    if good.sum() < 2 or not r[good].min() <= at_um <= r[good].max():
        return np.nan
    return float(np.interp(at_um, r[good], y[good]))


def analyse_optical(maps, tissue, config, *, fixed_orientation=None):
    pixel = config["pixel_size_um"]
    valid = (orientation_mask(maps, tissue, config["orientation_max_formal_se_deg"])
             if fixed_orientation is None else fixed_orientation & np.isfinite(maps["phi_deg"]))
    if valid.sum() < 20:
        raise ValueError("Fewer than 20 reliable orientation pixels; inspect acquisition and quality thresholds.")
    window = odd_window(config["local_window_um"], pixel)
    local_s, local_phi, support = local_axial(maps["phi_deg"], valid, window,
                                             config["minimum_window_coverage"])
    scale, fixed_scale_mask = scale_profile(maps["phi_deg"], valid, pixel,
                                            config["scale_windows_um"],
                                            config["minimum_window_coverage"])
    if not fixed_scale_mask.any():
        raise ValueError("No common support for all scale windows; reduce the largest configured window.")
    correlation = nematic_correlation(maps["phi_deg"], valid, pixel,
                                      config["correlation_bin_um"], config["correlation_max_um"])
    half, half_status = first_half_decay(correlation)
    depth, depth_map, depth_labels = boundary_depth_profile(maps["M2"], tissue, pixel,
                                                           config["depth_quantiles"])
    angle_stats = axial_summary(maps["phi_deg"], valid, config["orientation_bins"])
    hist, edges = np.histogram(maps["phi_deg"][valid] % 180,
                               bins=config["orientation_bins"], range=(0, 180))
    m2 = distribution(maps["M2"][tissue])
    m4 = distribution(maps["M4_reference"][tissue])
    s_stats = distribution(local_s[valid])
    local_spread = np.full_like(local_s, np.nan)
    positive_s = np.isfinite(local_s) & (local_s > 0)
    local_spread[positive_s] = np.rad2deg(np.sqrt(-2*np.log(local_s[positive_s])))/2
    centroid_coherence, centroid_n = centroid_texture_coherence(
        maps["phi_deg"], valid, tissue, pixel, return_count=True)
    ratio = depth[-1]["median"]/depth[0]["median"] if depth and depth[0]["median"] > 0 else np.nan
    values = {
        "optical_m2_median": m2["median"],
        "optical_m2_quartile_dispersion": m2["quartile_dispersion"],
        "optical_m2_core_rim_ratio": float(ratio),
        "orientation_S_global": angle_stats["S_global"],
        "orientation_entropy": angle_stats["angular_entropy"],
        "orientation_mean_axis_deg": angle_stats["mean_axis_deg"],
        "orientation_local_S_median": s_stats["median"],
        "orientation_local_S_iqr": s_stats["iqr"],
        "orientation_local_spread_median_deg": float(np.nanmedian(local_spread)),
        "orientation_C2_at_1um": interpolate_curve(correlation, "C2", 1),
        "orientation_half_decay_um": half,
        "orientation_centroid_texture_coherence": centroid_coherence,
        "optical_M4_reference_median": m4["median"],
    }
    shape = regionprops(tissue.astype(np.uint8))[0]
    values.update({"morphology_area_um2": tissue.sum()*pixel**2,
                   "morphology_eccentricity": float(shape.eccentricity),
                   "morphology_solidity": float(shape.solidity),
                   "morphology_perimeter_um": float(perimeter_crofton(tissue, directions=4)*pixel)})
    q2_valid = tissue & np.isfinite(maps["M2"]) & (maps["M2"] >= 0) & (maps["M2"] <= 1)
    delta = np.full(tissue.shape, np.nan)
    delta[q2_valid] = np.arcsin(maps["M2"][q2_valid])
    opd = delta * config["wavelength_nm"]/(2*np.pi)
    dn = opd/(config["section_thickness_um"]*1000)
    values["conditional_OPD_median_nm"] = float(np.nanmedian(opd[tissue]))
    values["conditional_delta_n_median"] = float(np.nanmedian(dn[tissue]))
    return {
        "values": values, "valid_orientation": valid, "S_local": local_s,
        "local_phi_deg": local_phi, "window_support": support, "window_px": window,
        "window_actual_um": window*pixel, "scale_rows": scale,
        "scale_common_mask": fixed_scale_mask, "correlation": correlation,
        "half_decay_status": half_status, "depth_rows": depth, "depth_map_um": depth_map,
        "depth_labels": depth_labels, "angle_count": hist, "angle_edges_deg": edges,
        "angle_summary": angle_stats, "M2_summary": m2, "M4_summary": m4,
        "local_S_summary": s_stats, "conditional_OPD_nm": opd, "conditional_delta_n": dn,
        "local_angular_spread_deg": local_spread,
        "centroid_texture_support_n": centroid_n,
        "invalid_arcsin_pixel_n": int(np.sum(tissue & ~q2_valid)),
    }


def feature_records(analysis, pa, maps, tissue, config):
    """Curated continuous features; QC and conditional physical conversions stay identifiable."""
    values = dict(analysis["values"])
    pa_stats = pa_grid_statistics(pa)
    values.update({"pa_doa_median": pa_stats["median"],
                   "pa_doa_quartile_dispersion": pa_stats["quartile_dispersion"],
                   "pa_doa_iqr": pa_stats["iqr"], "pa_doa_q95": pa_stats["q95"],
                   "pa_adjacent_abs_difference_mean": pa_stats["adjacent_abs_difference_mean"]})
    optical_n, orientation_n = int(tissue.sum()), int(analysis["valid_orientation"].sum())
    local_n = int(np.isfinite(analysis["S_local"]).sum())
    definitions = [
        ("optical_m2_median", "Optical response", "Median M2", "1", optical_n, True,
         "Median of sqrt(A2²+B2²)/A0 over the A0 tissue mask; zeros retained."),
        ("optical_m2_quartile_dispersion", "Optical response", "M2 quartile dispersion", "1", optical_n, True,
         "(Q75-Q25)/(Q75+Q25) of tissue M2; undefined for a zero denominator."),
        ("optical_m2_core_rim_ratio", "Optical response", "Core/rim M2 ratio", "1", optical_n, True,
         "Deepest / shallowest depth-quantile median M2; depth is nearest-boundary distance, not distance from a circle."),
        ("orientation_S_global", "Orientation", "Global axial order", "1", orientation_n, True,
         "Magnitude of mean exp(2i phi), with equal weights over reliable directions."),
        ("orientation_entropy", "Orientation", "Axial angular entropy", "1", orientation_n, True,
         f"Shannon entropy of {config['orientation_bins']} fixed equal axial bins, divided by log(bin count)."),
        ("orientation_local_S_median", "Orientation", "Median local axial order", "1", local_n, True,
         f"Median mask-normalized S at {analysis['window_actual_um']:.6g} µm; window coverage >= {config['minimum_window_coverage']}."),
        ("orientation_local_S_iqr", "Orientation", "Local order heterogeneity", "1", local_n, True,
         "Interquartile range of the supported local axial-order map."),
        ("orientation_local_spread_median_deg", "Orientation", "Local circular angular spread", "degree", local_n, True,
         "Median of sqrt(-2 log(S_local))/2 in degrees; not a confidence interval or the fibre angular FWHM."),
        ("orientation_C2_at_1um", "Orientation", "C2 at 1 µm", "1", orientation_n, True,
         "Pair-weighted cos(2 angular difference), interpolated at 1 µm from radial lag bins; not a cross-modality correlation."),
        ("orientation_half_decay_um", "Orientation", "Connected half-decay distance", "µm", orientation_n, True,
         "First downward crossing of G2=(C2-S_global²)/(1-S_global²) through 0.5; missing if unresolved or not reached."),
        ("orientation_centroid_texture_coherence", "Orientation", "Centroid texture coherence", "1", analysis["centroid_texture_support_n"], True,
         "Magnitude of mean exp[2i(phi-azimuth from tissue centroid)], excluding radius <0.25 µm; cannot distinguish radial from tangential alignment."),
        ("orientation_mean_axis_deg", "Conditional axis", "Mean optical axis", "degree", orientation_n,
         config["axis_calibration_confirmed"], "Axial circular mean under the configured optical sign and zero offset."),
        ("optical_M4_reference_median", "Optical reference", "Median M4 reference", "1", optical_n, False,
         "Fourth-harmonic amplitude/A0; retained as a diagnostic, not calibrated LD or PA DOA."),
        ("conditional_OPD_median_nm", "Conditional conversion", "Apparent OPD", "nm", optical_n,
         config["wavelength_thickness_confirmed"], "Median arcsin(M2)*lambda/(2pi); conditional single-retarder, low-retardance model."),
        ("conditional_delta_n_median", "Conditional conversion", "Effective birefringence", "1", optical_n,
         config["wavelength_thickness_confirmed"], "Apparent OPD divided by section thickness; not intrinsic fibre birefringence."),
        ("morphology_area_um2", "Morphology", "Optical mask area", "µm²", optical_n, True,
         "Tissue pixel count times optical pixel area; segmentation and calibration dependent."),
        ("morphology_eccentricity", "Morphology", "Mask eccentricity", "1", optical_n, True,
         "Eccentricity of the ellipse with the same second moments as the tissue mask."),
        ("morphology_solidity", "Morphology", "Mask solidity", "1", optical_n, True,
         "Tissue mask area / convex hull area."),
        ("morphology_perimeter_um", "Morphology", "Mask perimeter", "µm", optical_n, False,
         "Four-direction Crofton perimeter; sensitive to segmentation and imaging resolution."),
        ("pa_doa_median", "PA anisotropy", "Median PA DOA", "1", pa_stats["n"], True,
         "Median of all finite native PA values; no interpolation or optical masking."),
        ("pa_doa_quartile_dispersion", "PA anisotropy", "PA quartile dispersion", "1", pa_stats["n"], True,
         "(Q75-Q25)/(Q75+Q25) of all finite native PA DOA values."),
        ("pa_doa_iqr", "PA anisotropy", "PA DOA IQR", "1", pa_stats["n"], True,
         "Q75-Q25 of native PA DOA."),
        ("pa_doa_q95", "PA anisotropy", "PA DOA 95th percentile", "1", pa_stats["n"], True,
         "95th percentile of native PA DOA; descriptive upper-tail strength."),
        ("pa_adjacent_abs_difference_mean", "PA anisotropy", "Adjacent PA contrast", "1", pa_stats["adjacent_pair_n"], True,
         "Mean absolute difference over all finite horizontal and vertical native-neighbour pairs, each pair once."),
    ]
    records = []
    for key, family, label, unit, n, default, definition in definitions:
        value = values[key]
        status = "available" if np.isfinite(value) else "undefined"
        if key == "orientation_half_decay_um":
            status = analysis["half_decay_status"]
        if family.startswith("Conditional") and not default:
            status = "conditional_uncalibrated"
        records.append({"sample_id": config["sample_id"], "feature": key, "family": family,
                        "label": label, "value": value, "unit": unit, "support_n": n,
                        "default_model_candidate": bool(default), "status": status,
                        "definition": definition})
    qc = {"optical_tissue_n": optical_n, "orientation_valid_n": orientation_n,
          "orientation_rejected_n": optical_n-orientation_n,
          "orientation_valid_fraction": orientation_n/optical_n,
          "local_order_supported_n": local_n,
          "multiscale_common_n": int(analysis["scale_common_mask"].sum()),
          "PA_finite_n": pa_stats["n"], "PA_nan_n": int(np.isnan(pa).sum()),
          "fit_R2_median": float(np.nanmedian(maps["fit_R2"][tissue])),
          "angle_holdout_R2_median": float(np.nanmedian(maps["angle_holdout_R2"][tissue])),
          "formal_axis_se_median_deg": float(np.nanmedian(maps["phi_formal_se_deg"][tissue])),
          "M2_above_one_n": int(np.sum(tissue & (maps["M2"] > 1))),
          "PA_statistics": pa_stats}
    return records, qc
