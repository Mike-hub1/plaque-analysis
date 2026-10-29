"""Feature families emitted from full populations and technical scenarios."""
import numpy as np
from plaque_analysis.features.polarization_schema import FEATURE_METRICS, FEATURE_NAMES_ZH


def add_curated(add, features):
    for row in features:
        metric, statistic = FEATURE_METRICS.get(row["feature"], (row["feature"], "value"))
        conditional = row["family"].startswith("Conditional")
        reason = ("Absolute sign/zero-angle calibration is unconfirmed." if row["family"] == "Conditional axis"
                  else "Wavelength/thickness calibration is unconfirmed; single-retarder low-retardance model is conditional.") if row["status"] == "conditional_uncalibrated" else ""
        add(row["feature"], row["label"], row["value"], row["unit"], family=row["family"],
            metric=metric, statistic=statistic, support_n=row["support_n"],
            sampling_unit=("PA_adjacent_pair" if metric == "PA_pair_abs_difference" else
                           "PA_scan_point" if metric == "PA_DOA" else
                           "optical_window_centre" if metric in ("local_S", "angular_spread_deg") else "optical_pixel"),
            role="conditional" if conditional else "reference" if metric == "M4_reference" else "descriptor",
            status=row["status"], reason=reason, definition=row["definition"],
            feature_name_zh=FEATURE_NAMES_ZH[row["feature"]], curated=True,
            default_model_candidate=row["default_model_candidate"], source_data="features/plaque_features_long.csv",
            source_field=row["feature"])


def add_global_distributions(add, add_distribution, maps, tissue, analysis, pa, visual, config):
    pairs = np.array([p["absolute_DOA_difference"] for p in visual["PA_pairs"]])
    metric_values = [
        ("M2", "Second-harmonic response", "二阶调制幅值", maps["M2"][tissue], "1", "Optical response", "optical_pixel"),
        ("local_S", "Local axial order", "局部轴向有序度", analysis["S_local"], "1", "Orientation", "optical_window_centre"),
        ("angular_spread_deg", "Local angular spread", "局部角度离散度", analysis["local_angular_spread_deg"], "degree", "Orientation", "optical_window_centre"),
        ("M4_reference", "Fourth-harmonic reference", "四阶调制参考量", maps["M4_reference"][tissue], "1", "Optical reference", "optical_pixel"),
        ("PA_DOA", "Native PA DOA", "原生光声DOA", pa, "1", "PA anisotropy", "PA_scan_point"),
        ("PA_pair_abs_difference", "Adjacent-pair absolute DOA difference", "邻接对DOA绝对差", pairs, "1", "PA anisotropy", "PA_adjacent_pair"),
        ("PA_point_neighbour_mean", "Mean neighbour absolute DOA difference", "格点平均邻域DOA绝对差", visual["PA_neighbour_mean"], "1", "PA anisotropy", "PA_scan_point_with_neighbour"),
    ]
    for metric, label, zh, values, unit, family, sample_unit in metric_values:
        add_distribution(metric, label, values, unit=unit, family=family, feature_name_zh=zh,
                         sampling_unit=sample_unit, role="reference" if metric == "M4_reference" else "descriptor",
                         source_data="source_data/optical_native_ROI.npz" if not metric.startswith("PA_") else
                                     "source_data/panel_i_PA_neighbour_pairs_display_coordinates.csv" if sample_unit == "PA_adjacent_pair" else
                                     "source_data/panel_i_PA_cells.csv",
                         source_field={"angular_spread_deg": "local_angular_spread_deg", "PA_DOA": "DOA",
                                       "PA_pair_abs_difference": "absolute_DOA_difference",
                                       "PA_point_neighbour_mean": "neighbour_abs_difference_mean"}.get(metric, metric),
                         notes="The unweighted mean over point-wise neighbour means differs from the mean over unique pairs."
                               if metric == "PA_point_neighbour_mean" else "")
    for metric, values, unit in [("OPD_conditional", analysis["conditional_OPD_nm"], "nm"),
                                 ("delta_n_conditional", analysis["conditional_delta_n"], "1")]:
        confirmed = config["wavelength_thickness_confirmed"]
        add_distribution(metric, metric, values, unit=unit, family="Conditional conversion", role="conditional",
                         status="available" if confirmed else "conditional_uncalibrated",
                         reason="" if confirmed else "Unconfirmed wavelength/thickness; conditional retarder model.",
                         source_data="maps/"+("OPD_conditional_nm.npy" if metric == "OPD_conditional" else "delta_n_conditional.npy"),
                         source_field="finite array entries in the tissue mask")
    add("orientation_circular_std_deg", "Global axial circular spread", analysis["angle_summary"]["circular_std_deg"],
        "degree", family="Orientation", support_n=analysis["angle_summary"]["n"],
        feature_name_zh="全局轴向圆标准差", definition="sqrt(-2 ln(S_global))/2, converted to degrees.",
        source_data="maps/phi_deg.npy", source_field="phi_deg")


def add_spatial_profiles(add, add_distribution, analysis, visual, config):
    for index, group in enumerate(visual["depth_groups"]):
        for metric, group_key, unit, zh in [("M2", "M2", "1", "二阶调制幅值"),
                                            ("local_S", "S", "1", "局部轴向有序度"),
                                            ("angular_spread_deg", "spread", "degree", "局部角度离散度")]:
            add_distribution(metric, metric, group[group_key], unit=unit, family="Depth profile", scope="depth",
                             depth_region=index+1, feature_name_zh=zh,
                             sampling_unit="optical_pixel" if metric == "M2" else "optical_window_centre",
                             source_data="source_data/panel_f_region_distributions.csv", source_field=metric)
    for region in analysis["depth_rows"]:
        for key in ("depth_low_um", "depth_high_um", "depth_median_um"):
            add(key, key, region[key], "µm", family="Depth geometry", scope="depth_geometry",
                depth_region=region["depth_bin"], support_n=region["n"],
                definition="Nearest-boundary distance within the two-dimensional optical mask.",
                source_data="source_data/panel_e_depth_regions.csv", source_field=key)
    for cell in visual["spatial"]["rows"]:
        for metric, prefix in (("M2", "M2"), ("local_S", "local_S")):
            supported = cell[prefix+"_displayed"]
            add(metric+"_median", metric+" median", cell[prefix+"_median_unmasked"],
                family="Depth–azimuth", metric=metric, statistic="median", scope="depth_azimuth",
                depth_region=cell["depth_region"], azimuth_sector=cell["azimuth_sector"],
                azimuth_start_deg=cell["azimuth_start_deg"], azimuth_end_deg=cell["azimuth_end_deg"],
                support_n=cell[prefix+"_n"], sampling_unit="optical_pixel" if metric == "M2" else "optical_window_centre",
                status="available" if supported else "insufficient_support",
                reason="" if supported else f"Fewer than {config['spatial_cell_min_pixels']} supported observations in this cell.",
                definition="Median in the geometry-defined depth/azimuth cell; 0° right, clockwise azimuth.",
                source_data="source_data/panel_g_depth_azimuth_cells.csv", source_field=prefix+"_median_unmasked")
    for s, values in zip(analysis["scale_rows"], visual["scale_values"]):
        add_distribution("local_S", "Local axial order", values, family="Multiscale", scope="scale",
                         window_px=s["window_px"], window_um=s["window_um"], feature_name_zh="共同中心局部轴向有序度",
                         sampling_unit="shared_optical_window_centre", source_data="source_data/panel_h_multiscale_order.csv",
                         source_field="local_S", notes="Identical common centres at every window; no sampling in statistics.")


def add_correlations(add, analysis, loo_rows, loo_curves):
    corr = analysis["correlation"]
    for j in range(len(corr["C2"])):
        dims = {"lag_bin": j+1, "lag_um": float(corr["distance_um"][j]),
                "bin_left": float(corr["bin_left_um"][j]), "bin_right": float(corr["bin_right_um"][j])}
        for metric in ("C2", "G2_connected"):
            add(metric, metric, corr[metric][j], family="Spatial correlation", scope="correlation", metric=metric,
                support_n=corr["ordered_pair_count"][j], sampling_unit="ordered_optical_pixel_pair",
                definition="Pair-weighted mean cos(2Δφ)." if metric == "C2" else "(C2-S_global²)/(1-S_global²).",
                source_data="source_data/spatial_correlation.csv", source_field=metric, **dims)
        for k, fold in enumerate(loo_rows):
            add("C2", "Leave-one-angle C2", loo_curves[k, j], family="Spatial correlation", scope="correlation",
                metric="C2", role="sensitivity", scenario=f"omit_angle_{k+1:02d}", omitted_angle_deg=fold["omitted_angle_deg"],
                support_n=corr["ordered_pair_count"][j], sampling_unit="ordered_optical_pixel_pair",
                definition="C2 after omitting one acquisition frame; tissue and direction masks frozen at baseline.",
                source_data="source_data/leave_one_angle_curves.npz", source_field="C2", **dims)


def add_histograms(add, analysis, visual):
    for j, n in enumerate(analysis["angle_count"]):
        add("axial_histogram_fraction", "Axial orientation fraction", n/analysis["angle_count"].sum(),
            family="Orientation", scope="orientation_histogram", metric="axial_histogram", statistic="fraction",
            bin_left=float(analysis["angle_edges_deg"][j]), bin_right=float(analysis["angle_edges_deg"][j+1]),
            support_n=int(analysis["angle_count"].sum()), bin_count=int(n),
            definition="Fraction of reliable axial directions in fixed 0–180° bins; not position azimuth.",
            source_data="source_data/orientation_histogram.csv", source_field="fraction")
    hist = visual["PA_distribution"]
    for j, n in enumerate(hist["counts"]):
        add("PA_DOA_histogram_percent", "Native DOA histogram", hist["percent"][j], "%",
            family="PA anisotropy", scope="pa_histogram", metric="PA_DOA", statistic="bin_percent",
            bin_left=float(hist["edges"][j]), bin_right=float(hist["edges"][j+1]),
            bin_count=int(n), support_n=hist["summary"]["support_n"], sampling_unit="PA_scan_point",
            definition="100 × bin count / all finite native PA scan points; all finite values included.",
            source_data="source_data/panel_i_DOA_histogram.csv", source_field="percent")


def add_quality(add, add_distribution, qc, maps, tissue):
    for key, value in qc.items():
        if isinstance(value, dict):
            continue
        add(key, key, value, "count" if key.endswith("_n") else "degree" if key.endswith("_deg") else "1",
            family="Quality control", scope="qc", role="qc", support_n=1, sampling_unit="image_summary",
            definition="Support count, support fraction or reconstruction quality diagnostic; not a biological replicate.",
            source_data="quality_control/analysis_report.json", source_field="quality."+key)
    for key, unit in [("A0", "gray level"), ("fit_R2", "1"), ("angle_holdout_R2", "1"),
                      ("RMSE", "gray level"), ("phi_formal_se_deg", "degree")]:
        add_distribution(key, key, maps[key][tissue], unit=unit, family="Quality control", scope="qc_distribution",
                         role="qc", source_data="maps/"+key+".npy", source_field=key)


def add_technical_scenarios(add, features, analysis, loo_rows, loo_scales, threshold_rows, boundary_rows, angle_rows):
    lookup = {f["feature"]: f for f in features}
    for k, fold in enumerate(loo_rows):
        for key, value in fold.items():
            if key not in lookup:
                continue
            ref = lookup[key]
            metric, stat = FEATURE_METRICS.get(key, (key, "value"))
            add(key, ref["label"], value, ref["unit"], family=ref["family"], metric=metric, statistic=stat,
                scope="global", role="sensitivity", scenario=f"omit_angle_{k+1:02d}",
                omitted_angle_deg=fold["omitted_angle_deg"], support_n=ref["support_n"],
                sampling_unit="optical_pixel" if metric not in ("local_S", "angular_spread_deg") else "optical_window_centre",
                status="conditional_uncalibrated" if ref["status"] == "conditional_uncalibrated" else "available",
                reason="Calibration unconfirmed; perturbation value is diagnostic only." if ref["status"] == "conditional_uncalibrated" else "",
                definition=ref["definition"], feature_name_zh=FEATURE_NAMES_ZH[key],
                source_data="source_data/leave_one_angle_features.csv", source_field=key,
                notes="One frame omitted; baseline tissue and direction masks held fixed. Technical sensitivity, not a confidence interval.")
        for j, s in enumerate(analysis["scale_rows"]):
            add("local_S_median", "Leave-one-angle scale median", loo_scales[k, j], family="Multiscale",
                metric="local_S", statistic="median", scope="scale", role="sensitivity", scenario=f"omit_angle_{k+1:02d}",
                omitted_angle_deg=fold["omitted_angle_deg"], window_px=s["window_px"], window_um=s["window_um"],
                support_n=s["n"], sampling_unit="shared_optical_window_centre",
                definition="Shared-centre S median after omitting one acquisition frame; baseline masks frozen.",
                source_data="source_data/leave_one_angle_curves.npz", source_field="local_S_median")
    for row in threshold_rows:
        for key in ("valid_pixel_n", "valid_fraction", "S_global", "angular_entropy", "local_S_median"):
            add(key, key, row[key], "count" if key == "valid_pixel_n" else "1", family="Threshold sensitivity",
                scope="threshold_sensitivity", role="sensitivity", threshold_deg=row["formal_se_threshold_deg"],
                scenario=f"axis_SE_{row['formal_se_threshold_deg']:g}", support_n=row["valid_pixel_n"],
                sampling_unit="orientation_support_summary",
                definition="Direction formal-SE threshold changed; local window coverage rule retained. support_n counts reliable directions, not supported local windows.",
                source_data="source_data/orientation_quality_sensitivity.csv", source_field=key)
    for row in boundary_rows:
        for key in ("tissue_pixel_n", "orientation_pixel_n", "M2_median", "M2_quartile_dispersion", "S_global", "core_rim_ratio"):
            add(key, key, row[key], "count" if key.endswith("_n") else "1", family="Boundary sensitivity",
                scope="boundary_sensitivity", role="sensitivity", scenario="boundary_"+row["variant"],
                boundary_radius_px=row["boundary_radius_px"], boundary_radius_um=row["boundary_radius_um"],
                support_n=row["orientation_pixel_n"] if key == "S_global" else row["tissue_pixel_n"],
                definition="Disk erosion/original mask/dilation; recompute statistics without brightness changes.",
                source_data="source_data/boundary_mask_sensitivity.csv", source_field=key)
    for row in angle_rows:
        for key, value in row.items():
            if not key.endswith("pct_A0"):
                continue
            add(key, key, value, "%", family="Prediction quality", scope="angle_prediction", role="qc",
                acquisition_angle_deg=row["angle_deg"], support_n=row["tissue_n"],
                definition="Absolute held-out intensity prediction error / A0 × 100; model encoded in feature name.",
                source_data="source_data/angle_prediction_validation.csv", source_field=key)

