"""Figure 4: exports."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from plaque_analysis.io.polarization import write_csv
from plaque_analysis.analysis.polarization_visual import histogram_rows


def write_sources(out, stack, maps, coeff, tissue, analysis, pa, pa_display,
                  features, angle_rows, threshold_rows, loo_rows, loo_curves, loo_scales, config):
    source = out/"source_data"
    source.mkdir(exist_ok=True)
    arrays = out/"maps"
    arrays.mkdir(exist_ok=True)
    feature_dir = out/"features"
    feature_dir.mkdir(exist_ok=True)
    for name, value in maps.items():
        np.save(arrays/f"{name}.npy", value)
    np.savez_compressed(source/"optical_native_ROI.npz", raw_intensity=stack,
                        angles_deg=config["angles_deg"], coefficients=coeff,
                        tissue_mask=tissue, orientation_mask=analysis["valid_orientation"],
                        local_S=analysis["S_local"], local_phi_deg=analysis["local_phi_deg"],
                        local_angular_spread_deg=analysis["local_angular_spread_deg"],
                        local_window_support=analysis["window_support"],
                        multiscale_common_mask=analysis["scale_common_mask"],
                        depth_um=analysis["depth_map_um"], depth_bin=analysis["depth_labels"], **maps)
    for name, value in [("tissue_mask", tissue), ("orientation_mask", analysis["valid_orientation"]),
                        ("S_local", analysis["S_local"]), ("DOA_PA_native", pa),
                        ("local_angular_spread_deg", analysis["local_angular_spread_deg"]),
                        ("DOA_PA_display", pa_display), ("OPD_conditional_nm", analysis["conditional_OPD_nm"]),
                        ("delta_n_conditional", analysis["conditional_delta_n"])]:
        np.save(arrays/f"{name}.npy", value)
    rr, cc = np.indices(pa.shape)
    finite = np.isfinite(pa)
    write_csv(source/"PA_native_cells.csv", {"native_row": rr[finite], "native_column": cc[finite],
                                             "DOA": pa[finite],
                                             "x_um_from_grid_origin": cc[finite]*config["pa_native_column_pitch_um"]
                                             if config["pa_native_column_pitch_um"] else np.full(finite.sum(), np.nan),
                                             "y_um_from_grid_origin": rr[finite]*config["pa_native_row_pitch_um"]
                                             if config["pa_native_row_pitch_um"] else np.full(finite.sum(), np.nan)})
    np.savetxt(source/"PA_display_matrix.csv", pa_display, delimiter=",", fmt="%.7f")
    angles = (analysis["angle_edges_deg"][:-1]+analysis["angle_edges_deg"][1:])/2
    counts = analysis["angle_count"]
    write_csv(source/"orientation_histogram.csv",
              {"angle_bin_centre_deg": angles, "pixel_count": counts, "fraction": counts/counts.sum()})
    correlation = analysis["correlation"]
    write_csv(source/"spatial_correlation.csv", {
        key: value for key, value in correlation.items() if key != "baseline_S_squared"} | {
        "baseline_S_squared": np.repeat(correlation["baseline_S_squared"], len(correlation["C2"])),
        "leave_angle_C2_min": np.nanmin(loo_curves, axis=0),
        "leave_angle_C2_max": np.nanmax(loo_curves, axis=0)})
    write_csv(source/"panel_h_multiscale_order.csv", analysis["scale_rows"])
    write_csv(source/"angle_prediction_validation.csv", angle_rows)
    write_csv(source/"orientation_quality_sensitivity.csv", threshold_rows)
    write_csv(source/"leave_one_angle_features.csv", loo_rows)
    np.savez_compressed(source/"leave_one_angle_curves.npz", C2=loo_curves, local_S_median=loo_scales)
    write_csv(feature_dir/"plaque_features_long.csv", features)
    write_csv(feature_dir/"feature_dictionary.csv", [{k: v for k, v in row.items()
                                            if k not in ("value", "sample_id")} for row in features])
    write_csv(feature_dir/"plaque_features_wide.csv", [{"sample_id": config["sample_id"],
                                              **{r["feature"]: r["value"] for r in features}}])
    write_csv(feature_dir/"model_candidate_features.csv", [{"sample_id": config["sample_id"],
                                                   **{r["feature"]: r["value"] for r in features
                                                      if r["default_model_candidate"]}}])
    # A complete pixel-level table makes the analytical masks independently inspectable.
    yy, xx = np.where(tissue)
    pixels = {"roi_row": yy, "roi_column": xx, "x_um": (xx+.5)*config["pixel_size_um"],
              "y_um": (yy+.5)*config["pixel_size_um"],
              "orientation_qualified": analysis["valid_orientation"][tissue],
              "local_S": analysis["S_local"][tissue], "depth_um": analysis["depth_map_um"][tissue]}
    pixels.update({key: value[tissue] for key, value in maps.items()})
    pd.DataFrame(pixels).to_csv(source/"optical_tissue_pixels.csv.gz", index=False,
                                float_format="%.9g", compression="gzip")


def write_visual_sources(out, visual, analysis, pa_display):
    source = Path(out)/"source_data"
    write_csv(source/"panel_d_direction_glyphs.csv", visual["glyphs"])
    write_csv(source/"panel_e_depth_regions.csv", analysis["depth_rows"])
    write_csv(source/"panel_f_region_distributions.csv", visual["depth_summaries"])
    write_csv(source/"panel_f_density_histograms.csv", visual["depth_histograms"])
    rainclouds = [g for groups in visual["rainclouds"].values() for g in groups]
    write_csv(source/"panel_f_raincloud_summary.csv", [g["summary"] for g in rainclouds])
    write_csv(source/"panel_f_scatter_display.csv", [r for g in rainclouds for r in g["scatter"]])
    write_csv(source/"panel_f_raincloud_density.csv", [
        {"family": g["summary"]["family"], "depth_region": g["summary"]["depth_region"],
         "raw_value": float(x), "density_per_raw_unit": float(y)}
        for g in rainclouds for x, y in zip(g["density_x"], g["density"])])
    write_csv(source/"panel_f_median_curve.csv", [
        {"family": family, "x_category": float(x), "median_guide_raw_units": float(y)}
        for family, trend in visual["median_trends"].items()
        for x, y in zip(trend["x_category"], trend["median_guide"])])
    write_csv(source/"panel_g_depth_azimuth_cells.csv", visual["spatial"]["rows"])
    write_csv(source/"panel_h_scale_density_histograms.csv", visual["scale_histograms"])
    write_csv(source/"panel_h_scatter_display.csv", visual["scale_scatter"])
    write_csv(source/"panel_h_frame_layout.csv", [
        {"frame": j+1, "window_px": row["window_px"], "window_um": row["window_um"],
         "shared_horizontal_category": j+1, "display_layer": "oblique plane",
         "common_support_n": int(analysis["scale_common_mask"].sum()),
         **dict(zip(["crop_x0", "crop_y0", "crop_x1_exclusive", "crop_y1_exclusive"],
                    visual["scale_frame_crop_xyxy"]))}
        for j, row in enumerate(analysis["scale_rows"])])
    write_csv(source/"panel_i_PA_neighbour_pairs_display_coordinates.csv", visual["PA_pairs"])
    y, x = np.where(np.isfinite(pa_display))
    write_csv(source/"panel_i_PA_cells.csv", {"display_row": y, "display_column": x,
              "DOA": pa_display[y, x], "neighbour_abs_difference_mean": visual["PA_neighbour_mean"][y, x],
              "neighbour_n": visual["PA_neighbour_n"][y, x]})
    distribution_i = visual["PA_distribution"]
    write_csv(source/"panel_i_DOA_histogram.csv", [
        row | {"percent": float(percent)} for row, percent in zip(
            histogram_rows(distribution_i["values"], distribution_i["edges"], "DOA", "all"),
            distribution_i["percent"])])
    write_csv(source/"panel_i_DOA_distribution_summary.csv", [distribution_i["summary"]])
    write_csv(source/"panel_i_DOA_density_curve.csv", {
        "DOA": distribution_i["curve_x"], "density_per_DOA_unit": distribution_i["curve_density"],
        "percent_per_histogram_bin": distribution_i["curve_percent_per_bin"],
        "inside_display_axis": distribution_i["curve_x"] <= distribution_i["edges"][-1]})
    np.savez_compressed(source/"visual_feature_maps.npz", depth_region=analysis["depth_labels"],
                        spatial_sector=visual["spatial"]["sector_map"],
                        azimuth_deg=visual["spatial"]["azimuth_deg"],
                        centroid_row_col=visual["spatial"]["centroid_row_col"],
                        M2_spatial_median=visual["spatial"]["M2"],
                        S_spatial_median=visual["spatial"]["S"],
                        multiscale_S=visual["scale_maps"],
                        multiscale_display_crop_xyxy=visual["scale_frame_crop_xyxy"],
                        PA_neighbour_mean=visual["PA_neighbour_mean"],
                        PA_neighbour_count=visual["PA_neighbour_n"])
