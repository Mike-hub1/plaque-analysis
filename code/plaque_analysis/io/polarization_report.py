"""Figure 4: report."""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
import cv2
import matplotlib
import numpy as np
import pandas as pd
import platform
import scipy
import skimage
import sys
import tifffile
from plaque_analysis.io.polarization import resolve, sha256
from plaque_analysis.plotting.polarization_helpers import DEPTH_COLOURS
from plaque_analysis.common.provenance import code_hashes
from plaque_analysis.common.polarization_version import __version__


def build_analysis_report(config, out, source_meta, mask_record, qc, visual, analysis,
                          tissue, pa, pa_bbox, stack, maps, features, complete_report,
                          resolved_config, *, artwork_used=False):
    pixel = config["pixel_size_um"]
    pa_area = (np.isfinite(pa).sum()*config["pa_native_row_pitch_um"]*config["pa_native_column_pitch_um"]
               if config["pa_native_row_pitch_um"] and config["pa_native_column_pitch_um"] else np.nan)
    report = {
        "version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": config["sample_id"], "runtime": sys.executable,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                              "opencv": cv2.__version__, "tifffile": tifffile.__version__,
                              "scikit_image": skimage.__version__, "pandas": pd.__version__},
        "source": source_meta, "configuration": resolved_config, "mask": mask_record, "quality": qc,
        "visual_story": {
            "source_final_svg_sha256": sha256(resolve(config, "final_svg"))
            if artwork_used else None,
            "final_artwork_used": artwork_used,
            "direction_glyph_n": len(visual["glyphs"]),
            "direction_field": {
                "glyph": "Equal-length, rounded axial line segments without arrowheads",
                "reference": "Axer et al., NeuroImage 54 (2011), Fig. 5; style-only inheritance",
                "background": "M2, with global linear display range 0 to 0.2 and grey endpoints #181818 and #8A8A8A",
                "colour": "Cyclic HSV hue phi/180; constant saturation 0.60 and value 1.0",
                "requested_grid_spacing_um": config["direction_spacing_um"],
                "grid_spacing_pixels": max(1, round(config["direction_spacing_um"]/pixel)),
                "actual_grid_spacing_um": max(1, round(config["direction_spacing_um"]/pixel))*pixel,
                "constant_glyph_length_um": config["direction_length_um"],
                "local_axial_window_um": analysis["window_actual_um"],
                "no_3D_inclination_encoding": True,
            },
            "depth_azimuth_cell_n": len(visual["spatial"]["rows"]),
            "regional_rainclouds": {
                "layout": "Left display scatter; central median-pinched IQR box; right half-density",
                "palette": DEPTH_COLOURS,
                "density": "All valid observations; Gaussian KDE with Scott bandwidth and reflection at zero; observed value range",
                "density_width": "Each group normalized to its own peak; width does not encode sample count",
                "box": "25th to 75th percentiles; black median line; decorative median waist, not confidence limits",
                "whiskers": "Most extreme observations within Q1-1.5 IQR and Q3+1.5 IQR",
                "median_curve": "PCHIP through the five full-population medians, visual guide on categorical depth regions",
                "angular_spread_view": visual["angular_spread_view"],
                "display_sampling": {"n_per_group": config["f_scatter_count"],
                                     "base_seed": config["f_scatter_seed"],
                                     "method": "Uniform without replacement, separately by region and variable; only x is jittered",
                                     "affects_statistics": False},
                "groups": [g["summary"] for groups in visual["rainclouds"].values() for g in groups],
                "source_data": ["source_data/panel_f_raincloud_summary.csv", "source_data/panel_f_raincloud_density.csv",
                                "source_data/panel_f_scatter_display.csv", "source_data/panel_f_median_curve.csv"],
            },
            "scale_frame_stack": {
                "frame_n": len(visual["scale_maps"]),
                "window_widths_um": [r["window_um"] for r in analysis["scale_rows"]],
                "layout": "Six oblique image planes in a rising volume, following the author-provided visual reference",
                "horizontal_alignment": "Projected frame centre, guide, summary marker and tick share each category position",
                "projection": "One common affine display projection; source arrays are unchanged",
                "illustrative_context": "Grey walls, floor and depth shading; finite S pixels remain opaque",
                "context_opacity": {"image_backdrop": .86, "floor": .24, "rear_wall": .22, "ceiling": .18},
                "context_colours": {"backdrop_endpoints": ["#A3B5C1", "#102A3B"],
                                    "floor": "#E2E9ED", "plane_edges": "#D9E3E9"},
                "crop_xyxy_native_ROI": visual["scale_frame_crop_xyxy"],
                "crop_rule": "Bounding box of common support plus 12 native pixels; no supported observations omitted",
                "colour_range_S": [0, 1],
                "colourbar_layout": "Shared horizontal bar centred between the frame stack and the distribution",
                "scale_bar_label": "White, inside the first plane, centred above and parallel to the projected scale bar",
                "distribution": "Sparse scatter with full-population median line, IQR band and 5th–95th percentile bars",
                "summary_n_per_scale": int(analysis["scale_common_mask"].sum()),
                "display_sampling": {
                    "n_per_scale": len(visual["scale_scatter"])//len(visual["scale_maps"]),
                    "seed": config["h_scatter_seed"],
                    "method": "Uniform without replacement from common centres; identical centres and horizontal jitter at every scale",
                    "affects_statistics": False,
                    "source_data": "source_data/panel_h_scatter_display.csv",
                },
                "standalone_size_mm": [89, 82],
                "display_layers_are_analysis_windows_not_physical_sections": True,
            },
            "cell_min_supported_pixels": config["spatial_cell_min_pixels"],
            "spatial_M2_above_colour_max_n": int(np.sum(visual["spatial"]["M2"] > .1)),
            "spatial_S_below_colour_min_n": int(np.sum(visual["spatial"]["S"] < .5)),
            "PA_neighbour_pairs": len(visual["PA_pairs"]),
            "PA_DOA_distribution": visual["PA_distribution"]["summary"] | {
                "curve_y_units": "density times histogram bin width times 100",
                "all_finite_PA_cells_retained": True,
                "histogram_source": "source_data/panel_i_DOA_histogram.csv",
                "density_source": "source_data/panel_i_DOA_density_curve.csv",
                "summary_source": "source_data/panel_i_DOA_distribution_summary.csv",
                "curve_is_descriptive_not_additional_measurements": True,
            },
            "PA_cell_map_mean_is_not_unweighted_pair_mean": True,
            "c_is_final_svg_embedded_raster_with_preserved_aspect_ratio": True,
            "a_b_are_corrected_reconstruction": True,
            "abc_scale_bar_um": config["main_scale_bar_um"],
            "PA_calibration_record": config["PA_calibration_record"],
        },
        "ROI_pixel_n": int(tissue.size), "tissue_outside_ROI_not_analysed": True,
        "ROI_reason": "Previously recorded Figure 4 rectangular ROI; focus is the selected plaque.",
        "tissue_mask_fraction_of_ROI": float(tissue.mean()),
        "angle_sampling_rank": int(np.linalg.matrix_rank(np.column_stack([
            np.ones(11), np.cos(2*np.deg2rad(config['angles_deg'])), np.sin(2*np.deg2rad(config['angles_deg'])),
            np.cos(4*np.deg2rad(config['angles_deg'])), np.sin(4*np.deg2rad(config['angles_deg']))]))),
        "registration_applied": False, "intensity_normalization_applied": False,
        "PA_display_crop_rows_cols": pa_bbox,
        "optical_ROI_size_um": [stack.shape[2]*pixel, stack.shape[1]*pixel],
        "optical_tissue_area_um2": int(tissue.sum())*pixel**2, "PA_valid_support_area_um2": pa_area,
        "cross_modality_integration": "Feature concatenation for the author-identified same plaque; no pixelwise spatial registration, regression or inferred optical PA resolution.",
        "half_decay_status": analysis["half_decay_status"],
        "local_window_actual_um": analysis["window_actual_um"],
        "display_clipped_n": {"M2_above_max": int(np.sum(tissue & (maps["M2"] > config["display_m2_max"]))),
                              "PA_above_max": int(np.sum(pa > config["display_doa_max"])),
                              "local_spread_above_max": int(np.sum(analysis["local_angular_spread_deg"] > config["display_local_spread_max_deg"])),
                              "QC_M4_above_4pct": int(np.sum(tissue & (maps["M4_reference"] > .04)))},
        "conditional_conversion_outside_arcsin_domain_n": analysis["invalid_arcsin_pixel_n"],
        "statistical_unit": "One plaque. Pixels, overlapping windows, ordered pairs, PA cells and omitted angles are not biological replicates.",
        "uncertainty": "Spatial IQR and leave-one-angle min/max; no p-values or biological confidence intervals.",
        "features": features,
        "complete_feature_export": complete_report,
        "code_layout": "integrated plaque_analysis package",
        "code_sha256": code_hashes(),
    }
    return report

