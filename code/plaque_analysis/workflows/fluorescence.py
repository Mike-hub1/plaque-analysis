"""Figure 3: pipeline."""
from __future__ import annotations

from plaque_analysis.analysis.fluorescence import analyse_arrays
from plaque_analysis.settings.fluorescence import ID_COLUMNS, MeasurementConfig
from plaque_analysis.io.czi import load_czi_input_channels, load_czi_input_pair, read_mask, validate_raw_channel_wavelengths
from plaque_analysis.features.fluorescence_schema import ROLE_FEATURES, feature_dictionary
from plaque_analysis.plotting.raw_channels import render_raw_channel_figures
from plaque_analysis.plotting.fluorescence import render_figures
from plaque_analysis.analysis.fluorescence_quality import evaluate_segmentation, repeated_measurement_statistics, saturation_qc, validate_split_manifest
from plaque_analysis.io.fluorescence_report import write_run_report
from plaque_analysis.imaging.fluorescence_segmentation import correct_background, segment_signal_foreground
from plaque_analysis.analysis.fluorescence_sensitivity import sensitivity_analysis
from plaque_analysis.io.fluorescence_tables import export_complete_feature_tables
from plaque_analysis.io.fluorescence_utils import json_clean, package_source_hashes, sha256_file, write_json, write_matrix
from plaque_analysis.common.output import validate_output_path
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from skimage.measure import label as sk_label
import importlib.metadata
import json
import numpy as np
import pandas as pd
import platform
import sys


def process_two_images(args):
    read_options = {"scene_index": args.scene_index, "time_index": args.time_index, "z_index": args.z_index,
        "spacing_override": (args.pixel_size_y_um, args.pixel_size_x_um) if args.pixel_size_x_um is not None else None}
    raw_export_requested = not args.no_figures and not args.skip_raw_channel_figures
    raw_display, native_display_valid, raw_channel_info = None, None, None
    if raw_export_requested:
        raw_display, native_display_valid, spacing, acquisition = load_czi_input_channels(args.czi_path,
            (("Aβ/CR", args.abeta_channel, args.reference_abeta),
             ("LAMP1", args.lamp1_channel, args.reference_lamp1), ("tau", args.tau_channel, None)), **read_options)
        raw_channel_info = acquisition["channels"]
        wavelength_check = validate_raw_channel_wavelengths(raw_channel_info)
        saturation_qc(raw_display[2], native_display_valid[2], args.tau_effective_bits,
                      raw_channel_info[2].get("effective_bits_metadata"))
        raw, valid = raw_display[:2], native_display_valid[:2]
        acquisition["channels"] = raw_channel_info[:2]  # 现有分析接口仍严格限于 Aβ 与 LAMP1。
        acquisition["raw_figure_channels"] = raw_channel_info
        acquisition["raw_figure_wavelength_check"] = wavelength_check
    else:
        raw, valid, spacing, acquisition = load_czi_input_pair(args.czi_path,
            args.abeta_channel, args.lamp1_channel, reference_abeta=args.reference_abeta,
            reference_lamp1=args.reference_lamp1, **read_options)
    acquisition.update({"third_channel_used": raw_export_requested,
                        "third_channel_use_scope": "raw_figures_only" if raw_export_requested else "not_read",
                        "tau_used_for_feature_analysis": False,
                        "tau_used_for_raw_figures": raw_export_requested})
    shape = raw[0].shape
    if args.valid_mask:
        tissue = read_mask(args.valid_mask, shape)
        valid = [v & tissue for v in valid]
    roi = read_mask(args.background_roi, shape) if args.background_roi else None
    corrected = [correct_background(a, method=args.background_method, percentile=args.background_percentile,
                                    roi=roi, valid=v, rolling_ball_radius=args.rolling_ball_radius)
                 for a, v in zip(raw, valid)]
    signed, seg_images, background, bg_info = [list(x) for x in zip(*corrected)]
    seg_kwargs = [{"method": args.foreground_method, "n_classes": args.foreground_classes,
                   "manual_threshold": threshold} for threshold in (args.abeta_min_signal, args.lamp1_min_signal)]
    segmented = [segment_signal_foreground(a, valid=v, **kw) for a, v, kw in zip(seg_images, valid, seg_kwargs)]
    mask_a = (read_mask(args.abeta_mask, shape, labels=True) if args.abeta_mask
              else sk_label(segmented[0][0], connectivity=2))
    mask_a[~valid[0]] = 0
    mask_l = read_mask(args.lamp1_mask, shape) if args.lamp1_mask else segmented[1][0]
    mask_l &= valid[1]
    config = MeasurementConfig(
        window_radius_um=args.window_radius_um, ring_width_um=args.ring_width_um,
        edge_width_um=args.edge_width_um, angular_sectors=args.angular_sectors,
        min_gradient_centres=args.min_gradient_centres,
        min_object_pixels=args.min_object_pixels, min_glcm_pairs=args.min_glcm_pairs,
        texture_levels=args.texture_levels, texture_bin_edges=args.texture_bin_edges,
        max_saturated_fraction=args.max_saturated_fraction, n_clusters=args.n_clusters,
        random_state=args.random_state, parameters_frozen=args.parameters_frozen).resolved(spacing)
    effective_bits = (args.abeta_effective_bits, args.lamp1_effective_bits)
    metadata_bits = tuple(c["effective_bits_metadata"] for c in acquisition["channels"])
    identities = {"AnimalID": args.animal_id, "SectionID": args.section_id,
                  "FieldID": args.field_id, "PlaqueID": args.plaque_id or Path(args.czi_path).stem}
    saturation = [saturation_qc(a, m, e, b) for a, m, e, b in
                  zip(raw, (mask_a > 0, mask_l), effective_bits, metadata_bits)]
    acquisition.update({
        "script_sha256": sha256_file(__file__), "package_source_sha256": package_source_hashes(), "run_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "software_versions": {p: importlib.metadata.version(p) for p in
            ("numpy", "scipy", "scikit-image", "scikit-learn", "pandas", "matplotlib", "tifffile", "aicspylibczi")},
        "identities": identities, "parameters": asdict(config), "parser_arguments": vars(args),
        "covariates": {"age_months": args.age_months, "region": args.region,
                       "staining_batch": args.staining_batch, "acquisition_batch": args.acquisition_batch},
        "background": bg_info, "segmentation": [x[1] for x in segmented],
        "provided_object_mask": bool(args.abeta_mask), "provided_lamp1_mask": bool(args.lamp1_mask),
        "object_policy": "provided_instance_IDs_or_8_connected_abeta_components; no_LAMP1_selection",
        "saturation": [{k: v for k, v in s.items() if k != "mask"} for s in saturation],
        "window_policy": "Aβ boundary + fixed μm radius; nearest boundary ownership; ties excluded",
        "distance_convention": "pixel centres to exposed rectangular pixel-cell edges; half-grid exact EDT",
        "polarization_input": "not_provided", "cross_sample_fitting": False,
        "external_image_integrity_validation": "ZEN reference only if explicitly supplied",
    })
    # Audit every auxiliary measurement input; changing a mask changes run provenance.
    auxiliary = ("background_roi", "valid_mask", "abeta_mask", "lamp1_mask", "reference_mask_abeta",
                 "reference_mask_lamp1", "reference_abeta", "reference_lamp1", "repeat_table", "split_manifest")
    acquisition["auxiliary_inputs"] = {name: {"path": str(getattr(args, name)), "sha256": sha256_file(getattr(args, name))}
                                        for name in auxiliary if getattr(args, name)}
    if args.check_inputs:
        print(json.dumps(json_clean(acquisition), ensure_ascii=False, indent=2))
        return None
    protected = [Path(args.czi_path).resolve(), Path(__file__).resolve()]+[
        Path(getattr(args, name)).resolve() for name in auxiliary if getattr(args, name)]
    output = validate_output_path(args.out_dir, protected)
    if output.exists() and any(p.name != ".DS_Store" for p in output.iterdir()) and not args.overwrite:
        raise ValueError("输出目录已有结果；指定新目录或使用 --overwrite。")
    result = analyse_arrays(raw, signed, (mask_a, mask_l), valid, spacing, config,
                            effective_bits=effective_bits, metadata_bits=metadata_bits, identities=identities)
    split_audit = validate_split_manifest(pd.read_csv(args.split_manifest)) if args.split_manifest else {
        "status": "not_applicable_no_training_in_this_script",
        "required_downstream": "nested AnimalID grouping; all cross-sample fits restricted to training"}
    repeat_result = repeated_measurement_statistics(pd.read_csv(args.repeat_table)) if args.repeat_table else []
    output.mkdir(parents=True, exist_ok=True)
    (output/"matrices").mkdir(exist_ok=True)
    write_json(output/"acquisition_manifest.json", acquisition)
    for i, tag in enumerate(("blue_Aβ", "red_LAMP1")):
        for name, matrix in (("raw_detector", raw[i]), ("background", background[i]),
                             ("background_subtracted_signed", signed[i]), ("segmentation_nonnegative", seg_images[i]),
                             ("valid_observed_mask", valid[i].astype(np.uint8)), ("saturation_mask", saturation[i]["mask"].astype(np.uint8)),
                             ("foreground_mask", ((mask_a > 0) if i == 0 else mask_l).astype(np.uint8))):
            write_matrix(output/"matrices"/f"{tag}_{name}.tif", matrix, spacing)
    write_matrix(output/"matrices"/"abeta_instance_labels.tif", mask_a.astype(np.int32), spacing)
    if raw_export_requested:
        write_matrix(output/"matrices"/"tau_raw_detector.tif", raw_display[2], spacing)
        write_matrix(output/"matrices"/"tau_valid_observed_mask.tif", native_display_valid[2].astype(np.uint8), spacing)
    for role, name in (("X_A", "features_A"), ("Y_L", "outcomes_L"), ("Z_AL", "relations_AL")):
        pd.DataFrame(result["tables"][role], columns=ID_COLUMNS+list(ROLE_FEATURES[role])).to_csv(
            output/f"{name}.csv", index=False)
    pd.DataFrame(columns=ID_COLUMNS).to_csv(output/"features_P.csv", index=False)
    for name in ("feature_qc", "qc", "boundary_profiles", "angular_profiles", "glcm_directions", "neighborhood_composition"):
        pd.DataFrame(result[name] or [], columns=None if result[name] else ["status"]).to_csv(output/f"{name}.csv", index=False)
    pd.DataFrame(feature_dictionary(result["tables"], config)).to_csv(output/"feature_dictionary.csv", index=False)
    segmentation_checks = []
    for tag, reference_path, prediction in (("Aβ", args.reference_mask_abeta, mask_a > 0),
                                           ("LAMP1", args.reference_mask_lamp1, mask_l)):
        if reference_path:
            ref = read_mask(reference_path, shape)
            segmentation_checks.append({"channel": tag, **evaluate_segmentation(prediction, ref, spacing)})
    write_json(output/"segmentation_reference_validation.json",
               segmentation_checks or {"status": "not_provided_no_claim_of_segmentation_accuracy"})
    if repeat_result:
        pd.DataFrame(repeat_result).to_csv(output/"repeatability.csv", index=False)
    write_json(output/"split_isolation_audit.json", split_audit)
    sensitivity = []
    if not args.skip_sensitivity:
        sensitivity = sensitivity_analysis(raw, signed, seg_images, valid, mask_a, spacing, config,
             seg_kwargs, effective_bits, metadata_bits, (bool(args.abeta_mask), mask_l if args.lamp1_mask else False))
        pd.DataFrame(sensitivity).to_csv(output/"sensitivity_analysis.csv", index=False)
    complete_table_summary = export_complete_feature_tables(result, acquisition, output, sensitivity)
    figure_paths = []
    for obj in result["objects"]:
        oid = obj["id"]
        for name, matrix in (("signed_boundary_distance_um", obj["geometry"]["distance"]),
                             ("analysis_window", obj["window"].astype(np.uint8)),
                             ("abeta_intensity_rank", obj["ranks"][0]), ("lamp1_intensity_rank", obj["ranks"][1])):
            write_matrix(output/"matrices"/f"object_{oid:03d}_{name}.tif", matrix, spacing)
        for tag, rows in zip(("abeta", "lamp1"), obj["strata"]):
            pd.DataFrame(rows).to_csv(output/f"object_{oid:03d}_{tag}_intensity_strata.csv", index=False)
        if not args.no_figures:
            figure_paths.append(render_figures(
                raw, signed, (obj["mask_a"], mask_l), valid, obj["geometry"], obj["profiles"],
                obj["angular"], obj["relations_qc"], obj["features"], obj["outcomes"], obj["ranks"],
                obj["strata"], obj["saturation"], spacing, config, output, obj["identity"]["PlaqueID"], args.dpi))
    raw_figure_export = (render_raw_channel_figures(raw_display, native_display_valid, spacing,
        raw_channel_info, acquisition, output, args.dpi,
        effective_bits=(args.abeta_effective_bits, args.lamp1_effective_bits, args.tau_effective_bits))
        if raw_export_requested else {"status": "not_requested"})
    validation = {
        "segmentation_reference": "supplied_single_image" if segmentation_checks else "not_available",
        "real_repeat_imaging": "supplied" if repeat_result else "not_available_ICC_not_computed",
        "split_isolation": split_audit,
        "fig2_outcome_definition_compatibility": "not_verified; do_not_compare_new_window_to_legacy_Fig2_without_reanalysis",
        "external_biological_identity_validation": "not_available_from_two_channels",
        "parameters_frozen": config.parameters_frozen,
        "complete_feature_table_summary": complete_table_summary,
        "raw_channel_export": raw_figure_export,
        "excluded_object_count": sum(q["status"].startswith("excluded") for q in result["qc"]),
        "feature_qc_counts": pd.Series([q["status"] for q in result["feature_qc"]]).value_counts().to_dict(),
    }
    write_json(output/"validation_report.json", validation)
    write_run_report(output, args, spacing, config, result, validation, complete_table_summary)
    print(json.dumps(json_clean({"output": output, "objects": len(result["objects"]),
        "spacing_yx_um": spacing,
        "features_qc": validation["feature_qc_counts"], "complete_feature_table": complete_table_summary,
        "qc": result["qc"], "figures": figure_paths,
        "raw_channel_figures": raw_figure_export.get("exports", [])}),
        ensure_ascii=False, indent=2))
    return result

