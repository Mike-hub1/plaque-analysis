"""Figure 3: tables."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import ID_COLUMNS, VERSION
from plaque_analysis.features.fluorescence_schema import ROLE_FEATURES, WINDOW_QC_FEATURES, feature_dictionary
from plaque_analysis.io.fluorescence_utils import json_clean
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def build_complete_feature_tables(result, acquisition, sensitivity=()):
    """One authoritative long CSV plus a role-prefixed per-plaque matrix.

    No row is inferred from a display image. Excluded segmented objects stay as
    missing rows with their reason; an empty segmentation does not invent an object.
    """
    definitions = feature_dictionary(result["tables"], result["config"])
    dictionary = {(row["role"], row["feature"]): row for row in definitions}
    total_features = sum(map(len, ROLE_FEATURES.values()))
    if len(dictionary) != total_features:
        raise ValueError("特征字典不完整或存在重名。")
    records = {}
    for role, rows in result["tables"].items():
        for row in rows:
            if set(row)-set(ID_COLUMNS) != set(ROLE_FEATURES[role]):
                raise ValueError(f"{role} 实际测量字段与字典不一致；禁止导出时静默丢列。")
            key = (role, row["PlaqueID"])
            if key in records:
                raise ValueError("同一文件内出现重复 PlaqueID。")
            records[key] = row
    qc_lookup = {}
    for row in result["feature_qc"]:
        key = (row["PlaqueID"], row["role"], row["feature"])
        if key in qc_lookup:
            raise ValueError("特征 QC 存在重复记录。")
        qc_lookup[key] = row
    spacing = acquisition.get("spacing_yx_um", (np.nan, np.nan))
    cfg = result["config"]
    parameters = {"measurement": asdict(cfg), "background": acquisition.get("background"),
                  "segmentation": acquisition.get("segmentation"), "auxiliary": acquisition.get("auxiliary_inputs"),
                  "selection": acquisition.get("selection"), "spacing_yx_um": spacing}
    parameter_sha = hashlib.sha256(json.dumps(json_clean(parameters), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    channels = acquisition.get("channels", [{}, {}])
    common = {
        "source_file": Path(acquisition.get("input_file", "not_supplied")).name,
        "source_sha256": acquisition.get("input_sha256"), "script_sha256": acquisition.get("script_sha256"),
        "analysis_parameter_sha256": parameter_sha, "version": VERSION, "run_utc": acquisition.get("run_utc"),
        "pixel_size_y_um": spacing[0], "pixel_size_x_um": spacing[1],
        "abeta_channel_index": channels[0].get("channel_index"), "lamp1_channel_index": channels[1].get("channel_index"),
        "window_radius_um": cfg.window_radius_um, "ring_width_um": cfg.ring_width_um,
        "edge_width_um": cfg.edge_width_um, "angular_sectors": cfg.angular_sectors,
        "max_allowed_saturated_fraction": cfg.max_saturated_fraction,
        "parameters_frozen": cfg.parameters_frozen,
        **{key: acquisition.get("covariates", {}).get(key) for key in
           ("age_months", "region", "staining_batch", "acquisition_batch")},
        "segmentation_reference_validation": "not_evaluated_here_see_segmentation_reference_validation.json",
        "real_repeat_imaging_validation": "not_evaluated_here_see_validation_report.json",
    }
    for ci, tag in enumerate(("abeta", "lamp1")):
        bg = acquisition.get("background", [{}, {}])[ci]
        sg = acquisition.get("segmentation", [{}, {}])[ci]
        sat = acquisition.get("saturation", [{}, {}])[ci]
        supplied_mask = acquisition.get("provided_object_mask" if ci == 0 else "provided_lamp1_mask")
        common.update({tag+"_native_dtype": channels[ci].get("dtype"),
                       tag+"_effective_bits": sat.get("effective_bits"),
                       tag+"_effective_ceiling": sat.get("ceiling"),
                       tag+"_background_rule": bg.get("rule"),
                       tag+"_background_level": bg.get("background_level"),
                       tag+"_background_noise_sd": bg.get("background_noise_sd"),
                       tag+"_segmentation_method": "supplied_mask" if supplied_mask else sg.get("method"),
                       tag+"_foreground_low": None if supplied_mask else json_clean(sg.get("low")),
                       tag+"_foreground_high": None if supplied_mask else json_clean(sg.get("high"))})
    object_fields = ["object_id", "object_pixels", "object_selection_status", "object_truncated",
                     "lamp1_signal_touches_crop", "window_complete", "observable_window_fraction",
                     "abeta_saturated_fraction", "lamp1_saturated_fraction"]
    value_fields = ["role", "feature", "feature_name_zh", "family", "measurement_kind", "units", "value",
                    "diagnostic_value", "status", "reason", "candidate_abeta_only_input", "formula", "mask",
                    "interpretation_redundancy", "implementation", "source_url", "source_scope", "model_use",
                    "scale", "thresholds", "missing_conditions", "cross_sample_fit", "parameter_status",
                    "texture_lag_pixels", "texture_offsets_xy_um", "texture_bin_edges", "minimum_cardinal_pairs",
                    "sensitivity_scenarios", "sensitivity_valid_values", "sensitivity_missing_values",
                    "sensitivity_max_abs_change", "sensitivity_max_relative_change"]
    long_rows, wide_rows = [], []
    for image_qc in result["qc"]:
        if image_qc.get("object_id", 0) == 0:
            continue
        pid = image_qc["PlaqueID"]
        selected = image_qc["status"] == "computed_with_feature_level_qc"
        meta = {**{key: image_qc.get(key) for key in ID_COLUMNS}, **common,
                **{key: image_qc.get(key) for key in object_fields},
                "object_selection_status": image_qc["status"]}
        wide = dict(meta)
        object_scenarios = [s for s in sensitivity if s["base_object_id"] == image_qc["object_id"]]
        for (role, feature), definition in dictionary.items():
            value = records.get((role, pid), {}).get(feature, np.nan)
            if isinstance(value, (bool, np.bool_)):
                value = int(value)
            qc = qc_lookup.get((pid, role, feature))
            if qc is None:
                if not selected:
                    qc = {"diagnostic_value": np.nan, "status": "not_estimable",
                          "reason": image_qc.get("reason", image_qc["status"])}
                elif feature in WINDOW_QC_FEATURES:
                    qc = {"diagnostic_value": value, "status": "observed_diagnostic",
                          "reason": "not_a_replacement_for_the_complete_window_target" if feature == "lamp1_area_observed_um2" else ""}
                else:
                    raise ValueError(f"{pid}/{role}/{feature} 缺少特征级 QC。")
            finite = bool(np.isfinite(value))
            if not finite and not qc.get("reason"):
                raise ValueError(f"缺失值没有原因: {pid}/{feature}")
            if np.isinf(value) or np.isinf(qc.get("diagnostic_value", np.nan)):
                raise ValueError(f"特征包含无穷值: {pid}/{feature}")
            if "value" in qc and not np.isclose(qc["value"], value, equal_nan=True):
                raise ValueError(f"QC 值与测量表不一致: {pid}/{feature}")
            prefix = {"X_A": "A_", "Y_L": "L_", "Z_AL": "AL_"}[role]
            perturbations = np.array([s.get(prefix+feature, np.nan) for s in object_scenarios], dtype=float)
            good = perturbations[np.isfinite(perturbations)]
            max_delta = float(np.max(abs(good-float(value)))) if finite and len(good) else np.nan
            texture_info = {"texture_lag_pixels": None, "texture_offsets_xy_um": "",
                            "texture_bin_edges": "", "minimum_cardinal_pairs": None}
            if feature.startswith("glcm_"):
                step = int(feature.split("_")[1][3:]) if feature.startswith("glcm_lag") else 1
                directions = [r for r in result["glcm_directions"] if r["PlaqueID"] == pid and r["native_pixel_step"] == step]
                texture_info.update({"texture_lag_pixels": step,
                    "texture_offsets_xy_um": json.dumps([[r["dx"]*spacing[1], r["dy"]*spacing[0]] for r in directions]),
                    "texture_bin_edges": directions[0]["bin_edges"] if directions else "",
                    "minimum_cardinal_pairs": min((r["pairs"] for r in directions
                        if (r["dx"], r["dy"]) in ((step, 0), (0, step))), default=None)})
            long_rows.append({**meta, **{key: definition[key] for key in value_fields if key in definition},
                "value": value, "diagnostic_value": qc.get("diagnostic_value", value),
                "status": qc["status"], "reason": qc.get("reason", ""),
                "candidate_abeta_only_input": bool(role == "X_A" and finite and selected),
                **texture_info, "sensitivity_scenarios": len(perturbations),
                "sensitivity_valid_values": len(good), "sensitivity_missing_values": len(perturbations)-len(good),
                "sensitivity_max_abs_change": max_delta,
                "sensitivity_max_relative_change": max_delta/abs(float(value)) if finite and value != 0 else np.nan})
            column = role+"__"+feature
            wide[column] = value
            wide[column+"__status"] = qc["status"]
            wide[column+"__reason"] = qc.get("reason", "")
        wide_rows.append(wide)
    meta_columns = ID_COLUMNS+list(common)+object_fields
    wide_columns = meta_columns+[c for role, names in ROLE_FEATURES.items() for f in names
                                 for c in (role+"__"+f, role+"__"+f+"__status", role+"__"+f+"__reason")]
    long = pd.DataFrame(long_rows, columns=meta_columns+value_fields)
    for column in ("value", "diagnostic_value"):
        long[column] = long[column].astype(float)
    wide = pd.DataFrame(wide_rows, columns=wide_columns)
    if len(long) != len(wide)*total_features or long.duplicated(["PlaqueID", "role", "feature"]).any():
        raise ValueError("完整特征表必须每对象/特征一行。")
    return long, wide


def export_complete_feature_tables(result, acquisition, output, sensitivity):
    long, wide = build_complete_feature_tables(result, acquisition, sensitivity)
    for name, table in (("all_features_complete", long), ("all_features_wide", wide)):
        table.to_csv(output/f"{name}.csv", index=False, encoding="utf-8-sig", na_rep="", float_format="%.15g")
    return {"status": "written", "plaques_including_excluded": len(wide), "long_rows": len(long),
            "features_per_plaque": sum(map(len, ROLE_FEATURES.values())),
            "feature_counts_by_role": {role: len(names) for role, names in ROLE_FEATURES.items()},
            "wide_columns": len(wide.columns), "missing_value_policy": "empty numeric cell plus explicit reason, never zero-filled",
            "sensitivity_policy": "same-image perturbations; not biological replicates or ICC",
            "scope": "all scalar phenotype/QC fields; profiles, directional GLCM and strata retain their detailed tables"}

