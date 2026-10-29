"""Figure 3: analysis."""
from __future__ import annotations

from plaque_analysis.features.abeta import compute_abeta_features
from plaque_analysis.settings.fluorescence import ID_COLUMNS
from plaque_analysis.features.fluorescence_schema import JOINT_INTENSITY_FEATURES, LAMP1_INTENSITY_FEATURES
from plaque_analysis.features.fluorescence_geometry import build_windows, touches_unobserved
from plaque_analysis.features.lamp1 import compute_lamp1_outcomes
from plaque_analysis.analysis.fluorescence_quality import saturation_qc
from plaque_analysis.features.colocalization import compute_colocalization, extended_joint_features, neighborhood_composition
from plaque_analysis.features.texture import intensity_strata
import math
import numpy as np


def analyse_arrays(raw, signed, masks, valid, spacing, config, *,
                   effective_bits=(None, None), metadata_bits=(None, None), identities=None):
    """Pure numerical orchestrator; plotting/export cannot mutate the input arrays."""
    config = config.resolved(spacing)
    labels = np.asarray(masks[0], np.int32)
    mask_l = np.asarray(masks[1], bool)
    identities = identities or {}
    windows = build_windows(labels, valid[0], spacing, config.window_radius_um)
    tables = {"X_A": [], "Y_L": [], "Z_AL": []}
    feature_qc, image_qc, profiles_all, angular_all, directional_all, neighbors_all = [], [], [], [], [], []
    objects = []
    if not windows:
        image_qc.append({**{key: identities.get(key) for key in ID_COLUMNS}, "object_id": 0,
                         "object_pixels": 0, "status": "not_estimable_no_abeta_object",
                         "reason": "empty_abeta_segmentation"})
    for oid in sorted(windows):
        mask_a, geometry = labels == oid, windows[oid]
        ident = {key: identities.get(key) for key in ID_COLUMNS}
        ident["PlaqueID"] = f"{identities.get('PlaqueID') or 'object'}_{oid:03d}"
        if mask_a.sum() < config.min_object_pixels:
            image_qc.append({**ident, "object_id": oid, "status": "excluded_too_small",
                            "object_pixels": int(mask_a.sum()), "reason": "below_prespecified_min_object_pixels"})
            continue
        fa, qa, directions, sat_a = compute_abeta_features(
            raw[0], signed[0], mask_a, spacing, config, effective_bits=effective_bits[0],
            metadata_bits=metadata_bits[0], truncated=geometry["object_truncated"])
        common_valid = valid[0] & valid[1]
        yl, ql, profile, angular, window = compute_lamp1_outcomes(
            signed[1], mask_l, mask_a, common_valid, geometry, spacing, config)
        sat_l = saturation_qc(raw[1], mask_l & window, effective_bits[1], metadata_bits[1])
        za, reasons = compute_colocalization(signed[0], signed[1], mask_a, mask_l, window)
        extended_joint, extended_reasons = extended_joint_features(signed[0], signed[1], mask_a, mask_l, window, spacing)
        za.update(extended_joint); reasons.update(extended_reasons)
        window_saturation = [saturation_qc(r, window, e, b) for r, e, b in zip(raw, effective_bits, metadata_bits)]
        pixels_l = np.column_stack(np.where(mask_l & window))
        pixels_a = np.column_stack(np.where(mask_a))
        delta = (np.mean(pixels_l, axis=0)-np.mean(pixels_a, axis=0))*np.array(spacing) if len(pixels_l) else None
        displacement = float(np.linalg.norm(delta)) if delta is not None else np.nan
        za.update({"lamp1_centroid_offset_um": displacement,
                   "lamp1_centroid_offset_norm": displacement/math.sqrt(mask_a.sum()*np.prod(spacing)/np.pi)})
        qz = []
        for key, value in list(za.items()):
            reason = reasons.get(key, "")
            intensity_key = key in JOINT_INTENSITY_FEATURES
            if not yl["window_complete"]:
                reason = "truncated_analysis_window"
            elif intensity_key and (not sat_a["effective_bits_known"] or not sat_l["effective_bits_known"]):
                reason = "effective_bit_depth_unknown"
            elif intensity_key and any(s["saturated_fraction_mask"] > config.max_saturated_fraction
                                       for s in [sat_a, sat_l]+window_saturation):
                reason = "intensity_clipped_at_effective_bit_depth_limit"
            elif not np.isfinite(value) and not reason:
                reason = "no_lamp1_positive_centroid"
            za[key] = value if not reason else np.nan
            qz.append({"role": "Z_AL", "feature": key, "value": za[key], "diagnostic_value": value,
                       "status": "not_estimable" if reason else "computed_pending_validation", "reason": reason})
        if (not sat_l["effective_bits_known"] or any(s["saturated_fraction_mask"] > config.max_saturated_fraction
                                                     for s in (sat_l, window_saturation[1]))):
            for key in LAMP1_INTENSITY_FEATURES:
                yl[key] = np.nan
            for row in ql:
                if row["feature"] in LAMP1_INTENSITY_FEATURES:
                    row.update(value=np.nan, status="not_estimable", reason="lamp1_intensity_ceiling_or_unknown_bits")
            for row in profile:
                row["mean_signed_signal_complete"] = np.nan
        ranks_a, strata_a = intensity_strata(np.maximum(signed[0], 0), mask_a, config.n_clusters, config.random_state)
        ranks_l, strata_l = intensity_strata(np.maximum(signed[1], 0), mask_l & window, config.n_clusters, config.random_state)
        neighbor = neighborhood_composition(ranks_a, ranks_l, window, len(strata_l))
        for table, record in zip(tables.values(), (fa, yl, za)):
            table.append({**ident, **record})
        feature_qc.extend({**ident, **r} for r in qa+ql+qz)
        profiles_all.extend({**ident, **r} for r in profile)
        angular_all.extend({**ident, **r} for r in angular)
        directional_all.extend({**ident, **r,
            "eligible_intensity_qc": bool(sat_a["effective_bits_known"] and
                sat_a["saturated_fraction_mask"] <= config.max_saturated_fraction and not geometry["object_truncated"])}
            for r in directions)
        neighbors_all.extend({**ident, **r} for r in neighbor)
        image_qc.append({**ident, "object_id": oid, "object_pixels": int(mask_a.sum()),
                        "status": "computed_with_feature_level_qc",
                        "object_truncated": geometry["object_truncated"],
                        "lamp1_signal_touches_crop": touches_unobserved(mask_l, valid[1]),
                        "window_complete": yl["window_complete"],
                        "observable_window_fraction": yl["observable_window_fraction"],
                        "ambiguous_window_pixels": geometry["ambiguous_pixels_near_object"],
                        "abeta_saturated_fraction": sat_a["saturated_fraction_mask"],
                        "lamp1_saturated_fraction": sat_l["saturated_fraction_mask"],
                        "segmentation_validation": "pending_independent_reference_set"})
        objects.append({"id": oid, "identity": ident, "geometry": geometry, "mask_a": mask_a,
                        "window": window, "profiles": profile, "angular": angular, "features": fa,
                        "outcomes": yl, "relations": za, "relations_qc": qz, "saturation": [sat_a, sat_l],
                        "ranks": [ranks_a, ranks_l], "strata": [strata_a, strata_l]})
    return {"tables": tables, "feature_qc": feature_qc, "qc": image_qc,
            "boundary_profiles": profiles_all, "angular_profiles": angular_all,
            "glcm_directions": directional_all, "neighborhood_composition": neighbors_all,
            "objects": objects, "config": config}

