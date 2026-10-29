"""Figure 3: plot data."""
from __future__ import annotations

from plaque_analysis.features.fluorescence_geometry import touches_unobserved
from plaque_analysis.features.lamp1 import radial_angular_cells
from skimage.measure import label as sk_label
import numpy as np
import pandas as pd


def prepare_visual_data(raw, signed, masks, valid, geometry, profiles, spacing, config,
                        out_dir, plaque_id):
    """Visualization-only aggregation. All pixels in the declared observed W are kept."""
    window = geometry["expected"] & valid[0] & valid[1]
    sy, sx = spacing
    yy, xx = np.indices(window.shape)
    ay, ax = np.where(masks[0])
    cy, cx = ay.mean(), ax.mean()
    angle = np.mod(np.degrees(np.arctan2((yy-cy)*sy, (xx-cx)*sx)), 360)
    pixel = pd.DataFrame({"x_um": xx[window]*sx, "y_um": yy[window]*sy,
        "distance_um": geometry["distance"][window], "angle_deg": angle[window],
        "abeta_signed": signed[0][window], "lamp1_signed": signed[1][window],
        "abeta_positive": masks[0][window], "lamp1_positive": masks[1][window]})
    pixel["binary_class"] = pixel.abeta_positive.astype(int)+2*pixel.lamp1_positive.astype(int)
    pixel["ring_index"] = -1
    ring_rows = []
    for k, r in enumerate(profiles):
        domain = (pixel.distance_um >= r["lower_um"]) & ((pixel.distance_um <= r["upper_um"])
                   if k == len(profiles)-1 else (pixel.distance_um < r["upper_um"]))
        pixel.loc[domain, "ring_index"] = k
        values = pixel.loc[domain, "lamp1_signed"].to_numpy()
        q = np.percentile(values, [25, 50, 75]) if len(values) else [np.nan]*3
        ring_rows.append({**r, "ring_index": k, "signal_q25": q[0], "signal_median": q[1], "signal_q75": q[2]})
    if len(pixel) and (pixel.ring_index < 0).any():
        raise ValueError("可视化环带未覆盖完整观测窗口。")
    cells = radial_angular_cells(masks[0], masks[1], window, geometry, profiles, spacing,
                                 config.angular_sectors)
    patch_labels = sk_label(masks[1] & window, connectivity=2)
    patch_rows = []
    for label in np.unique(patch_labels[patch_labels > 0]):
        y, x = np.where(patch_labels == label)
        patch_rows.append({"patch_id": int(label), "pixels": len(x), "area_um2": len(x)*sx*sy,
            "centroid_x_um": float(x.mean()*sx), "centroid_y_um": float(y.mean()*sy),
            "touches_window_boundary": touches_unobserved(patch_labels == label, window)})
    master = pd.read_csv(out_dir/"all_features_complete.csv")
    master = master[master.PlaqueID == plaque_id].copy()
    sensitivity_path = out_dir/"sensitivity_analysis.csv"
    sensitivity = (pd.read_csv(sensitivity_path) if sensitivity_path.is_file() and
                   master.sensitivity_scenarios.max() > 0 else pd.DataFrame())
    if len(master) and len(sensitivity):
        sensitivity = sensitivity[sensitivity.base_object_id == master.object_id.iloc[0]].copy()
    directions = pd.read_csv(out_dir/"glcm_directions.csv")
    directions = (directions[directions.PlaqueID == plaque_id].copy() if "PlaqueID" in directions else
                  pd.DataFrame(columns=["dx", "dy", "native_pixel_step", "distance_um", "contrast", "homogeneity", "entropy"]))
    composition = pd.read_csv(out_dir/"neighborhood_composition.csv")
    composition = (composition[composition.PlaqueID == plaque_id].copy() if "PlaqueID" in composition else
                   pd.DataFrame(columns=["abeta_rank", "lamp1_state", "proportion", "baseline_proportion", "denominator_pixels"]))
    return {"pixel": pixel, "rings": pd.DataFrame(ring_rows), "cells": pd.DataFrame(cells),
            "patches": pd.DataFrame(patch_rows, columns=["patch_id", "pixels", "area_um2",
                "centroid_x_um", "centroid_y_um", "touches_window_boundary"]), "patch_labels": patch_labels,
            "window": window, "master": master, "sensitivity": sensitivity,
            "directions": directions, "composition": composition}


def empirical_curve(ax, values, color, label=None, **kwargs):
    values = np.sort(np.asarray(values, float))
    values = values[np.isfinite(values)]
    if len(values):
        ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                where="post", color=color, lw=1, label=label, **kwargs)


def feature_atlas_data(data, role):
    """Display accepted baseline values only; an undefined relative change stays missing."""
    master = data["master"][data["master"].role == role].copy().reset_index(drop=True)
    scenarios = data["sensitivity"].reset_index(drop=True)
    prefix = {"X_A": "A_", "Y_L": "L_", "Z_AL": "AL_"}[role]
    rows = []
    for i, feature in master.iterrows():
        for j, scenario in scenarios.iterrows():
            value = scenario.get(prefix+feature.feature, np.nan)
            value = float(value) if pd.notna(value) else np.nan
            baseline = float(feature.value)
            finite = np.isfinite(baseline) and np.isfinite(value)
            absolute = value-baseline if finite else np.nan
            relative = absolute/abs(baseline)*100 if finite and baseline != 0 else np.nan
            state = ("relative_change" if np.isfinite(relative) else "unchanged_zero" if finite and absolute == 0
                     else "zero_baseline_absolute_change" if finite else "withheld_or_unavailable")
            rows.append({"role": role, "feature": feature.feature, "feature_row": i, "scenario_column": j,
                "parameter": scenario.parameter, "factor": scenario.factor, "baseline_value": baseline,
                "perturbed_value": value, "absolute_change": absolute, "relative_change_percent": relative,
                "display_status": state, "baseline_status": feature.status, "baseline_reason": feature.reason,
                "object_match_iou": scenario.object_match_iou,
                "matched_fragment_count": scenario.matched_fragment_count,
                "base_labels_in_matched_object": scenario.base_labels_in_matched_object})
    return master, scenarios, pd.DataFrame(rows)


def load_saved_visual_data(out_dir, plaque_id, window, mask_l):
    """只读绘图明细；不会用重算表覆盖既有结果，也不重新拟合分层。"""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in plaque_id)
    source = out_dir/"plot_source_data"/safe_id
    data = {key: pd.read_csv(source/(suffix+".csv")) for key, suffix in
            (("pixel", "pixels"), ("rings", "radial_summary"), ("cells", "radial_angular"), ("patches", "lamp1_patches"))}
    data["window"] = np.asarray(window, bool)
    data["patch_labels"] = sk_label(mask_l & window, connectivity=2)
    for key, filename in (("master", "all_features_complete"), ("directions", "glcm_directions"),
                          ("composition", "neighborhood_composition")):
        table = pd.read_csv(out_dir/(filename+".csv"))
        data[key] = table[table.PlaqueID == plaque_id].copy()
    sensitivity_file = out_dir/"sensitivity_analysis.csv"
    data["sensitivity"] = pd.read_csv(sensitivity_file) if sensitivity_file.exists() else pd.DataFrame()
    if len(data["sensitivity"]):
        oid = data["master"].object_id.iloc[0]
        data["sensitivity"] = data["sensitivity"][data["sensitivity"].base_object_id == oid].copy()
    if len(data["pixel"]) != int(window.sum()):
        raise ValueError("缓存绘图像素与缓存窗口不一致，停止重绘而不重算数据。")
    for r in data["rings"].itertuples():
        selected = data["pixel"].ring_index == r.ring_index
        if int(selected.sum()) != r.observed_pixels:
            raise ValueError("缓存环带的像素分母不一致。")
    return data

