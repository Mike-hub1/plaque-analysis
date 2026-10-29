"""Figure 3: segmentation."""
from __future__ import annotations

from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import label as sk_label
import numpy as np


def correct_background(raw, *, method="percentile", percentile=5.0, roi=None,
                       valid=None, rolling_ball_radius=50.0):
    """定量支路保留 signed raw-background；分割支路另作非负裁剪。"""
    image = np.asarray(raw, dtype=np.float64)
    valid = np.ones(image.shape, bool) if valid is None else np.asarray(valid, bool)
    if not np.any(valid):
        raise ValueError("无已观测有效像素，无法估计背景。")
    if roi is not None:
        selected = valid & roi
        if selected.sum() < 3:
            raise ValueError("背景 ROI 至少须有 3 个有效像素。")
        level = float(image[selected].mean())
        background = np.full(image.shape, level)
        noise_sd = float(image[selected].std(ddof=1))
        rule, verified = "independently_supplied_roi_mean", True
    elif method == "percentile":
        if not 0 <= percentile < 100:
            raise ValueError("背景百分位需在 [0,100)。")
        level = float(np.percentile(image[valid], percentile))
        background = np.full(image.shape, level)
        noise_sd, rule, verified = np.nan, f"crop_percentile_{percentile:g}", False
    elif method == "none":
        background = np.zeros_like(image)
        level, noise_sd, rule, verified = 0.0, np.nan, "none", False
    elif method == "rolling-ball":
        from skimage.restoration import rolling_ball
        if not np.all(valid) or not np.isfinite(rolling_ball_radius) or rolling_ball_radius <= 0:
            raise ValueError("rolling-ball 要求完整观测平面及正半径。")
        background = rolling_ball(image, radius=rolling_ball_radius)
        level, noise_sd, rule, verified = float(np.median(background)), np.nan, "rolling_ball", False
    else:
        raise ValueError(f"未知背景规则: {method}")
    signed = image-background
    return signed, np.maximum(signed, 0), background, {
        "rule": rule, "background_level": level, "background_noise_sd": noise_sd,
        "roi_supplied": verified, "negative_quantification_pixels": int(np.count_nonzero((signed < 0) & valid)),
        "quantification_clipped": False,
    }


def segment_signal_foreground(corrected, *, method="multi-otsu-hysteresis",
                              n_classes=4, manual_threshold=None, valid=None,
                              threshold_factor=1.0):
    image = np.maximum(np.asarray(corrected, float), 0)
    valid = np.ones(image.shape, bool) if valid is None else np.asarray(valid, bool)
    values = image[valid]
    if not np.isfinite(threshold_factor) or threshold_factor <= 0 or not 3 <= n_classes <= 5:
        raise ValueError("前景参数无效。")
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("分割输入无有限有效像素。")
    reason = ""
    if manual_threshold is not None:
        if not np.isfinite(manual_threshold) or manual_threshold < 0:
            raise ValueError("手动信号下限必须非负且有限。")
        low = high = float(manual_threshold)*threshold_factor
        rule = "manual_absolute"
    elif method == "positive":
        low = high = 0.0
        rule = "legacy_positive_exploratory"
    elif len(np.unique(values)) < 3:
        low = high = float("inf")
        rule, reason = method, "insufficient_intensity_levels"
    elif method == "multi-otsu-hysteresis":
        thresholds = threshold_multiotsu(values, classes=min(n_classes, len(np.unique(values))))
        low, high = map(float, thresholds[:2]*threshold_factor)
        rule = method
    elif method == "otsu":
        low = high = float(threshold_otsu(values))*threshold_factor
        rule = method
    else:
        raise ValueError(f"未知前景规则: {method}")
    candidates = valid & (image > low)
    components = sk_label(candidates, connectivity=1)  # 固定 4 邻接，与旧 hysteresis 一致
    seed_ids = np.unique(components[valid & (image > high)])
    mask = candidates & np.isin(components, seed_ids[seed_ids > 0])
    return mask, {"method": rule, "low": low, "high": high, "reason": reason,
                  "connectivity": 4, "pixels": int(mask.sum()), "threshold_factor": threshold_factor}

