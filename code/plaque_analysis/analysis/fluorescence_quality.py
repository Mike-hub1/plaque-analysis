"""Figure 3: quality."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import ID_COLUMNS
import numpy as np


def saturation_qc(raw, mask, effective_bits=None, metadata_bits=None):
    """有效位深优先于容器位深；不从显示 LUT 的 BitCountRange 推断探测器有效位深。"""
    raw = np.asarray(raw)
    container_bits = np.iinfo(raw.dtype).bits if raw.dtype.kind in "ui" else None
    bits = effective_bits if effective_bits is not None else metadata_bits
    source = "user" if effective_bits is not None else "acquisition_metadata"
    if bits is None and container_bits == 8:
        bits, source = 8, "native_8bit_storage"
    if bits is None:
        ceiling = float(np.iinfo(raw.dtype).max) if container_bits else None
        source = "container_only_effective_bits_unknown"
    else:
        bits = int(bits)
        if bits < 1 or container_bits is None or bits > container_bits:
            raise ValueError("有效位深不适合当前 detector 数据类型。")
        ceiling = float(2**bits-1)
        if np.max(raw) > ceiling:
            raise ValueError(f"像素值超过声明的 {bits}-bit 上限；检查有效位深或左对齐编码。")
    sat = (raw >= ceiling) if ceiling is not None else np.zeros(raw.shape, bool)
    n = int(np.count_nonzero(mask))
    return {
        "effective_bits": bits, "bit_depth_source": source,
        "effective_bits_known": bits is not None, "ceiling": ceiling,
        "saturated_pixels_field": int(np.count_nonzero(sat)),
        "saturated_pixels_mask": int(np.count_nonzero(sat & mask)),
        "saturated_fraction_mask": float(np.count_nonzero(sat & mask)/n) if n else np.nan,
        "mask": sat,
    }


def evaluate_segmentation(predicted, reference, spacing):
    p, r = np.asarray(predicted, bool), np.asarray(reference, bool)
    intersection, union = int((p & r).sum()), int((p | r).sum())
    npix, nr = int(p.sum()), int(r.sum())
    return {"predicted_area_um2": npix*np.prod(spacing), "reference_area_um2": nr*np.prod(spacing),
            "signed_area_bias_um2": (npix-nr)*np.prod(spacing),
            "relative_area_bias": (npix-nr)/nr if nr else np.nan,
            "segmentation_dice": 2*intersection/(npix+nr) if npix+nr else np.nan,
            "segmentation_jaccard": intersection/union if union else np.nan,
            "pixel_precision": intersection/npix if npix else np.nan,
            "pixel_recall": intersection/nr if nr else np.nan,
            "false_positive_pixels": int((p & ~r).sum()), "false_negative_pixels": int((~p & r).sum()),
            "status": "reference_comparison_not_biological_channel_overlap"}


def repeated_measurement_statistics(table):
    required = {"unit_id", "repeat_id", "feature", "value"}
    if not required <= set(table.columns):
        raise ValueError("重复表需含 unit_id,repeat_id,feature,value；unit_id 必须指同一斑块。")
    results = []
    for feature, data in table.groupby("feature"):
        if data.duplicated(["unit_id", "repeat_id"]).any():
            raise ValueError("重复测量表的同一单位/重复/特征存在重复行。")
        wide = data.pivot(index="unit_id", columns="repeat_id", values="value")
        arr = wide.to_numpy(dtype=float)
        n, k = arr.shape
        row = {"feature": feature, "units": n, "repeats": k, "ICC_A1": np.nan,
               "median_within_unit_cv": np.nan, "status": "not_estimable",
               "reason": "need_balanced_complete_at_least_3_units_and_2_repeats"}
        if n >= 3 and k >= 2 and np.all(np.isfinite(arr)):
            grand = arr.mean()
            ms_unit = k*np.sum((arr.mean(1)-grand)**2)/(n-1)
            ms_repeat = n*np.sum((arr.mean(0)-grand)**2)/(k-1)
            residual = arr-arr.mean(1)[:, None]-arr.mean(0)[None, :]+grand
            ms_error = np.sum(residual**2)/((n-1)*(k-1))
            denom = ms_unit+(k-1)*ms_error+k*(ms_repeat-ms_error)/n
            if denom > 0:
                row.update({"ICC_A1": float((ms_unit-ms_error)/denom),
                            "status": "computed", "reason": ""})
            positive_mean = arr.mean(1) > 0
            if np.any(positive_mean):
                row["median_within_unit_cv"] = float(np.median(
                    arr.std(1, ddof=1)[positive_mean]/arr.mean(1)[positive_mean]))
            row["cv_nonpositive_mean_units"] = int(np.count_nonzero(~positive_mean))
        results.append(row)
    return results


def validate_split_manifest(table):
    required = {"AnimalID", "SectionID", "FieldID", "PlaqueID", "split"}
    if not required <= set(table.columns):
        raise ValueError("分折清单需含 AnimalID,SectionID,FieldID,PlaqueID,split。")
    if table[list(required)].isna().any().any() or (table[list(required)].astype(str) == "").any().any():
        raise ValueError("分折标识不可缺失。")
    if not set(table["split"]) <= {"train", "validation", "test"}:
        raise ValueError("split 只能是 train/validation/test。")
    for columns in (["AnimalID"], ["AnimalID", "SectionID"], ["AnimalID", "FieldID"],
                    ID_COLUMNS):
        if (table.groupby(columns)["split"].nunique() > 1).any():
            raise ValueError("数据泄漏: "+"/".join(columns)+" 跨 split。")
    for column in ("source_sha256", "repeat_group"):
        if column in table:
            known = table.dropna(subset=[column])
            known = known[known[column].astype(str) != ""]
            if (known.groupby(column)["split"].nunique() > 1).any():
                raise ValueError(f"数据泄漏: {column} 跨 split。")
    return {"status": "passed", "rows": len(table), "animals": table.AnimalID.nunique(),
            "cross_sample_fitting_in_this_script": False,
            "downstream_fit_provenance": "not_verifiable_without_training_pipeline"}

