"""Polarization acquisition and configuration input."""
from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET
from plaque_analysis.common.files import sha256_file as sha256, json_clean as clean_json, write_json
from plaque_analysis.common.paths import ARTWORK_ROOT, Data


def load_config(path, *, data_dir=Data):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    dataset = Path(data_dir).expanduser()
    placeholder = Path("your project/dataset")
    if dataset != placeholder:
        for key in ("raw_image_dir", "pa_csv", "mask_npy"):
            if not config.get(key):
                continue
            try:
                suffix = Path(config[key]).relative_to(placeholder)
            except ValueError:
                continue
            config[key] = str(dataset.resolve() / suffix)
    for key, value in {"final_svg": str(ARTWORK_ROOT / "Figure4.svg"),
                       "panel_c_svg_id": "image159650e4e3",
                       "panel_c_colourbar_svg_id": "image66f52dad9f",
                       "pa_reference_field_shape_px": [608, 634], "spatial_sectors": 8,
                       "spatial_cell_min_pixels": 50, "direction_spacing_um": .32,
                       "direction_length_um": .25, "h_scatter_count": 80,
                       "h_scatter_seed": 20260908, "f_scatter_count": 80,
                       "f_scatter_seed": 20260909, "f_spread_display_max_deg": 35.0,
                       "pa_histogram_bins": 26}.items():
        config.setdefault(key, value)
    if len(config["angles_deg"]) != 11:
        raise ValueError("This acquisition requires exactly 11 configured angles.")
    positive = ["pixel_size_um", "local_window_um", "correlation_bin_um", "correlation_max_um",
                "orientation_max_formal_se_deg", "wavelength_nm", "section_thickness_um",
                "f_spread_display_max_deg"]
    for key in positive:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 < config["minimum_window_coverage"] <= 1:
        raise ValueError("minimum_window_coverage must be in (0,1].")
    if config["orientation_bins"] < 2 or config["depth_quantiles"] < 2:
        raise ValueError("Histogram and depth profiles require at least two bins.")
    if config["depth_quantiles"] != 5 or config["spatial_sectors"] != 8:
        raise ValueError("The current visual figure contract uses five depth regions and eight azimuth sectors.")
    if (config["spatial_cell_min_pixels"] < 1 or config["direction_spacing_um"] <= 0
            or config["direction_length_um"] <= 0):
        raise ValueError("Spatial support count and direction glyph dimensions must be positive.")
    for panel_key in ("f", "h"):
        if (type(config[f"{panel_key}_scatter_count"]) is not int or config[f"{panel_key}_scatter_count"] < 1
                or type(config[f"{panel_key}_scatter_seed"]) is not int or config[f"{panel_key}_scatter_seed"] < 0):
            raise ValueError("Scatter count must be a positive integer and seed a nonnegative integer.")
    if type(config["pa_histogram_bins"]) is not int or config["pa_histogram_bins"] < 2:
        raise ValueError("PA histogram bin count must be an integer of at least two.")
    for key in ["pa_native_row_pitch_um", "pa_native_column_pitch_um"]:
        if config[key] is not None and config[key] <= 0:
            raise ValueError(f"{key} must be positive or null.")
    config["config_root"] = str(path.parent)
    return config


def resolve(config, key):
    value = Path(config[key]).expanduser()
    return value if value.is_absolute() else Path(config["config_root"])/value


def validate_input_paths(config):
    """Require configured acquisition files before any output is created."""
    keys = ["raw_image_dir", "pa_csv"] + (["mask_npy"] if config.get("mask_npy") else [])
    if any(not config.get(key) or "your project" in Path(config[key]).parts for key in keys):
        raise ValueError(
            "Configure raw_image_dir and pa_csv in --config: replace 'your project/dataset' "
            "with your dataset path. Relative input paths are resolved beside the configuration file.")
    required = [resolve(config, "raw_image_dir") / f"{k}.tif" for k in range(1, 12)]
    required.append(resolve(config, "pa_csv"))
    if config.get("mask_npy"):
        required.append(resolve(config, "mask_npy"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input file is missing: " + missing[0])


def ome_metadata(path):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        description = tif.pages[0].description
        root = ET.fromstring(description)
        result = {}
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag in ("Pixels", "Objective", "Detector"):
                result[tag] = dict(element.attrib)
        return result


def read_acquisition(config):
    import cv2
    import numpy as np

    validate_input_paths(config)
    x0, y0, x1, y1 = config["roi_xyxy"]
    images, sources = [], []
    shape = None
    for k in range(1, 12):
        path = resolve(config, "raw_image_dir")/f"{k}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Required acquisition frame is missing: {path}")
        if config["channel"] == "gray":
            a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        elif config["channel"] == "green":
            rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            a = rgb[..., 1] if rgb is not None and rgb.ndim == 3 else None
        else:
            raise ValueError("channel must be gray or green.")
        if a is None:
            raise ValueError(f"Could not read frame {k}.")
        if shape is not None and a.shape != shape:
            raise ValueError("All acquisition frames must share one image shape.")
        shape = a.shape
        if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
            raise ValueError("Configured ROI is outside the acquired frame.")
        images.append(a[y0:y1, x0:x1])
        sources.append({"file": path.name, "angle_deg": config["angles_deg"][k-1],
                        "sha256": sha256(path), "file_bytes": path.stat().st_size})
    metadata = ome_metadata(resolve(config, "raw_image_dir")/"1.tif")
    stack = np.stack(images)
    pa_path = resolve(config, "pa_csv")
    pa = np.loadtxt(pa_path, delimiter=",")
    if pa.ndim != 2 or np.isinf(pa).any():
        raise ValueError("PA CSV must be a 2D matrix with finite numbers or NaN.")
    metadata.update({"full_shape": list(shape), "roi_shape": list(stack.shape[1:]),
                     "frame_sources": sources, "pa_source_sha256": sha256(pa_path),
                     "pa_raw_shape": list(pa.shape)})
    return stack, pa, metadata


def write_csv(path, rows):
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
