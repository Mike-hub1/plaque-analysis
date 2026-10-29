"""Figure 3: czi io."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import VERSION
from plaque_analysis.io.fluorescence_utils import array_sha256, sha256_file
from pathlib import Path
from skimage.measure import label as sk_label
import numpy as np
import tifffile


def xml_value(root, path, numeric=False):
    element = root.find(path)
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    if numeric:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def read_mask(path, shape, *, labels=False):
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"掩膜须是同尺寸二维矩阵 {shape}: {path}, 实际 {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"掩膜含负数/非有限数: {path}")
    if labels:
        if not np.all(array == np.floor(array)):
            raise ValueError("Aβ 对象掩膜必须使用非负整数标签。")
        values = np.unique(array[array > 0])
        if len(values) == 1:
            return sk_label(array > 0, connectivity=2).astype(np.int32)
        return array.astype(np.int32)
    return array > 0


def _native_plane(czi, selections):
    array, _ = czi.read_image(**selections)
    array = np.asarray(array).squeeze()
    if array.ndim != 2 or array.dtype.kind not in "uif" or not np.all(np.isfinite(array)):
        raise ValueError("读取结果必须为唯一二维、有限、原生灰度 detector 矩阵。")
    return np.array(array, copy=True)


def _read_channel_verified(czi, selections):
    """libCZI mosaic 输出逐像素与原生 M 子块覆盖重建核对，不自行平均重叠值。"""
    if not czi.is_mosaic():
        raw = _native_plane(czi, selections)
        return raw, np.ones(raw.shape, bool), {
            "mode": "native_plane", "origin_xy": [0, 0],
            "tiles": [], "native_subblock_check": "not_needed",
        }
    boxes = czi.get_all_mosaic_tile_bounding_boxes(**selections)
    if not boxes:
        raise ValueError("未找到所选 detector plane 的原始图块。")
    tiles = []
    for info, box in boxes.items():
        raw_tile = _native_plane(czi, {**selections, "M": info.m_index})
        if raw_tile.shape != (box.h, box.w):
            raise ValueError("图块矩阵与原生 bbox 不符；不执行隐式缩放。")
        tiles.append((int(info.m_index), box, raw_tile))
    tiles.sort(key=lambda t: t[0])
    if len({t[0] for t in tiles}) != len(tiles):
        raise ValueError("同一 M 编号对应多个未限定平面，请明确维度选择。")
    x0, y0 = min(t[1].x for t in tiles), min(t[1].y for t in tiles)
    x1 = max(t[1].x+t[1].w for t in tiles)
    y1 = max(t[1].y+t[1].h for t in tiles)
    raw = np.asarray(czi.read_mosaic(region=(x0, y0, x1-x0, y1-y0),
                                    scale_factor=1.0, **selections)).squeeze()
    if raw.shape != (y1-y0, x1-x0):
        raise ValueError("mosaic 尺寸与原始坐标不一致。")
    coverage = np.zeros(raw.shape, np.uint16)
    reconstruction = {}
    for name, ordered in (("higher_M_priority", tiles), ("lower_M_priority", tiles[::-1])):
        canvas = np.zeros_like(raw)
        for mi, b, tile in ordered:
            ys, xs = slice(b.y-y0, b.y-y0+b.h), slice(b.x-x0, b.x-x0+b.w)
            canvas[ys, xs] = tile
            if name == "higher_M_priority":
                coverage[ys, xs] += 1
        reconstruction[name] = canvas
    mismatch = {k: int(np.count_nonzero(v != raw)) for k, v in reconstruction.items()}
    matches = [k for k, n in mismatch.items() if n == 0]
    if not matches:
        raise ValueError("libCZI 输出不匹配已验证的原生图块覆盖规则；需要无拉伸 ZEN 参考导出。")
    return np.array(raw, copy=True), coverage > 0, {
        "mode": "libCZI_read_mosaic_scale_1", "origin_xy": [x0, y0],
        "native_subblock_check": "pixel_exact", "matching_overlap_policies": matches,
        "mismatch_counts": mismatch, "overlap_pixels": int(np.count_nonzero(coverage > 1)),
        "tiles": [{"M": mi, "x": b.x, "y": b.y, "width": b.w, "height": b.h,
                   "dtype": str(t.dtype), "array_sha256": array_sha256(t),
                   "container_max_count": int(np.count_nonzero(t == np.iinfo(t.dtype).max))
                   if t.dtype.kind in "ui" else None} for mi, b, t in tiles],
    }


def load_czi_input_channels(czi_path, channel_specs, *, scene_index=0, time_index=0,
                            z_index=0, spacing_override=None):
    """在同一原生坐标系读取指定通道；每项为（生物学名称、C 索引、可选参考 TIFF）。"""
    from aicspylibczi import CziFile
    path = Path(czi_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".czi":
        raise ValueError(f"请输入存在的原始 CZI 文件: {path}")
    czi = CziFile(path)
    dims = dict(zip(czi.dims, map(int, czi.size)))
    indices = [index for _, index, _ in channel_specs]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("须提供不同的 detector channel，不能将同一通道重复命名。")
    selections = {}
    for dim, index in (("S", scene_index), ("T", time_index), ("Z", z_index)):
        if index < 0 or (dim not in dims and index != 0) or index >= dims.get(dim, 1):
            raise ValueError(f"无效 {dim} 索引: {index}")
        if dim in dims:
            selections[dim] = index
    for dim in czi.dims:
        if dim not in "CSTZMYX":
            if dims[dim] != 1:
                raise ValueError(f"存在未显式限定的 {dim} 维度；禁止隐式投影。")
            selections[dim] = 0
    raw, valid, channel_info = [], [], []
    channels = czi.meta.findall(".//Information/Image/Dimensions/Channels/Channel")
    for role, index, reference in channel_specs:
        if not 0 <= index < dims.get("C", 1):
            raise ValueError(f"无效 C 索引: {index}")
        array, observed, trace = _read_channel_verified(czi, {**selections, "C": index})
        if reference is not None:
            ref = tifffile.imread(reference)
            if ref.dtype != array.dtype or not np.array_equal(ref, array):
                raise ValueError(f"{role} 与提供的无拉伸参考 TIFF 不逐像素一致。")
            trace["external_reference"] = {"status": "pixel_exact", "sha256": sha256_file(reference)}
        else:
            trace["external_reference"] = {"status": "not_provided"}
        ch = channels[index] if index < len(channels) else czi.meta
        info = {
            "role": role, "channel_index": index,
            "biological_identity_source": "user_staining_record",
            "fluor_template": xml_value(ch, "Fluor"),
            "channel_name": ch.get("Name"),
            "excitation_nm": xml_value(ch, "ExcitationWavelength", True),
            "detection_window_nm": xml_value(ch, "DetectionWavelength/Ranges"),
            "gain": xml_value(ch, "DetectorSettings/Gain", True),
            "digital_gain": xml_value(ch, "DetectorSettings/DigitalGain", True),
            "offset": xml_value(ch, "DetectorSettings/Offset", True),
            "pinhole_airy": xml_value(ch, "PinholeSizeAiry", True),
            "pixel_dwell_time_s": xml_value(ch, "LaserScanInfo/PixelTime", True),
            "effective_bits_metadata": xml_value(ch, "ComponentBitCount", True),
            "dtype": str(array.dtype), "shape_yx": list(array.shape),
            "raw_min": float(array.min()), "raw_max": float(array.max()),
            "array_sha256": array_sha256(array), **trace,
        }
        if info["effective_bits_metadata"] is None:
            info["effective_bits_metadata"] = xml_value(czi.meta, ".//Information/Image/ComponentBitCount", True)
        raw.append(array); valid.append(observed); channel_info.append(info)
    if any(a.shape != raw[0].shape or info["origin_xy"] != channel_info[0]["origin_xy"]
           for a, info in zip(raw[1:], channel_info[1:])):
        raise ValueError("所选通道尺寸/原点不同，需事先独立配准；不自动缩放或拉伸。")
    scale = {}
    for d in czi.meta.findall(".//Scaling/Items/Distance"):
        v = xml_value(d, "Value", True)
        if v is not None and v > 0:
            scale[d.get("Id")] = v*1e6
    spacing = tuple(spacing_override) if spacing_override else (scale.get("Y"), scale.get("X"))
    if any(s is None or not np.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("缺少有效 μm/pixel 标定；请给出 --pixel-size-x-um/--pixel-size-y-um。")
    objective = czi.meta.find(".//Objectives/Objective")
    manifest = {
        "version": VERSION, "input_file": str(path), "input_sha256": sha256_file(path),
        "axes": dims, "selection": selections, "channels": channel_info,
        "spacing_yx_um": spacing, "spacing_source": "user_override" if spacing_override else "CZI_Scaling",
        "objective_model": xml_value(objective, "Manufacturer/Model") if objective is not None else None,
        "numerical_aperture": xml_value(objective, "LensNA", True) if objective is not None else None,
        "registration": "native_common_grid; chromatic_registration_not_independently_validated",
        "analysis_dimension": "2D_selected_plane", "third_channel_used": False,
    }
    return raw, valid, spacing, manifest


def load_czi_input_pair(czi_path, abeta_channel=0, lamp1_channel=2, *,
                        scene_index=0, time_index=0, z_index=0, spacing_override=None,
                        reference_abeta=None, reference_lamp1=None):
    """保留原两通道分析接口；第三通道不会进入特征计算。"""
    return load_czi_input_channels(czi_path,
        (("Aβ/CR", abeta_channel, reference_abeta), ("LAMP1", lamp1_channel, reference_lamp1)),
        scene_index=scene_index, time_index=time_index, z_index=z_index, spacing_override=spacing_override)


def validate_raw_channel_wavelengths(channel_info):
    """核对用户指定的染色映射；缺少波长时记录未核实，不根据荧光模板猜蛋白。"""
    if len(channel_info) != 3:
        raise ValueError("原始图导出需要依次提供 Aβ、LAMP1、tau 三个通道。")
    if len({c["channel_index"] for c in channel_info}) != 3:
        raise ValueError("Aβ、LAMP1 和 tau 不能复用同一个原始通道。")
    rows = []
    for name, expected, info in zip(("Aβ", "LAMP1", "tau"), (561., 405., 488.), channel_info):
        actual = info.get("excitation_nm")
        if actual is not None and (not np.isfinite(actual) or abs(actual-expected) > 2):
            raise ValueError(f"{name} 所选 C{info['channel_index']} 的激发波长为 {actual} nm，"
                             f"与用户指定的 {expected:g} nm 不符；请核对通道参数和染色记录。")
        rows.append({"channel": name, "channel_index": info["channel_index"],
                     "expected_excitation_nm": expected, "metadata_excitation_nm": actual,
                     "status": "matched" if actual is not None else "metadata_missing_user_selection_only"})
    return rows

