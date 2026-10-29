"""Figure 3: plot raw."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import FIGURE_FONT, VERSION
from plaque_analysis.io.czi import validate_raw_channel_wavelengths
from plaque_analysis.plotting.fluorescence_guides import write_figures_index
from plaque_analysis.plotting.fluorescence_helpers import add_scale_bar, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text
from plaque_analysis.plotting.fluorescence_style import RAW_CHANNEL_FIGURE_FOLDER, RAW_CHANNEL_NAMES, RAW_CHANNEL_RGB, RAW_CHANNEL_VIEWS, configure_plotting, save_figure
from plaque_analysis.analysis.fluorescence_quality import saturation_qc
from plaque_analysis.io.fluorescence_utils import array_sha256, write_json
import json
import numpy as np


def prepare_raw_channel_displays(raw, valid, channel_info, effective_bits=(None, None, None)):
    """只建立原始数据的显示副本；不扣背景、不分割、不使用百分位拉伸。"""
    if not (len(raw) == len(valid) == len(channel_info) == len(effective_bits) == 3):
        raise ValueError("须同时提供 Aβ、LAMP1、tau 的三组原始矩阵、观测掩膜和位深信息。")
    planes, rows = [], []
    for name, array, observed, info, bits in zip(RAW_CHANNEL_NAMES, raw, valid, channel_info, effective_bits):
        array, observed = np.asarray(array), np.asarray(observed, dtype=bool)
        if array.ndim != 2 or array.shape != raw[0].shape or observed.shape != array.shape:
            raise ValueError("原始通道导出要求同尺寸二维矩阵与原生观测掩膜。")
        if not observed.any() or not np.all(np.isfinite(array)):
            raise ValueError("原始通道没有有效观测或含非有限值。")
        sat = saturation_qc(array, observed, bits, info.get("effective_bits_metadata"))
        lower = 0.
        upper = sat["ceiling"]
        policy = "effective_bit_depth_range" if sat["effective_bits_known"] else "container_range_effective_bits_unknown"
        if upper is None:  # 原生浮点无整数容器上限；显式记录全视野线性显示范围。
            lower = min(0., float(array[observed].min()))
            upper = max(0., float(array[observed].max()))
            if upper <= lower:
                upper = lower+1.
            policy = "floating_point_observed_range"
        elif float(array[observed].min()) < 0:
            raise ValueError("整数原始探测器值不能为负。")
        plane = np.clip((array.astype(np.float64)-lower)/(upper-lower), 0., 1.)
        plane[~observed] = 0.  # 显示时另将未观测位置标灰，不把未观测误作阴性。
        planes.append(plane)
        rows.append({"channel": name, "channel_index": info["channel_index"],
            "excitation_nm": info.get("excitation_nm"), "dtype": str(array.dtype),
            "shape_yx": list(array.shape), "native_array_sha256": array_sha256(array),
            "native_observed_pixels": int(observed.sum()), "display_min": lower, "display_max": upper,
            "display_policy": policy, "gamma": 1., "background_subtraction": False,
            "segmentation_applied": False, "percentile_stretch": False,
            "pseudocolor_rgb": list(RAW_CHANNEL_RGB[len(rows)]),
            "effective_bits": sat["effective_bits"], "bit_depth_source": sat["bit_depth_source"],
            "ceiling_pixels_observed": int((sat["mask"] & observed).sum()),
            "ceiling_fraction_observed": float((sat["mask"] & observed).sum()/observed.sum())})
    views = {}
    for filename, indices in RAW_CHANNEL_VIEWS:
        rgb = np.zeros((*raw[0].shape, 3), dtype=np.float64)
        for ci in indices:
            rgb += planes[ci][..., None]*np.asarray(RAW_CHANNEL_RGB[ci])
        # 三个通道各占一个 RGB 分量，避免加色后再次归一化或提高亮度。
        observed = np.logical_and.reduce([valid[ci] for ci in indices])
        views[filename] = {"rgb": rgb, "observed": observed,
                           "unobserved_pixels": int((~observed).sum())}
    return rows, views


def render_raw_channel_figures(raw, valid, spacing, channel_info, acquisition, output, dpi,
                              effective_bits=(None, None, None), *, preserve_manifest=False):
    """一份 CZI 所选平面仅导出一次，不因检出多个斑块而复制整幅视野。"""
    wavelength_check = validate_raw_channel_wavelengths(channel_info)
    rows, views = prepare_raw_channel_displays(raw, valid, channel_info, effective_bits)
    plt = configure_plotting()
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    folder = output/"figures"/RAW_CHANNEL_FIGURE_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    export_rows = []
    h, w = raw[0].shape
    sy, sx = spacing
    extent = (-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2)
    for filename, indices in RAW_CHANNEL_VIEWS:
        fig = plt.figure(figsize=(100/25.4, 96/25.4))
        title = " + ".join(RAW_CHANNEL_NAMES[i] for i in indices)
        figure_text(fig, 3, 91, title, fontsize=8, weight="bold")
        if len(indices) == 1:
            ci = indices[0]
            nm = channel_info[ci].get("excitation_nm")
            excitation = f"{nm:g} nm" if nm is not None else "excitation unverified"
            subtitle = f"C{channel_info[ci]['channel_index']} | {excitation} | native {w} × {h} pixels"
        else:
            subtitle = f"Native channel merge | {w} × {h} pixels | common CZI grid"
        figure_text(fig, 3, 86, subtitle, fontsize=5.8)
        ax = figure_axis(fig, 3, 22, 94, 62)
        view = views[filename]
        shown = view["rgb"].copy()
        shown[~view["observed"]] = 189/255
        # 单通道和合并图直接使用同一线性 RGB 分量，避免独立 LUT 映射改变亮度。
        ax.imshow(shown, interpolation="nearest", extent=extent)
        ax.axis("off")
        missing = view["unobserved_pixels"]
        if len(indices) == 1:
            cmap = LinearSegmentedColormap.from_list(f"raw_channel_{ci}", [(0., 0., 0.), RAW_CHANNEL_RGB[ci]])
            mappable = plt.cm.ScalarMappable(
                norm=Normalize(vmin=rows[ci]["display_min"], vmax=rows[ci]["display_max"]), cmap=cmap)
            small_colorbar(fig, mappable, (22, 14, 56, 2), "Raw detector signal (a.u.)",
                           ticks=[rows[ci]["display_min"], rows[ci]["display_max"]])
        else:
            for ci, x in zip(indices, (6, 31, 63)):
                word, color = (("blue", "#0000BF"), ("red", "#BF0000"), ("green", "#008000"))[ci]
                figure_text(fig, x, 14, f"{RAW_CHANNEL_NAMES[ci]} ({word})", fontsize=6, color=color)
        add_scale_bar(ax, raw[0].shape, spacing)
        figure_text(fig, 3, 3, "Raw detector data | linear display | no background subtraction", fontsize=5.4)
        if missing:
            figure_text(fig, 3, 19, "Grey: at least one required channel is unobserved", fontsize=5.2)
        save_figure(fig, folder/filename, dpi)
        export_rows.append({"file_stem": filename, "channels": [RAW_CHANNEL_NAMES[i] for i in indices],
                            "display_mode": "single_channel_pseudocolor" if len(indices) == 1 else "additive_merge",
                            "unobserved_pixels_marked_grey": missing,
                            "formats": ["svg", "png"], "png_dpi": dpi})
    report = {"version": VERSION, "status": "exported", "source_sha256": acquisition["input_sha256"],
        "selection": acquisition["selection"], "spacing_yx_um": spacing,
        "source_channels": channel_info, "wavelength_check": wavelength_check,
        "display_channels": rows, "exports": export_rows,
        "tau_used_for_feature_analysis": False, "registration": acquisition["registration"]}
    if not preserve_manifest:
        write_json(output/"raw_channel_export_manifest.json", report)
    lines = ["# CZI 原始通道与合并图：中文说明", "",
        "本目录展示本轮所选 CZI 平面的原始探测器信号。三张单通道均为伪彩图：Aβ 蓝色、LAMP1 红色、tau 绿色，色条分别由黑色过渡至对应通道颜色；两张合并图使用同一组颜色与强度范围进行加色。图像全部来自背景校正之前的矩阵，不读取旧 RGB 截图，不执行阈值分割、平滑、去噪、百分位拉伸或伽马增强。", "",
        "## 文件与用途", "", "| 图像 | PNG | SVG | 显示内容 |", "| --- | --- | --- | --- |"]
    for filename, indices in RAW_CHANNEL_VIEWS:
        meaning = ("原始信号的"+("蓝色", "红色", "绿色")[indices[0]]+"伪彩显示") if len(indices) == 1 else "蓝色为 Aβ、红色为 LAMP1"+("、绿色为 tau" if len(indices) == 3 else "")
        lines.append(f"| {'＋'.join(RAW_CHANNEL_NAMES[i] for i in indices)} | [PNG]({filename}.png) | [SVG]({filename}.svg) | {meaning} |")
    lines += ["", "## 通道、标定与显示范围", "",
        "通道的生物学名称采用用户染色记录：561 nm 对应 Aβ，405 nm 对应 LAMP1，488 nm 对应 tau。颜色仅是显示编码，不是激发波长对应的物理颜色，也不从软件荧光模板推断蛋白身份。", "",
        f"本轮原生矩阵为 {h}×{w} 像素；纵横间距分别为 {sy:.6f}、{sx:.6f} 微米。所选维度为 {json.dumps(acquisition['selection'], ensure_ascii=False)}；没有进行 Z 投影或时间平均。所有图保留相同完整视野、方向、长宽比和物理比例尺，不按斑块掩膜或病理窗口裁切。", "",
        "| 通道 | 原始索引 | 激发波长（nm） | 原始类型 | 线性显示下限 | 线性显示上限 | 全视野已观测像素中达到上限的比例 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        nm = row["excitation_nm"]
        lines.append(f"| {row['channel']} | C{row['channel_index']} | {nm:.4g} | {row['dtype']} | {row['display_min']:g} | {row['display_max']:g} | {row['ceiling_fraction_observed']:.2%} |" if nm is not None else
                     f"| {row['channel']} | C{row['channel_index']} | 元数据缺失，未核实 | {row['dtype']} | {row['display_min']:g} | {row['display_max']:g} | {row['ceiling_fraction_observed']:.2%} |")
    lines += ["", "整数通道默认按零至有效位深上限线性显示；若有效位深未知，则仅用整数容器上限显示，不据此宣称已知真实有效位深。原生浮点通道没有整数位深上限时，使用全视野已观测范围并记录。归一化仅发生在显示副本，单通道与合并图使用同一组范围，合并后不再次拉伸。", "",
        "## 五幅图的具体解释", "",
        "- Aβ：原始 561 nm 通道的蓝色伪彩图，蓝色分量线性对应本通道的探测器记录。亮核心可能达到存储或有效位深上限；本导出不能恢复被截顶的强度差异。",
        "- LAMP1：原始 405 nm 通道的红色伪彩图，包含尚未背景校正的低强度信号和背景。不能将本图中的所有非零像素直接当作 LAMP1 阳性病理面积。",
        "- tau：原始 488 nm 通道的绿色伪彩图，仅新增显示与原始数据保存，不进入现有 Aβ、LAMP1 特征表、分割、结局或模型。",
        "- 双通道合并：蓝色分量为 Aβ、红色分量为 LAMP1；紫色仅表示该显示位置存在两通道信号。合并图本身不证明共定位、物理接触或核心—晕环机制。",
        "- 三通道合并：在同一双通道图上添加绿色 tau 分量。Aβ 与 LAMP1 的显示范围和位置保持不变，不能把混合颜色当作新病理类型。", "",
        "## 图像完整性与使用限制", "",
        "1. CZI 原生图块拼接沿用逐像素核对的规则；不会平均重叠图块。共同像素网格不等于完成独立的色差配准验证。",
        "2. 原生未观测位置在单通道标灰；合并图只有所需通道均已观测的像素才参与加色，其余标灰，不把缺失当作阴性。各图标灰像素数记录在导出清单。",
        "3. 保留低强度背景是原始导出的预期行为。任何后续背景校正或掩膜分析均属于独立的分析分支，不能把已处理分层图称为原始荧光图。",
        "4. 相同颜色亮度不等于相同蛋白浓度；不同通道的激发、增益、标记效率和探测特性不同。",
        "5. 当前 PNG 和 SVG 是带文字、比例尺及显示映射的展示文件，不是无损原始强度矩阵。数值分析应读取 CZI 或 matrices 中的原生 TIFF；SVG 内显微图像仍是栅格数据，文字、色条和比例尺可编辑。", "",
        f"图形格式仅为 SVG 与 {dpi} DPI PNG。图中英文、数字及希腊字母采用 {FIGURE_FONT}。最近邻显示不会增加显微采样或修复饱和。", "",
        "- [原始通道导出清单](../../raw_channel_export_manifest.json)：通道映射、显示范围、来源散列、原生图块核对及未观测计数。",
        "- [本轮采集与分析清单](../../acquisition_manifest.json)：原始文件、维度选择和运行参数。",
        "- [Aβ 原始矩阵](../../matrices/blue_Aβ_raw_detector.tif) · [LAMP1 原始矩阵](../../matrices/red_LAMP1_raw_detector.tif) · [tau 原始矩阵](../../matrices/tau_raw_detector.tif)。", ""]
    (folder/"说明.md").write_text("\n".join(lines), encoding="utf-8")
    write_figures_index(output/"figures")
    return report

