"""Figure 3: plot layout."""
from __future__ import annotations

import numpy as np


FIGURE_BUNDLES = {
    "boundary_phenotyping": {"folder": "01_边界空间表型", "title": "以 Aβ 边界为参照的局部 LAMP1 表型", "width_mm": 183,
        "panels": dict(zip("abcdefghi", ["Aβ原始信号", "LAMP1原始信号", "信号掩膜与固定窗口", "边界距离坐标",
            "边界分布与强度离散", "径向角向组织", "LAMP1信号斑片", "双通道信号关系", "特征可测量性"]))},
    "abeta_organization": {"folder": "02_Aβ自身构型", "title": "Aβ 自身的二维几何与信号组织", "width_mm": 183,
        "panels": dict(zip("abcdef", ["像素单元几何", "内向深度与掩膜孔隙", "Aβ强度分布", "内部信号梯度", "多尺度方向纹理", "无量纲几何指标"]))},
    "lamp1_spatial_architecture": {"folder": "03_LAMP1空间表型", "title": "固定窗口内的 LAMP1 空间分布与信号斑片", "width_mm": 183,
        "panels": dict(zip("abcdef", ["窗口内信号斑片", "径向角向密度", "环带强度云雨图", "外区角向分布", "斑片面积经验分布", "环带面积组成"]))},
    "channel_relationships": {"folder": "04_双通道关系", "title": "Aβ 与 LAMP1 的空间共现及强度关系", "width_mm": 183,
        "panels": dict(zip("abcdef", ["二值空间共现", "信号散点与边缘分布", "不同统计域的相关性", "重叠与覆盖系数", "Aβ内外LAMP1强度", "外邻域组成气泡图"]))},
    "feature_atlas_X_A": {"folder": "05_Aβ特征图谱", "title": "Aβ 候选特征与参数敏感性完整图谱", "width_mm": 183, "panels": {}},
    "feature_atlas_Y_L": {"folder": "06_LAMP1特征图谱", "title": "LAMP1 读出与窗口质控完整图谱", "width_mm": 183, "panels": {}},
    "feature_atlas_Z_AL": {"folder": "07_联合特征图谱", "title": "双通道联合描述与敏感性完整图谱", "width_mm": 183, "panels": {}},
    "blue_Aβ_intensity_strata": {"folder": "08_Aβ强度分层", "title": "Aβ 前景信号的排序强度分层", "width_mm": 100, "panels": {}},
    "red_LAMP1_intensity_strata": {"folder": "09_LAMP1强度分层", "title": "固定窗口内 LAMP1 前景的排序强度分层", "width_mm": 100, "panels": {}},
    "manuscript_figure": {"folder": "10_正文整合图", "title": "单斑块构型与边界参照的 LAMP1 多维表型", "width_mm": 183,
        "panels": dict(zip("abcdefghi", ["原始双通道合并", "Aβ强度分层", "LAMP1强度分层", "几何形态与边界距离",
            "边界分布与强度离散", "无量纲几何指标", "径向角向密度", "信号散点与边缘分布", "重叠与双向覆盖"]))},
}


def figure_bundle_base(plaque_dir, key):
    spec = FIGURE_BUNDLES[key]
    folder = plaque_dir/spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder/("组合图" if spec["panels"] else "单图")


def save_individual_panels(fig, base, dpi):
    """Export original artists, not bitmap crops; keep ticks, colourbars and local notes."""
    from matplotlib.transforms import Bbox
    spec = next(v for v in FIGURE_BUNDLES.values() if v["folder"] == base.parent.name)
    axes, texts = list(fig.axes), list(fig.texts)
    original_visibility = {artist: artist.get_visible() for artist in axes+texts}
    folder = base.parent/"子图"
    folder.mkdir(exist_ok=True)
    try:
        for letter in fig._panel_order:
            selected_axes = [a for a in axes if getattr(a, "_figure_panel", None) == letter]
            selected_text = [t for t in texts if getattr(t, "_figure_panel", None) == letter]
            if not selected_axes:
                raise ValueError(f"子图 {letter} 没有可导出的绘图区。")
            for artist in axes+texts:
                artist.set_visible(original_visibility[artist] and artist in selected_axes+selected_text)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = [a.get_tightbbox(renderer) for a in selected_axes]
            bounds += [t.get_window_extent(renderer) for t in selected_text if t.get_visible() and t.get_text().strip()]
            bounds = [b for b in bounds if b is not None and np.all(np.isfinite(b.extents))]
            bbox = Bbox.union(bounds).transformed(fig.dpi_scale_trans.inverted()).padded(1.8/25.4)
            key = next(k for k, v in FIGURE_BUNDLES.items() if v is spec)
            note = PANEL_STANDALONE_NOTES.get((key, letter))
            extra = None
            if note:
                fw, fh = fig.get_size_inches()
                extra = fig.text((bbox.x0+1.8/25.4)/fw, (bbox.y0-1/25.4)/fh, note,
                                 fontsize=5.6, va="top", linespacing=1.35)
                fig.canvas.draw()
                note_bounds = extra.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                bbox = Bbox.union([bbox, note_bounds.padded(1.8/25.4)])
            name = folder/f"{letter}_{spec['panels'][letter]}"
            try:
                for suffix in (".svg", ".png"):
                    fig.savefig(str(name)+suffix, dpi=dpi, bbox_inches=bbox, pad_inches=0)
            finally:
                if extra is not None:
                    extra.remove()
    finally:
        for artist, visible in original_visibility.items():
            artist.set_visible(visible)
        fig.canvas.draw()


PANEL_STANDALONE_NOTES = {
    ("boundary_phenotyping", "a"): "Gold contour: effective intensity ceiling.",
    ("boundary_phenotyping", "d"): "Black: pixel-cell boundary; gold: raster-mask hull.\nDashed: geometric axes; negative distance is inside Aβ.",
    ("boundary_phenotyping", "g"): "Blue: Aβ; dashed: fixed window.\nTone encodes log10 image-patch area.",
    ("boundary_phenotyping", "h"): "Blue: Aβ only; terracotta: LAMP1 only.\nPurple: double-positive; grey: neither.\nGold dashed line: intensity ceiling.",
    ("boundary_phenotyping", "i"): "Dark: computed; grey: window QC.\nGold: QC-withheld, not a zero value.",
    ("abeta_organization", "b"): "Gold outlines: holes in the segmentation mask.",
    ("abeta_organization", "d"): "Blue: Aβ boundary; grey: excluded stencils.",
    ("lamp1_spatial_architecture", "a"): "Blue: Aβ; dashed: fixed window.\nPatches are image components, not organelles.",
    ("lamp1_spatial_architecture", "b"): "×: incomplete or <3 observed pixels.\nAngle increases clockwise from the right.",
    ("lamp1_spatial_architecture", "c"): "Blue: inside Aβ; terracotta: outside Aβ.\nIQR describes pixels, not a confidence interval.",
    ("lamp1_spatial_architecture", "d"): "Open points: incomplete sectors.",
    ("channel_relationships", "b"): "Blue: Aβ only; terracotta: LAMP1 only.\nPurple: double-positive; grey: neither.\nGold: ceiling; signal relationships are diagnostic.",
    ("channel_relationships", "c"): "Open symbols: QC-withheld diagnostic values.",
    ("channel_relationships", "f"): "Bubble area: proportion.\nColour: proportion minus window baseline.\nIntensity-rank relationships are diagnostic.",
}


def figure_axis(fig, x, y, width, height, **kwargs):
    """Place axes in millimetres, independent of export DPI."""
    fw, fh = fig.get_size_inches()*25.4
    ax = fig.add_axes([x/fw, y/fh, width/fw, height/fh], **kwargs)
    ax._figure_panel = getattr(fig, "_active_panel", None)
    return ax


def figure_text(fig, x, y, text, *, panel="current", **kwargs):
    fw, fh = fig.get_size_inches()*25.4
    artist = fig.text(x/fw, y/fh, text, **kwargs)
    artist._figure_panel = getattr(fig, "_active_panel", None) if panel == "current" else panel
    return artist


def panel_heading(fig, letter, title, x, y):
    fig._active_panel = letter
    if not hasattr(fig, "_panel_order"):
        fig._panel_order = []
    fig._panel_order.append(letter)
    figure_text(fig, x, y, letter, fontsize=8, weight="bold", va="bottom")
    figure_text(fig, x+4, y, title, fontsize=7, va="bottom")


def style_axis(ax, *, grid=False):
    from matplotlib.ticker import MaxNLocator
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if grid:
        ax.grid(axis="y", color="#E6E6E8", linewidth=.35)
    ax.set_axisbelow(True)


def plot_value(value, digits=3):
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"

