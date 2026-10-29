"""Figure 3: plot publication."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import FIGURE_FONT
from plaque_analysis.plotting.fluorescence_helpers import add_scale_bar, draw_boundary_distributions, draw_channel_scatter, draw_geometry_boundary, ring_angle_heatmap, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, apply_nature_text_policy, save_figure
import numpy as np


def render_publication_figure(plt, data, raw, signed, masks, valid, geometry, saturation,
                              spacing, config, ranks, stratum_rows, base, dpi, *, a4=False):
    """正文 183×170 mm；A4 校样保持正文图的实际尺寸，不放大后再缩回。"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    page_w, page_h = (210., 297.) if a4 else (183., 170.)
    ox, oy = (13.5, 111.) if a4 else (0., 0.)
    fig = plt.figure(figsize=(page_w/25.4, page_h/25.4))
    def axis(x, y, w, h, **kwargs):
        return figure_axis(fig, x+ox, y+oy, w, h, **kwargs)
    def text(x, y, value, **kwargs):
        return figure_text(fig, x+ox, y+oy, value, **kwargs)
    def heading(letter, title, x, y):
        # 编号用于正文引用；描述性标题移至图注，避免占用小面板空间。
        fig._active_panel = letter
        if not hasattr(fig, "_panel_order"):
            fig._panel_order = []
        fig._panel_order.append(letter)
        text(x, y, letter, fontsize=8, weight="bold", va="bottom")
    def colourbar(im, box, label, ticks=None):
        x, y, w, h = box
        return small_colorbar(fig, im, (x+ox, y+oy, w, h), label, ticks=ticks)
    def key(x, y, colour, label):
        ax = axis(x, y-.6, 2.2, 1.5)
        ax.add_patch(Rectangle((0, 0), 1, 1, color=colour)); ax.axis("off")
        text(x+3.2, y, label, fontsize=5.5, va="center")

    heading("a", "Aβ–LAMP1 native merge", 1, 166)
    rgb = np.zeros((*raw[0].shape, 3), dtype=float)
    ranges = []
    for ci, component in ((0, 2), (1, 0)):
        ceiling = saturation[ci].get("ceiling")
        lower = 0. if ceiling is not None else min(0., float(raw[ci][valid[ci]].min()))
        upper = ceiling if ceiling is not None else max(0., float(raw[ci][valid[ci]].max()))
        if upper <= lower:
            upper = lower+1.
        ranges.append((lower, upper))
        rgb[..., component] = np.clip((raw[ci].astype(float)-lower)/(upper-lower), 0, 1)
    rgb[~(valid[0] & valid[1])] = 189/255
    sy, sx = spacing; h, w = raw[0].shape
    extent = (-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2)
    ax = axis(1, 124, 56, 38.5)
    ax.imshow(rgb, interpolation="nearest", extent=extent); ax.axis("off")
    scale_bar_um = add_scale_bar(ax, raw[0].shape, spacing)
    key(2, 120.5, "#0000FF", "Aβ"); key(22, 120.5, "#FF0000", "LAMP1")
    text(2, 116.5, "Raw detectors; no background subtraction", fontsize=5.5)
    raw_note = f"Linear {ranges[0][0]:g}–{ranges[0][1]:g} a.u." if ranges[0] == ranges[1] else "Channel-specific linear detector ranges"
    text(2, 113.5, raw_note, fontsize=5.3)
    for ci, letter, x, cmap_name in ((0, "b", 63.5, "Blues"), (1, "c", 126., "Oranges")):
        heading(letter, ("Aβ intensity strata", "LAMP1 intensity strata")[ci], x, 166)
        rows = stratum_rows[ci]; k = max(1, len(rows))
        colours = plt.get_cmap(cmap_name)(np.linspace(.25, .95, k))
        cmap = ListedColormap(colours); cmap.set_bad("#14141A")
        ax = axis(x, 124, 56, 38.5)
        ax.imshow(np.ma.masked_where(ranks[ci] < 1, ranks[ci]), interpolation="nearest", extent=extent,
                  cmap=cmap, norm=BoundaryNorm(np.arange(.5, k+1.5), k))
        ax.axis("off"); add_scale_bar(ax, raw[ci].shape, spacing)
        # 通道名是必要的身份标记，不是重复的描述性标题。
        text(x+2, 158.5, ("Aβ", "LAMP1")[ci], fontsize=6.5, color="white", va="center")
        legend = axis(x, 120, 56, 2.2)
        for j, colour in enumerate(colours):
            legend.add_patch(Rectangle((j/k, 0), 1/k, 1, fc=colour, ec="white", lw=.4))
        legend.set_xlim(0, 1); legend.set_ylim(0, 1); legend.axis("off")
        for j, r in enumerate(rows):
            text(x+(j+.5)*56/k, 117.2, f"{r['rank']}: {r['min']:.3g}–{r['max']:.3g}", ha="center", fontsize=5)
        if not rows:
            text(x, 117.2, "No positive pixels; strata not estimable", fontsize=5)
        note = (f"Ceiling {saturation[0]['saturated_fraction_mask']:.1%}; diagnostic strata" if ci == 0 and saturation[0]["mask"].any()
                else "Rank 1 = brightest; corrected a.u.")
        text(x, 113.5, note, fontsize=5.3)

    heading("d", "Geometry and boundary", 1, 105)
    ax = axis(1, 68, 46, 31.625)
    im = draw_geometry_boundary(ax, masks[0], geometry, data["window"], spacing, config.window_radius_um)
    colourbar(im, (6, 62, 36, 1.8), "Signed distance (μm)", ticks=[-10, 0, 10] if config.window_radius_um == 10 else None)
    legend = axis(1, 51, 46, 4); legend.axis("off")
    legend.legend(handles=[Line2D([], [], color="#25272C", lw=.65, label="Boundary"),
        Line2D([], [], color=FIGURE_COLORS["ceiling"], lw=.65, label="Hull"),
        Line2D([], [], color="#25272C", lw=.6, ls="--", label="Axes")],
        loc="center", ncol=3, fontsize=5.5, handlelength=1.3, handletextpad=.3,
        columnspacing=.8, borderaxespad=0)

    heading("e", "Boundary-resolved LAMP1", 54, 105)
    ax_density = axis(62, 90, 59, 12)
    ax_signal = axis(62, 62, 59, 24)
    draw_boundary_distributions(ax_density, ax_signal, data, marker_size=2.8)
    legend = axis(61, 49, 60, 4); legend.axis("off")
    legend.legend(handles=[Line2D([], [], color=FIGURE_COLORS["ink"], marker="o", ms=2.8,
        lw=1.1, label="Median / IQR"), Line2D([], [], color=FIGURE_COLORS["both"], marker="s",
        mfc="white", ms=2.5, ls="--", lw=.7, label="Mean")], loc="center", ncol=2,
        fontsize=5.3, handlelength=1.5, handletextpad=.5, columnspacing=1.2)

    heading("f", "Shape ratios", 126, 105)
    ax = axis(149, 64, 32, 38)
    keys = ["solidity", "bounding_box_extent", "circularity", "eroded_area_fraction", "mask_hole_area_fraction"]
    names = ["Solidity", "Box extent", "Circularity", "Eroded fraction", "Hole fraction"]
    lookup = data["master"].set_index("feature")
    for y, feature in enumerate(keys):
        value = lookup.loc[feature, "value"]
        if np.isfinite(value):
            ax.plot([0, value], [y, y], color=FIGURE_COLORS["A"], alpha=.4, lw=1.3)
            ax.plot(value, y, "o", color=FIGURE_COLORS["A"], ms=3.2)
            ax.text(value+.04, y-.22, f"{value:.3f}", fontsize=5.5, ha="left")
        else:
            ax.text(.5, y, "NA", fontsize=5, ha="center")
    style_axis(ax); ax.set_yticks(range(5), names, fontsize=5.8)
    ax.set_xlim(0, 1); ax.set_ylim(4.7, -.7); ax.set_xticks([0, .5, 1])
    ax.set_xlabel("Ratio", fontsize=6, labelpad=2); ax.tick_params(axis="y", pad=2)

    heading("g", "Radial × angular density", 1, 43)
    ax = axis(16, 16.5, 40, 22.5)
    im = ring_angle_heatmap(ax, data)
    ax.set_xlabel("Angle (°)", fontsize=5.5, labelpad=1.5)
    ax.set_ylabel("Distance (μm)", fontsize=5.5, labelpad=1.5)
    ax.set_xticks([0, 180, 360]); ax.tick_params(axis="y", labelsize=5, pad=2)
    labels = [label.get_text().replace(" to ", ":") for label in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks(), labels, fontsize=5)
    colourbar(im, (19, 6.3, 34, 1.5), "LAMP1-positive area fraction", ticks=[0, .5, 1])

    heading("h", "Signals (diagnostic)", 64, 43)
    ax = axis(75, 14, 38, 24)
    draw_channel_scatter(ax, data, saturation[0], raw[0].astype(float)-signed[0])
    # 787 像素等小型图保留矢量点，便于 Illustrator 编辑，不抽样。
    for collection in ax.collections:
        collection.set_rasterized(False)
    ax.set_xlabel("Aβ signal (a.u.)", fontsize=5.7, labelpad=1.5)
    ax.set_ylabel("LAMP1 (a.u.)", fontsize=5.7, labelpad=1.5)
    hist_x = axis(75, 39, 38, 3)
    hist_y = axis(114, 14, 3, 24)
    pixel = data["pixel"]
    hist_x.hist(pixel.abeta_signed, bins=24, color=FIGURE_COLORS["A"], alpha=.8)
    hist_y.hist(pixel.lamp1_signed, bins=24, color=FIGURE_COLORS["L"], alpha=.8, orientation="horizontal")
    hist_x.set_xlim(ax.get_xlim()); hist_y.set_ylim(ax.get_ylim())
    hist_x.axis("off"); hist_y.axis("off")
    legend = axis(63.5, 1.1, 59, 6); legend.axis("off")
    legend.legend(handles=[Line2D([], [], marker="o", ls="none", ms=2.5, color=c, label=label)
        for c, label in (("#C6C9CF", "Neither"), (FIGURE_COLORS["A"], "Aβ only"),
                          (FIGURE_COLORS["L"], "LAMP1 only"), (FIGURE_COLORS["both"], "Both"))]
        + [Line2D([], [], color=FIGURE_COLORS["ceiling"], ls="--", lw=.8, label="Ceiling")],
        loc="center", ncol=3, fontsize=5, handlelength=1.2, handletextpad=.3, columnspacing=.7)

    heading("i", "Overlap and coverage", 125, 43)
    ax = axis(153, 13, 28, 27)
    metrics = [("dice", "Dice"), ("jaccard", "Jaccard"),
        ("abeta_area_covered_by_lamp1_fraction", "Coverage Aβ→L"),
        ("lamp1_area_inside_abeta_fraction", "Coverage L→Aβ"),
        ("manders_A_in_L", "Manders Aβ→L"), ("manders_L_in_A", "Manders L→Aβ")]
    for y, (feature, name) in enumerate(metrics):
        row = lookup.loc[feature]
        value = row.value if np.isfinite(row.value) else row.diagnostic_value
        if np.isfinite(value):
            ax.plot([0, value], [y, y], color=FIGURE_COLORS["both"], lw=.8, alpha=.45)
            ax.plot(value, y, "o", ms=3, mec=FIGURE_COLORS["both"], mew=.8,
                    mfc=FIGURE_COLORS["both"] if np.isfinite(row.value) else "white")
        else:
            ax.text(.5, y, "NA", ha="center", va="center", fontsize=5)
    style_axis(ax); ax.set_yticks(range(6), [p[1] for p in metrics], fontsize=5.2)
    ax.set_ylim(5.6, -.6); ax.set_xlim(0, 1.04); ax.set_xticks([0, .5, 1])
    ax.set_xlabel("Coefficient / fraction", fontsize=5.7, labelpad=1.5)
    text(128, 2.5, "Filled: area; open: diagnostic", fontsize=5.2)

    if a4:
        figure_text(fig, 13.5, 287, "Fig. 3 | Boundary-referenced Aβ–LAMP1 phenotyping", fontsize=7, weight="bold", panel=None)
        caption = publication_caption_en(saturation, scale_bar_um)
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(family=FIGURE_FONT, size=6.5)
        y = 102.
        for paragraph in caption:
            words, line = paragraph.split(), ""
            for word in words:
                candidate = (line+" "+word).strip()
                width_mm = TextPath((0, 0), candidate, prop=prop).get_extents().width*25.4/72
                if width_mm > 183 and line:
                    figure_text(fig, 13.5, y, line, fontsize=6.5, panel=None); y -= 3.4
                    line = word
                else:
                    line = candidate
            if line:
                figure_text(fig, 13.5, y, line, fontsize=6.5, panel=None); y -= 4.3
        if y < 9:
            raise ValueError("A4 校样图注超出下边距；不得通过缩小字体掩盖。")
        del fig._panel_order  # 仅正文图导出一套子图；不重复导出 A4 中的相同面板。
    apply_nature_text_policy(fig)
    save_figure(fig, base, dpi)


def publication_caption_en(saturation, scale_bar_um):
    return [
        "Fig. 3 | Two-dimensional plaque geometry and local LAMP1 spatial phenotypes in one representative object.",
        "a, Native Aβ (blue) and LAMP1 (red) detector signals from the same CZI plane; linear display without background subtraction. "
        f"b,c, Separately ranked, background-corrected Aβ and LAMP1 intensity strata; rank 1 is brightest, and ranks are not biological subtypes or comparable concentrations across channels. Scale bars, {scale_bar_um:g} μm.",
        "d, Signed distance to the Aβ segmentation boundary (negative inside; positive outside), with the pixel-cell boundary, raster-mask convex hull and geometric principal axes. "
        "e, Ring-wise LAMP1-positive area fractions (top) and all observed pixel intensities (bottom). n denotes observed pixels per ring; circles and bars show the median and IQR, and open squares show the mean. Lines guide the eye; no continuous model is fitted.",
        "f, Dimensionless mask-geometry descriptors, each with its own definition. "
        "g, LAMP1-positive area fraction by distance band and angular sector; × marks incomplete cells or cells with fewer than three observed pixels. Angles increase clockwise from the image right. "
        "h, Paired pixel signals with marginal distributions; the gold dashed line marks the corrected Aβ ceiling. "
        "i, Binary overlap and directional coverage, together with QC-withheld intensity-weighted Manders values shown as open diagnostic points. L denotes LAMP1.",
        f"Aβ ceiling-affected mask pixels: {saturation[0]['saturated_fraction_mask']:.2%}. Intensity strata and joint-signal metrics remain diagnostic where QC fails. "
        "Pixels are not independent biological replicates; IQRs are not confidence intervals. This single-object description does not establish a core–halo mechanism or incremental predictive value beyond area.",
    ]

