"""Figure 3: plot abeta."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_data import empirical_curve
from plaque_analysis.plotting.fluorescence_helpers import draw_geometry_boundary, mask_contour, show_native, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, panel_heading, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, save_figure
from scipy import ndimage, stats
import numpy as np


def render_abeta_plate(plt, data, signed, masks, geometry, saturation, spacing, config, base, dpi):
    from matplotlib.colors import LinearSegmentedColormap
    fig = plt.figure(figsize=(183/25.4, 140/25.4))
    figure_text(fig, 3, 136, "Aβ-only organization | candidate structural descriptors", fontsize=9, weight="bold")
    fraction = saturation[0]["saturated_fraction_mask"]
    figure_text(fig, 3, 130, f"Native 2-D sampling; {fraction:.1%} of Aβ-mask pixels reach the intensity ceiling. Intensity panels are diagnostic.", fontsize=6.2)
    panel_heading(fig, "a", "Geometry and boundary distance", 3, 122)
    ax = figure_axis(fig, 3, 83, 48, 36)
    im = draw_geometry_boundary(ax, masks[0], geometry, data["window"], spacing, config.window_radius_um)
    small_colorbar(fig, im, (10, 81, 32, 1.5), "Signed distance (μm)")
    figure_text(fig, 4, 73, "Black: boundary; gold: hull; dashed: principal axes", fontsize=5.3)
    sy, sx = spacing
    panel_heading(fig, "b", "Interior depth and mask holes", 62, 122)
    ax = figure_axis(fig, 63, 85, 48, 32)
    depth = np.ma.masked_where(~masks[0], -geometry["distance"])
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#F0F0F2")
    im = show_native(ax, depth, spacing, cmap=cmap, vmin=0)
    filled = ndimage.binary_fill_holes(np.pad(masks[0], 1))[1:-1, 1:-1]
    mask_contour(ax, filled & ~masks[0], spacing, FIGURE_COLORS["ceiling"], .7)
    small_colorbar(fig, im, (70, 81, 31, 1.5), "Inward depth (μm)")
    panel_heading(fig, "c", "Aβ intensity distribution", 122, 122)
    ax = figure_axis(fig, 136, 86, 40, 31)
    values = signed[0][masks[0]]
    empirical_curve(ax, values, FIGURE_COLORS["A"])
    if saturation[0]["mask"].any():
        ceiling_values = signed[0][saturation[0]["mask"] & masks[0]]
        if len(ceiling_values) and np.ptp(ceiling_values) == 0:
            ax.axvline(ceiling_values[0], color=FIGURE_COLORS["ceiling"], lw=.8, ls="--")
    ax.set(xlabel="Corrected signal (a.u.)", ylabel="Pixel cumulative fraction", ylim=(0, 1.03))
    style_axis(ax)
    ax.text(.04, .92, f"n={len(values)} pixels\nDiagnostic only", transform=ax.transAxes, fontsize=5.8, va="top")
    panel_heading(fig, "d", "Internal signal gradients", 3, 66)
    ax = figure_axis(fig, 3, 26, 49, 35)
    interior = ndimage.binary_erosion(masks[0], structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    y, x = np.where(interior)
    gradient = np.full(masks[0].shape, np.nan)
    gradient[y, x] = np.hypot((signed[0][y, x+1]-signed[0][y, x-1])/(2*sx),
                              (signed[0][y+1, x]-signed[0][y-1, x])/(2*sy))
    cmap = LinearSegmentedColormap.from_list("internal_gradient", ["#F5EFF7", "#AF95BF", "#604075"])
    cmap.set_bad("#EEEEF0")
    im = show_native(ax, np.ma.masked_invalid(gradient), spacing, cmap=cmap, vmin=0)
    mask_contour(ax, masks[0], spacing, FIGURE_COLORS["A"], .5)
    small_colorbar(fig, im, (10, 22, 32, 1.5), "Gradient magnitude (a.u./μm)")
    figure_text(fig, 4, 10, f"{len(y)} valid stencils; diagnostic only", fontsize=5.8)
    panel_heading(fig, "e", "Multiscale texture (diagnostic)", 61, 66)
    directions = data["directions"]
    # All curves are Aβ-only; blue tones and marker shapes encode direction, not channel identity.
    direction_colors = ["#234D70", "#487A9C", "#739BAF", "#A0B7C8"]
    direction_markers = ["o", "s", "^", "D"]
    for j, (key, label) in enumerate((("contrast", "Contrast"), ("homogeneity", "Homogeneity"), ("entropy", "Entropy (bit)"))):
        ax = figure_axis(fig, 74, 46-j*14, 37, 10)
        style_axis(ax)
        for (dx, dy), color, marker in zip(((1, 0), (0, 1), (1, 1), (-1, 1)), direction_colors, direction_markers):
            rows = directions[(directions.dx == dx*directions.native_pixel_step) &
                              (directions.dy == dy*directions.native_pixel_step)].sort_values("distance_um")
            ax.plot(rows.distance_um, rows[key], marker=marker, ms=2.5, lw=.75, color=color)
        ax.set_ylabel(label, fontsize=5.5, labelpad=2)
        if j < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Actual pixel-pair distance (μm)", fontsize=5.6, labelpad=2)
    from matplotlib.lines import Line2D
    key_ax = figure_axis(fig, 62, 6, 54, 4)
    key_ax.axis("off")
    key_ax.legend(handles=[Line2D([], [], marker=m, color=c, lw=.6, markersize=2.6, label=label)
        for m, c, label in zip(direction_markers, direction_colors, ("0°", "90°", "45°", "135°"))],
        loc="center", ncol=4, fontsize=5.6, handlelength=1, handletextpad=.3, columnspacing=.7)
    panel_heading(fig, "f", "Dimensionless geometry", 122, 66)
    ax = figure_axis(fig, 146, 23, 30, 34)
    style_axis(ax)
    keys = ["solidity", "bounding_box_extent", "circularity", "eroded_area_fraction", "mask_hole_area_fraction"]
    names = ["Solidity", "Box extent", "Circularity", "Eroded fraction", "Hole fraction"]
    lookup = data["master"].set_index("feature")
    for y, key in enumerate(keys):
        value = lookup.loc[key, "value"]
        if np.isfinite(value):
            ax.plot([0, value], [y, y], color=FIGURE_COLORS["A"], alpha=.4, lw=2)
            ax.plot(value, y, "o", color=FIGURE_COLORS["A"], ms=3.5)
            ax.text(min(value+.035, .97), y-.16, f"{value:.3f}", fontsize=5.5, ha="left" if value < .9 else "right")
        else:
            ax.text(.5, y, "NA", fontsize=5.5, ha="center")
    ax.set_yticks(range(len(keys)), names, fontsize=5.5)
    ax.set_xlim(0, 1); ax.set_ylim(4.65, -.65); ax.set_xlabel("Ratio", labelpad=2)
    figure_text(fig, 3, 2, "Mask holes are segmentation topology, not molecular pores. Ceiling-affected intensity and texture values remain excluded from accepted features.", fontsize=5.7, panel=None)
    save_figure(fig, base, dpi)

