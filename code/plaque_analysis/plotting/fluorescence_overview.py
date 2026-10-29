"""Figure 3: plot main."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_helpers import add_scale_bar, draw_channel_scatter, draw_geometry_boundary, draw_patch_map, draw_qc_summary, mask_contour, ring_angle_heatmap, show_native, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, panel_heading, plot_value, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, save_figure
import numpy as np


def render_main_plate(plt, data, raw, signed, masks, geometry, saturation, spacing,
                      config, base, dpi):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    fig = plt.figure(figsize=(183/25.4, 170/25.4))
    figure_text(fig, 3, 166, "Boundary-referenced Aβ–LAMP1 phenotyping", fontsize=9, weight="bold")
    figure_text(fig, 3, 160, "Native detector data  →  local spatial organization  →  measurement eligibility", fontsize=6.5)
    for ci, x in enumerate((3, 48)):
        panel_heading(fig, "ab"[ci], ("Aβ detector signal", "LAMP1 detector signal")[ci], x, 152)
        ax = figure_axis(fig, x, 126, 39, 25)
        im = show_native(ax, raw[ci], spacing, vmin=0, vmax=float(raw[ci].max()) or 1)
        if saturation[ci]["mask"].any():
            mask_contour(ax, saturation[ci]["mask"], spacing, FIGURE_COLORS["ceiling"], .65)
        add_scale_bar(ax, raw[ci].shape, spacing)
        small_colorbar(fig, im, (x+4, 122, 28, 1.5), "Detector signal (a.u.)")
    panel_heading(fig, "c", "Signal masks and window", 93, 152)
    ax = figure_axis(fig, 93, 126, 39, 25)
    classes = masks[0].astype(int)+2*masks[1].astype(int)
    cmap = ListedColormap(["#F0F0F2", FIGURE_COLORS["A"], FIGURE_COLORS["L"], FIGURE_COLORS["both"]])
    show_native(ax, classes, spacing, cmap=cmap, vmin=0, vmax=3)
    mask_contour(ax, geometry["expected"], spacing, FIGURE_COLORS["ink"], .65, linestyles="--")
    figure_text(fig, 94, 120, "Aβ", color=FIGURE_COLORS["A"], fontsize=6)
    figure_text(fig, 104, 120, "LAMP1", color=FIGURE_COLORS["L"], fontsize=6)
    figure_text(fig, 121, 120, "Both", color=FIGURE_COLORS["both"], fontsize=6)
    figure_text(fig, 94, 114, "Dashed outline: fixed window", fontsize=5.6)
    panel_heading(fig, "d", "Geometry and boundary", 138, 152)
    ax = figure_axis(fig, 138, 126, 39, 25)
    im = draw_geometry_boundary(ax, masks[0], geometry, data["window"], spacing, config.window_radius_um)
    small_colorbar(fig, im, (142, 122, 28, 1.5), "Signed distance (μm)")
    panel_heading(fig, "e", "Boundary-resolved LAMP1 distributions", 3, 106)
    ring = data["rings"]
    ax_density = figure_axis(fig, 16, 92, 76, 10)
    ax_signal = figure_axis(fig, 16, 61, 76, 26)
    pixel = data["pixel"]
    for ax in (ax_density, ax_signal):
        style_axis(ax)
        ax.axvspan(float(ring.lower_um.min()), 0, color=FIGURE_COLORS["A"], alpha=.08, lw=0)
        ax.axvline(0, color=FIGURE_COLORS["ink"], lw=.6, ls=":")
        ax.set_xlim(float(ring.lower_um.min()), float(ring.upper_um.max()))
    ax_density.plot(ring.centre_um, ring.area_density_observed, "--", color="#A4A4AA", lw=.8)
    ax_density.plot(ring.centre_um, ring.area_density_complete, "o-", color=FIGURE_COLORS["L"], ms=3, lw=1)
    ax_density.set_ylim(0, 1.32)
    ax_density.set_yticks([0, .5, 1])
    ax_density.set_ylabel("Area density", fontsize=5.7, labelpad=2)
    ax_density.tick_params(axis="x", bottom=False, labelbottom=False)
    for r in ring.itertuples():
        ax_density.text(r.centre_um, 1.15, f"n={r.observed_pixels}", ha="center", fontsize=5)
    for positive, color in ((False, "#BFC1C6"), (True, FIGURE_COLORS["L"])):
        rows = pixel[pixel.lamp1_positive == positive]
        ax_signal.scatter(rows.distance_um, rows.lamp1_signed, s=3, color=color, alpha=.32,
                          linewidths=0, rasterized=True)
    ax_signal.errorbar(ring.centre_um, ring.signal_median,
        yerr=[ring.signal_median-ring.signal_q25, ring.signal_q75-ring.signal_median],
        fmt="o", color=FIGURE_COLORS["ink"], ms=3.2, lw=1.4, capsize=2, zorder=5)
    ax_signal.plot(ring.centre_um, ring.mean_signed_signal_observed, "s--", ms=2.6, lw=.75,
                   color=FIGURE_COLORS["both"], markerfacecolor="white", zorder=6)
    ax_signal.set_xlabel("Distance to Aβ boundary (μm)", labelpad=2)
    ax_signal.set_ylabel("LAMP1 signal (a.u.)", labelpad=2)
    figure_text(fig, 16, 49, "All pixels; circles/bars: median/IQR; open squares: mean", fontsize=5.7)
    panel_heading(fig, "f", "Radial × angular organization", 102, 106)
    ax = figure_axis(fig, 119, 66, 58, 35)
    im = ring_angle_heatmap(ax, data, exterior=True, annotate=True)
    small_colorbar(fig, im, (132, 53, 32, 1.5), "LAMP1-positive area density", ticks=[0, .5, 1])
    figure_text(fig, 105, 43, "Exterior only; ×: incomplete or <3 observed pixels", fontsize=5.5)
    panel_heading(fig, "g", "LAMP1 signal patches", 3, 38)
    ax = figure_axis(fig, 3, 11, 48, 26)
    draw_patch_map(ax, data, masks, spacing)
    largest = float(data["patches"].area_um2.max()) if len(data["patches"]) else np.nan
    figure_text(fig, 4, 8, f"{len(data['patches'])} patches; largest: {plot_value(largest)} μm²", fontsize=5.5)
    diagnostic = data["master"].query("role == 'Z_AL' and status == 'not_estimable'").shape[0] > 0
    panel_heading(fig, "h", "Signal relationships (diagnostic)" if diagnostic else "Two-channel signal", 62, 38)
    ax = figure_axis(fig, 75, 13, 41, 23)
    draw_channel_scatter(ax, data, saturation[0], raw[0].astype(float)-signed[0])
    ax.set_xlabel("Aβ signal (a.u.)", fontsize=5.7, labelpad=1.5)
    ax.set_ylabel("LAMP1 (a.u.)", fontsize=5.7, labelpad=1.5)
    panel_heading(fig, "i", "Measurability", 127, 38)
    ax = figure_axis(fig, 143, 13, 32, 23)
    draw_qc_summary(ax, data["master"])
    ax.set_xlabel("Computed / total fields", fontsize=5.7, labelpad=1.5)
    figure_text(fig, 3, 2, "One plaque; pixels are not biological replicates. Gold: ceiling/QC-withheld. Fixed window does not establish a complete halo.", fontsize=5.7, panel=None)
    save_figure(fig, base, dpi)

