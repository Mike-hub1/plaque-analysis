"""Figure 3: plot lamp1."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_data import empirical_curve
from plaque_analysis.plotting.fluorescence_helpers import draw_patch_map, ring_angle_heatmap, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, panel_heading, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, save_figure
from scipy import ndimage, stats
import numpy as np
import pandas as pd


def render_lamp1_plate(plt, data, masks, angular, spacing, config, base, dpi):
    fig = plt.figure(figsize=(183/25.4, 140/25.4))
    figure_text(fig, 3, 136, "Local LAMP1 phenotype | distribution, angular structure and signal patches", fontsize=9, weight="bold")
    figure_text(fig, 3, 130, "All measurements refer to the Aβ-defined window; image patches are not identified organelles or biological replicates.", fontsize=6.2)
    panel_heading(fig, "a", "Window-restricted signal patches", 3, 122)
    ax = figure_axis(fig, 3, 88, 48, 31)
    im = draw_patch_map(ax, data, masks, spacing)
    small_colorbar(fig, im, (10, 83, 31, 1.5), "log10 patch area (μm²)")
    panel_heading(fig, "b", "Radial × angular density", 62, 122)
    ax = figure_axis(fig, 81, 87, 32, 30)
    im = ring_angle_heatmap(ax, data)
    ax.set_xlabel("Angle (°)", fontsize=5.8, labelpad=2)
    ax.set_ylabel("Distance (μm)", fontsize=5.8, labelpad=1.5)
    ax.set_xticks([0, 180, 360])
    small_colorbar(fig, im, (84, 74, 26, 1.5), "LAMP1 area density", ticks=[0, .5, 1])
    panel_heading(fig, "c", "Ring-wise signal distributions", 122, 122)
    ax = figure_axis(fig, 139, 88, 37, 30)
    style_axis(ax)
    ring = data["rings"]
    colors = [FIGURE_COLORS["A"] if r.upper_um <= 0 else FIGURE_COLORS["L"] for r in ring.itertuples()]
    for y, (r, color) in enumerate(zip(ring.itertuples(), colors)):
        rows = data["pixel"][data["pixel"].ring_index == r.ring_index]
        values = rows.lamp1_signed.to_numpy()
        if len(values) >= 3 and np.ptp(values) > 0:
            grid = np.linspace(values.min(), values.max(), 160)
            density = stats.gaussian_kde(values, bw_method="scott")(grid)
            ax.fill_between(grid, y+.09, y+.09+density/max(density)*.3, color=color, alpha=.4, lw=.6)
        # Deterministic display jitter only; every observed pixel remains in the plot.
        jitter = -.12-.16*np.mod(np.arange(len(values))*.61803398875, 1)
        ax.scatter(values, y+jitter, s=2.5, color=color, alpha=.4, linewidths=0, rasterized=True)
        ax.plot([r.signal_q25, r.signal_q75], [y, y], color=FIGURE_COLORS["ink"], lw=1.5)
        ax.plot(r.signal_median, y, "o", ms=2.5, color=FIGURE_COLORS["ink"])
    ax.set_yticks(range(len(ring)), [f"{r.lower_um:.1f}:{r.upper_um:.1f}" for r in ring.itertuples()], fontsize=5.3)
    ax.set_ylim(-.5, len(ring)-.5)
    ax.set_xlabel("LAMP1 signal (a.u.)", fontsize=5.8, labelpad=2)
    ax.set_ylabel("Distance (μm)", fontsize=5.8, labelpad=1.5)
    figure_text(fig, 126, 77, "All pixels; median/IQR; half-KDE (Scott)", fontsize=5.6)
    panel_heading(fig, "d", "Exterior angular distribution", 3, 63)
    ax = figure_axis(fig, 10, 22, 39, 33, projection="polar")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(-1)
    angles = pd.DataFrame(angular)
    if len(angles):
        theta = np.radians(angles.centre_deg.to_numpy())
        observed = angles.density_observed.to_numpy()
        ax.bar(theta, observed, width=2*np.pi/config.angular_sectors*.88,
               color=FIGURE_COLORS["L"], edgecolor="white", lw=.4, alpha=.7)
        for t, value, status in zip(theta, observed, angles.status):
            ax.plot(t, value, "o", ms=3, color=FIGURE_COLORS["L"],
                    markerfacecolor=FIGURE_COLORS["L"] if status == "complete" else "white")
    ax.set_ylim(0, 1); ax.set_rticks([.5, 1]); ax.set_rlabel_position(45)
    ax.set_thetagrids([0, 90, 180, 270], fontsize=5.8)
    ax.tick_params(labelsize=5.5, pad=0)
    ax.grid(lw=.4, color="#CBCDD1"); ax.spines["polar"].set_linewidth(.5)
    figure_text(fig, 4, 12, "0° at right, clockwise; radius = area density", fontsize=5.5)
    panel_heading(fig, "e", "Patch-area empirical distribution", 62, 63)
    ax = figure_axis(fig, 75, 24, 39, 32)
    patch_areas = data["patches"].area_um2.to_numpy()
    empirical_curve(ax, patch_areas, FIGURE_COLORS["L"])
    if len(patch_areas):
        ax.scatter(np.sort(patch_areas), np.arange(1, len(patch_areas)+1)/len(patch_areas),
                   s=12, color=FIGURE_COLORS["L"], zorder=3)
    style_axis(ax)
    ax.set(xlabel="8-connected patch area (μm²)", ylabel="Cumulative patch fraction", ylim=(0, 1.05))
    ax.text(.03, .94, f"n={len(patch_areas)} image patches", transform=ax.transAxes, va="top", fontsize=5.8)
    panel_heading(fig, "f", "Area composition by boundary band", 122, 63)
    ax = figure_axis(fig, 141, 24, 35, 32)
    style_axis(ax)
    for y, r in enumerate(ring.itertuples()):
        positive = r.lamp1_positive_pixels*np.prod(spacing)
        total = r.observed_area_um2
        ax.barh(y, positive, color=FIGURE_COLORS["L"], height=.65)
        ax.barh(y, total-positive, left=positive, color=FIGURE_COLORS["neutral"], height=.65)
    ax.set_yticks(range(len(ring)), [f"{r.lower_um:.1f}:{r.upper_um:.1f}" for r in ring.itertuples()], fontsize=5.3)
    ax.set_xlabel("Observed ring area (μm²)", fontsize=5.8, labelpad=2)
    ax.set_ylabel("Distance (μm)", fontsize=5.8, labelpad=1.5)
    figure_text(fig, 130, 12, "Terracotta: LAMP1+; grey: other pixels", fontsize=5.5)
    figure_text(fig, 3, 2, "Observed portions only where incomplete; × denotes insufficient radial-sector support. Fixed-window truncation is not evidence of a complete halo.", fontsize=5.7, panel=None)
    save_figure(fig, base, dpi)

