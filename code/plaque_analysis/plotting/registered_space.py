"""Spatial panels a/c/e use the same calibrated target-image coordinates."""

import numpy as np
from plaque_analysis.plotting.transcriptomics_style import CELL_COLORS, CLASS_COLORS, REGION_COLORS, display_coordinates, spatial_axes, add_scale_bar


def draw_a(data, settings):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    source = data["a"]
    for name, rows in source.groupby("broad_celltype", sort=False):
        xy = display_coordinates(rows, settings)
        ax.scatter(xy[:, 0], xy[:, 1], s=1, color=CELL_COLORS.get(name, "#C8C8C8"),
                   alpha=.9, edgecolors="none", label=f"{name} ({len(rows):,})", rasterized=True)
    spatial_axes(ax, settings)
    add_scale_bar(ax)
    ax.set_title("a  RNA-qualified nuclei")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, markerscale=3)
    return fig


def draw_c(data, settings):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    source = data["c"]
    valid = source.valid_for_spatial_sampling
    fig, ax = plt.subplots(figsize=(7, 5.8), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("registration", ["#C8DFD6", "#3C9395", "#2D4C6B"])
    xy = display_coordinates(source.loc[valid], settings)
    values = source.loc[valid, "registration_error_um"]
    vmax = float(settings.get("registration_color_max_um", 6))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("registration_color_max_um must be positive")
    points = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=1, cmap=cmap,
                        vmin=0, vmax=vmax, edgecolors="none", rasterized=True)
    excluded = source.loc[~valid]
    xy = display_coordinates(excluded, settings)
    finite = np.isfinite(xy).all(axis=1)
    ax.scatter(xy[finite, 0], xy[finite, 1], color="#CE7E33", marker="x", s=6,
               linewidths=.5, label=f"Unavailable ({len(excluded):,})", rasterized=True)
    if (~finite).any():
        ax.text(.02, .98, f"{int((~finite).sum())} unavailable coordinates not drawable",
                transform=ax.transAxes, va="top", fontsize=7)
    spatial_axes(ax, settings)
    add_scale_bar(ax)
    ax.set_title("c  Landmark leave-one-out registration error")
    ax.legend(fontsize=7, frameon=False)
    fig.colorbar(points, ax=ax, label="Interpolated LOO error (µm)",
                 extend="max" if len(values) and values.max() > vmax else "neither")
    return fig


def draw_e(data, settings):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    for name, rows in data["e_cells"].groupby("inferred_region_code", sort=False):
        xy = display_coordinates(rows, settings)
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c=REGION_COLORS.get(name, "#E6E6E6"),
                   alpha=.65, edgecolors="none", label=name, rasterized=True)
    for name, rows in data["e_plaques"].groupby("class", sort=True):
        xy = display_coordinates(rows, settings)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c=CLASS_COLORS[name],
                   edgecolors="white", linewidths=.4, label=f"{name} ({len(rows)})")
    spatial_axes(ax, settings)
    add_scale_bar(ax)
    ax.set_title("e  Fixed imaging classes and neuron-inferred regions")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, markerscale=2)
    return fig
