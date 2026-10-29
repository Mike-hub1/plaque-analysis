"""Figure 3: plot helpers."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_layout import figure_axis, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS
from skimage.morphology import convex_hull_image
import numpy as np


def show_native(ax, image, spacing, *, cmap="gray", vmin=None, vmax=None):
    sy, sx = spacing
    h, w = image.shape
    extent = (-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent)
    ax.axis("off")
    return im


def mask_contour(ax, mask, spacing, color="white", linewidth=.6, **kwargs):
    # Explicit centre coordinates match all exported geometric measurements.
    if mask.any() and not mask.all():
        ax.contour(np.arange(mask.shape[1])*spacing[1], np.arange(mask.shape[0])*spacing[0],
                   mask.astype(float), levels=[.5], colors=[color], linewidths=linewidth, **kwargs)


def small_colorbar(fig, mappable, box, label, *, ticks=None, orientation="horizontal"):
    ax = figure_axis(fig, *box)
    cb = fig.colorbar(mappable, cax=ax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(.35)
    cb.ax.tick_params(labelsize=5.3, length=2, width=.35, pad=1.2)
    cb.set_label(label, size=5.7, labelpad=1.5)
    return cb


def ring_angle_heatmap(ax, data, *, exterior=False, annotate=False):
    from matplotlib.colors import LinearSegmentedColormap
    cells = data["cells"]
    if exterior:
        cells = cells[cells.lower_um >= 0]
    ring_ids = sorted(cells.ring_index.unique())
    sectors = sorted(cells.sector.unique())
    matrix = cells.pivot(index="ring_index", columns="sector", values="density_complete").reindex(index=ring_ids, columns=sectors)
    cmap = LinearSegmentedColormap.from_list("lamp_density", ["#FCF5EF", "#DCA287", "#AE4E37"])
    cmap.set_bad("#DADCDF")
    im = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   extent=(0, 360, 0, len(ring_ids)), interpolation="nearest")
    labels = []
    for y, ri in enumerate(ring_ids):
        r = cells[cells.ring_index == ri].iloc[0]
        labels.append(f"{r.lower_um:.1f} to {r.upper_um:.1f}")
        for s in sectors:
            value = matrix.loc[ri, s]
            if not np.isfinite(value):
                ax.text((s+.5)*360/len(sectors), y+.5, "×", ha="center", va="center", fontsize=5.5)
            elif annotate:
                ax.text((s+.5)*360/len(sectors), y+.5, f"{value:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if value > .65 else FIGURE_COLORS["ink"])
    ax.set_yticks(np.arange(len(ring_ids))+.5, labels, fontsize=5.5)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Angle (°; clockwise from right)", labelpad=2)
    ax.set_ylabel("Boundary distance (μm)", labelpad=2)
    return im


def draw_channel_scatter(ax, data, saturation_a, background_a):
    colors = ["#C6C9CF", FIGURE_COLORS["A"], FIGURE_COLORS["L"], FIGURE_COLORS["both"]]
    pixel = data["pixel"]
    for group in range(4):
        selected = pixel.binary_class == group
        ax.scatter(pixel.loc[selected, "abeta_signed"], pixel.loc[selected, "lamp1_signed"],
                   s=4, c=colors[group], alpha=.45, edgecolors="none", rasterized=True)
    if saturation_a.get("ceiling") is not None:
        # Constant background only; the ceiling is not a single corrected value under rolling-ball.
        if np.ptp(background_a) == 0:
            ceiling = saturation_a["ceiling"]-float(np.ravel(background_a)[0])
            if pixel.abeta_signed.max() >= ceiling:
                ax.axvline(ceiling, color=FIGURE_COLORS["ceiling"], lw=.7, ls="--")
    ax.set_xlabel("Aβ corrected signal (a.u.)", labelpad=2)
    ax.set_ylabel("LAMP1 corrected signal (a.u.)", labelpad=2)
    style_axis(ax)


def draw_patch_map(ax, data, masks, spacing, *, label_patches=True):
    from matplotlib.colors import LinearSegmentedColormap
    sizes = data["patches"]
    display = np.full(data["patch_labels"].shape, np.nan)
    for row in sizes.itertuples():
        display[data["patch_labels"] == row.patch_id] = np.log10(row.area_um2)
    cmap = LinearSegmentedColormap.from_list("lamp_patches", ["#F0CBBB", "#C77255", "#7C3229"])
    cmap.set_bad("#F2F2F3")
    finite = display[np.isfinite(display)]
    im = show_native(ax, np.ma.masked_invalid(display), spacing, cmap=cmap,
                     vmin=float(finite.min()) if len(finite) else 0,
                     vmax=float(finite.max()+1e-9) if len(finite) else 1)
    mask_contour(ax, masks[0], spacing, FIGURE_COLORS["A"], .55)
    mask_contour(ax, data["window"], spacing, FIGURE_COLORS["ink"], .45, linestyles="--")
    if label_patches:
        for row in sizes.nlargest(4, "area_um2").itertuples():
            ax.text(row.centroid_x_um, row.centroid_y_um, str(row.patch_id), ha="center", va="center",
                    color="white", fontsize=6, weight="bold",
                    bbox={"boxstyle": "circle,pad=.15", "fc": FIGURE_COLORS["ink"], "ec": "none", "alpha": .8})
    return im


def draw_qc_summary(ax, master):
    style_axis(ax)
    groups = [
        ("Aβ shape", (master.role == "X_A") & master.family.isin(["shape", "shape_topology"])),
        ("Aβ signal", (master.role == "X_A") & ~master.family.isin(["shape", "shape_topology", "texture", "multiscale_texture"])),
        ("Aβ texture", (master.role == "X_A") & master.family.isin(["texture", "multiscale_texture"])),
        ("LAMP1", (master.role == "Y_L") & (master.measurement_kind == "phenotype")),
        ("Joint", master.role == "Z_AL"),
        ("Window QC", master.measurement_kind != "phenotype"),
    ]
    for y, (label, selection) in enumerate(groups):
        subset = master[selection]
        n, ok = len(subset), int(subset.value.notna().sum())
        computed_color = FIGURE_COLORS["neutral"] if label == "Window QC" else FIGURE_COLORS["ink"]
        ax.barh(y, ok, color=computed_color, height=.62)
        ax.barh(y, n-ok, left=ok, color=FIGURE_COLORS["withheld"], height=.62)
        ax.text(n+1, y, f"{ok}/{n}", fontsize=5.5, va="center")
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=5.8)
    ax.set_xlim(0, max(int(sel.sum()) for _, sel in groups)+10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of fields", labelpad=2)


def add_scale_bar(ax, shape, spacing):
    sy, sx = spacing
    width, height = shape[1]*sx, shape[0]*sy
    candidates = [1, 2, 5, 10, 20, 50, 100]
    length = max([v for v in candidates if v <= width/4], default=width/5)
    x, y = width*0.06, height*0.91
    ax.plot([x, x+length], [y, y], lw=2, color="white")
    ax.text(x+length/2, y-height*0.055, f"{length:g} μm", color="white", ha="center", fontsize=6)
    return length


def pixel_cell_outline(ax, mask, spacing, *, color, linewidth=.55, linestyle="solid", zorder=3):
    """以原生像素单元的外露边绘制掩膜；不对轮廓平滑或插值。"""
    from matplotlib.collections import LineCollection
    mask = np.asarray(mask, bool)
    sy, sx = spacing
    padded = np.pad(mask, 1)
    segments = []
    neighbors = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
    offsets = (((-.5, -.5), (.5, -.5)), ((-.5, .5), (.5, .5)),
               ((-.5, -.5), (-.5, .5)), ((.5, -.5), (.5, .5)))
    for neighbor, ends in zip(neighbors, offsets):
        for y, x in np.argwhere(mask & ~neighbor):
            segments.append([((x+dx)*sx, (y+dy)*sy) for dx, dy in ends])
    lines = LineCollection(segments, colors=[color], linewidths=linewidth,
                           linestyles=linestyle, zorder=zorder)
    ax.add_collection(lines)
    return lines


def draw_geometry_boundary(ax, mask_a, geometry, window, spacing, window_radius_um):
    """一幅图共用掩膜、物理坐标和距离场；显示几何不改变既有特征定义。"""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("geometry_boundary", ["#3F7C9A", "#F8F5F0", "#C8795B"])
    cmap.set_bad("#EEEFF0")
    distance = np.asarray(geometry["distance"])
    limit = max(window_radius_um, float(np.max(-distance[mask_a])) if mask_a.any() else 0.)
    im = show_native(ax, np.ma.masked_where(~window, distance), spacing,
                     cmap=cmap, vmin=-limit, vmax=limit)
    # 凸包来自同一栅格掩膜，与原有 solidity 定义保持一致。
    hull = convex_hull_image(mask_a) if mask_a.any() else mask_a
    pixel_cell_outline(ax, hull, spacing, color=FIGURE_COLORS["ceiling"], linewidth=.55, zorder=3)
    pixel_cell_outline(ax, mask_a, spacing, color="#25272C", linewidth=.65, zorder=4)
    yy, xx = np.where(mask_a)
    if len(xx) > 1:
        sy, sx = spacing
        xy = np.column_stack((xx*sx, yy*sy))
        center = xy.mean(axis=0)
        covariance = np.cov(xy, rowvar=False, bias=True)+np.diag([sx*sx/12, sy*sy/12])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        for j in (0, 1):
            ends = center[:, None]+eigenvectors[:, j:j+1]*np.array([[-2, 2]])*np.sqrt(eigenvalues[j])
            ax.plot(ends[0], ends[1], color="white", lw=1.25, zorder=5)
            ax.plot(ends[0], ends[1], color="#25272C", lw=.55, ls="--", zorder=6)
        ax.plot(*center, "o", ms=2.4, mfc="white", mec="#25272C", mew=.5, zorder=7)
    return im


def draw_boundary_distributions(ax_density, ax_signal, data, *, marker_size=3.2):
    """共用径向图实现；完整保留面积密度、全部像素、均值、中位数和 IQR。"""
    ring, pixel = data["rings"], data["pixel"]
    for ax in (ax_density, ax_signal):
        style_axis(ax)
        ax.axvspan(float(ring.lower_um.min()), 0, color=FIGURE_COLORS["A"], alpha=.08, lw=0)
        ax.axvline(0, color=FIGURE_COLORS["ink"], lw=.55, ls=":")
        ax.set_xlim(float(ring.lower_um.min()), float(ring.upper_um.max()))
    ax_density.plot(ring.centre_um, ring.area_density_observed, "--", color="#A4A4AA", lw=.65)
    ax_density.plot(ring.centre_um, ring.area_density_complete, "o-", color=FIGURE_COLORS["L"], ms=marker_size, lw=.85)
    ax_density.set_ylim(0, 1.35)
    ax_density.set_yticks([0, .5, 1])
    ax_density.set_ylabel("Positive area\nfraction", fontsize=5.5, labelpad=2)
    ax_density.tick_params(axis="x", bottom=False, labelbottom=False)
    for r in ring.itertuples():
        ax_density.text(r.centre_um, 1.18, f"n={r.observed_pixels}", ha="center", fontsize=5)
    for positive, color in ((False, "#BFC1C6"), (True, FIGURE_COLORS["L"])):
        rows = pixel[pixel.lamp1_positive == positive]
        ax_signal.scatter(rows.distance_um, rows.lamp1_signed, s=2.5, color=color, alpha=.30,
                          linewidths=0, rasterized=False)
    ax_signal.errorbar(ring.centre_um, ring.signal_median,
        yerr=[ring.signal_median-ring.signal_q25, ring.signal_q75-ring.signal_median],
        fmt="o", color=FIGURE_COLORS["ink"], ms=marker_size, lw=1.15, capsize=1.8, zorder=5)
    ax_signal.plot(ring.centre_um, ring.mean_signed_signal_observed, "s--", ms=marker_size-.3, lw=.7,
                   color=FIGURE_COLORS["both"], markerfacecolor="white", zorder=6)
    ax_signal.set_xlabel("Distance to Aβ boundary (μm)", fontsize=6, labelpad=2)
    ax_signal.set_ylabel("LAMP1 signal (a.u.)", fontsize=6, labelpad=2)
    return ax_density, ax_signal

