"""Human ROI images and directly calculated optical, PA and LAMP1 summaries."""

import numpy as np
from plaque_analysis.settings.microenvironment import INK
from plaque_analysis.plotting.microenvironment_style import heading


def section_colors(human):
    import matplotlib.pyplot as plt
    return {section: plt.get_cmap("tab10")(index % 10)
            for index, section in enumerate(dict.fromkeys(human.section_id))}


def row_style(ax, human, show_ids=False):
    ax.set_ylim(human.row_y.max()+.8, -1.6)
    ax.set_yticks(human.row_y, human.roi_id if show_ids else [])
    ax.tick_params(axis="y", length=0, labelsize=5.8)
    ax.spines["left"].set_visible(False)
    for index, (section, rows) in enumerate(human.groupby("section_id", sort=False)):
        for y in rows.row_y.iloc[1::2]:
            ax.axhspan(y-.5, y+.5, color="#F1F4F6", lw=0, zorder=-1)
        if show_ids:
            ax.text(0, rows.row_y.min()-.8, section, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6, fontweight="bold")
        if index:
            ax.axhline(rows.row_y.min()-1.25, color="#BFC8CB", lw=.6)
    ax.grid(axis="x", color="#E1E6E9", lw=.3)


def draw_structure(container, tables):
    heading(container, "g", "Human plaque structure")
    axes = container.subplots(1, 3, gridspec_kw={"wspace": .35})
    data = tables["panel_g_human_structure"]
    colors = section_colors(data)
    specs = (("S_global", "Global S", None), ("m2_median", "M₂", "m2"),
             ("doa_median", "DOA", "doa"))
    for j, (ax, (field, name, key)) in enumerate(zip(axes, specs)):
        row_style(ax, data, j == 0)
        for row in data.to_dict("records"):
            color, x, y = colors[row["section_id"]], row[field], row["row_y"]
            if not np.isfinite(x):
                continue
            if key:
                ax.plot([row[f"{key}_q25"], row[f"{key}_q75"]], [y, y], color=color,
                        alpha=.3, lw=2.2, solid_capstyle="butt")
            else:
                ax.plot([0, x], [y, y], color=color, alpha=.55, lw=.8)
            ax.plot(x, y, "o", ms=3.2, mfc="white" if row["segmentation_boundary_limited"] else color,
                    mec=color if row["segmentation_boundary_limited"] else "white", mew=.6)
        if key:
            ax.margins(x=.15)
        else:
            ax.set_xlim(-.04, 1.04)
        ax.set_title(name)
    container.text(.5, .005, "Bars: within-plaque IQR; open points: mask touches field boundary",
                   ha="center", fontsize=5.5)


def draw_lamp1(container, tables):
    heading(container, "h", "Relative LAMP1 intensity")
    ax = container.subplots()
    human, data = tables["panel_g_human_structure"], tables["panel_h_lamp1"]
    colors = section_colors(human)
    row_style(ax, human, True)
    for _, group in data.groupby("roi_id", sort=False):
        rows = group.set_index("region")
        color, y, paired = colors[group.section_id.iloc[0]], group.row_y.iloc[0], bool(group.paired.iloc[0])
        if paired:
            ax.plot([rows.loc["inside", "relative_intensity"], rows.loc["outside_0_5um", "relative_intensity"]],
                    [y, y], color=color, lw=.75, alpha=.75)
        for region, row in rows.iterrows():
            if not row.available or not np.isfinite(row.relative_intensity):
                continue
            ax.plot(row.relative_intensity, y, "o" if region == "inside" else "D", ms=3.3,
                    mfc=color if paired else "white", mec="white" if paired else color, mew=.6)
    ax.axvline(1, color=INK, ls=(0, (2, 2)), lw=.6)
    ax.margins(x=.15)
    ax.set_xlabel("Relative LAMP1 intensity (fold)")
    support = tables["panel_h_pair_support"]
    note = "; ".join(f"{row.section_id}: {row.n_pairs}/{row.n_rois} pairs" for row in support.itertuples())
    ax.set_title(note, fontsize=6, wrap=True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=marker, color=INK, mfc=fill, ms=3, label=label)
               for marker, fill, label in (("o", INK, "Inside"), ("D", INK, "Outside 0–5 µm"),
                                          ("o", "white", "Outside unavailable"))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.12), ncol=1, frameon=False, fontsize=6)


def _physical_image(ax, image, spacing, **kwargs):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    h, w = image.shape[:2]
    extent = None if spacing is None else (0, w*spacing[0], h*spacing[1], 0)
    artist = ax.imshow(image, extent=extent, interpolation="nearest", **kwargs)
    if spacing is not None:
        width = w*spacing[0]
        length = 5. if width >= 7 else width/3
        bar = AnchoredSizeBar(ax.transData, length, f"{length:g} µm", loc="lower left",
                              pad=.3, color="white", frameon=False, size_vertical=width*.008)
        ax.add_artist(bar)
    ax.set_axis_off()
    return artist


def draw_images(container, tables):
    """Render the representative measured ROI, without reading manuscript artwork."""
    data = tables["panel_f_image"]
    heading(container, "f", f"Human ROI {data['roi_id']}")
    axes = container.subplots(2, 2)
    rgb = np.asarray(data["rgb"])
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(float) / np.iinfo(rgb.dtype).max
    elif rgb.max() > 1 or rgb.min() < 0:
        raise ValueError("Floating-point RGB display images must be in [0, 1]")
    _physical_image(axes[0, 0], rgb, data["spacing_xy"])
    axes[0, 0].set_title("Congo-red image")
    for ax, key, title, cmap, limits in ((axes[0, 1], "m2", "M₂", "viridis", None),
                                        (axes[1, 0], "phi", "Axis φ (°)", "hsv", (0, 180)),
                                        (axes[1, 1], "doa", "DOA", "viridis", None)):
        image = data[key]
        spacing = data["doa_spacing_xy"] if key == "doa" else data["spacing_xy"]
        options = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        artist = _physical_image(ax, image, spacing, cmap=cmap, **options)
        ax.set_title(title if spacing is not None else f"{title} · native PA grid")
        container.colorbar(artist, ax=ax, orientation="horizontal", fraction=.065, pad=.03,
                           ticks=[0, 90, 180] if key == "phi" else None)
