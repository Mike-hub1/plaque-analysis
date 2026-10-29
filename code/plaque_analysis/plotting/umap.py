"""Shared global UMAP with class, confidence and sample annotation rings."""

import numpy as np
from plaque_analysis.plotting.transcriptomics_style import CELL_COLORS, ABBREVIATIONS, CONFIDENCE_COLORS


def ring_segments(ax, start, end, fractions, colours, radius, width):
    from matplotlib.patches import Wedge
    cursor = start
    for label, fraction in fractions.items():
        stop = cursor - (start - end) * fraction
        if fraction > 0:
            ax.add_patch(Wedge((0, 0), radius, stop, cursor, width=width,
                               facecolor=colours[label], edgecolor="white", linewidth=.2))
        cursor = stop


def draw_b(data):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    source = data["b"]
    xy = source[["UMAP1", "UMAP2"]].to_numpy(float)
    if not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Panel b requires finite UMAP coordinates for RNA-qualified cells")
    # One translation and uniform scaling preserve the global embedding geometry.
    xy = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2
    radius = np.linalg.norm(xy, axis=1).max()
    if radius > 0:
        xy = xy * (.88 / radius)
    fig, ax = plt.subplots(figsize=(8, 9), layout="constrained")
    colours = source.broad_celltype.map(CELL_COLORS).fillna("#C8C8C8")
    ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=1, edgecolors="none", rasterized=True)
    classes = [name for name in CELL_COLORS if source.broad_celltype.eq(name).any()]
    classes += sorted(set(source.broad_celltype) - set(classes))
    samples = list(dict.fromkeys(source.sample_id.astype(str)))
    sample_colours = {name: plt.get_cmap("Set2")(k % 8) for k, name in enumerate(samples)}
    for index, name in enumerate(classes):
        # Equal-width sectors annotate within-class proportions, not cell counts.
        start = 90 - index * 360 / len(classes)
        end = start - 360 / len(classes) + 6
        rows = source.loc[source.broad_celltype.eq(name)]
        ring_segments(ax, start, end, {name: 1}, {name: CELL_COLORS.get(name, "#C8C8C8")}, 1.16, .025)
        confidence = rows.confidence.fillna("low").str.lower().value_counts(normalize=True)
        ring_segments(ax, start, end, confidence.to_dict(), CONFIDENCE_COLORS, 1.105, .025)
        group = rows.sample_id.astype(str).value_counts(normalize=True)
        ring_segments(ax, start, end, group.to_dict(), sample_colours, 1.05, .025)
        angle = np.deg2rad((start + end) / 2)
        ax.text(1.30*np.cos(angle), 1.30*np.sin(angle), ABBREVIATIONS.get(name, name),
                ha="center", va="center", fontsize=8)
        for fraction in (0, .5, 1):
            theta = np.deg2rad(start - (start-end)*fraction)
            ax.text(1.205*np.cos(theta), 1.205*np.sin(theta), f"{fraction*100:g}",
                    ha="center", va="center", fontsize=6)
    handles = [Patch(color=CELL_COLORS.get(name, "#C8C8C8"), label=name) for name in classes]
    handles += [Patch(color=color, label=f"Confidence: {name}") for name, color in CONFIDENCE_COLORS.items()]
    handles += [Patch(color=color, label=f"Sample: {name}") for name, color in sample_colours.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3, fontsize=7, frameon=False)
    ax.set(xlim=(-1.42, 1.42), ylim=(-1.42, 1.42), aspect="equal",
           title="b  Global UMAP · outer to inner: class, confidence, sample")
    ax.set_axis_off()
    return fig
