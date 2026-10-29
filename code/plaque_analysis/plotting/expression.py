"""Expression heatmap with classwise trees and gene-program membership links."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import dendrogram


def _draw_tree(ax, tree, *, row=False, offset=0):
    if tree is None:
        ax.set_axis_off()
        return
    coordinates = dendrogram(tree, no_plot=True)
    for positions, heights in zip(coordinates["icoord"], coordinates["dcoord"]):
        position = (np.asarray(positions) - 5) / 10 + offset
        if row:
            ax.plot(heights, position, color="#53585C", lw=.6)
        else:
            ax.plot(position, heights, color="#53585C", lw=.6)
    ax.set_axis_off()


def _draw_membership(ax, genes, membership):
    """Use equally weighted links: memberships encode neither expression nor enrichment."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path

    gene_positions = {gene: position for position, gene in enumerate(genes)}
    memberships = membership.copy()
    memberships["gene_y"] = memberships.gene.map(gene_positions)
    if memberships.gene_y.isna().any() or memberships.duplicated(["gene", "program"]).any():
        raise ValueError("Membership links must reference unique displayed gene-program pairs")
    summary = memberships.groupby("program").gene_y.agg(["mean", "count"]).reset_index()
    summary = summary.sort_values(["mean", "program"])
    count = len(summary)
    if not count:
        raise ValueError("The heatmap requires at least one gene-program membership")
    # Program order follows the average linked gene position, reducing crossings.
    centers = np.linspace(.6, max(.6, len(genes) - 1.6), count)
    node_spacing = (len(genes) - 2.2) / max(count - 1, 1)
    width = min(.15, max(.025, node_spacing * .75 / summary["count"].max()))
    colour = "#6D91AC"
    for program, center in zip(summary.program, centers):
        members = memberships.loc[memberships.program.eq(program)].sort_values(["gene_y", "gene"])
        offsets = (np.arange(len(members)) - (len(members) - 1) / 2) * width
        for gene_y, offset in zip(members.gene_y, offsets):
            destination = center + offset
            half = width * .38
            vertices = [(1.1, gene_y - half), (1.75, gene_y - half),
                        (2.65, destination - half), (3.15, destination - half),
                        (3.15, destination + half), (2.65, destination + half),
                        (1.75, gene_y + half), (1.1, gene_y + half),
                        (1.1, gene_y - half)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(vertices, codes), facecolor=colour, edgecolor="none", alpha=.35))
        height = max(width * len(members), .15)
        ax.add_patch(Rectangle((3.15, center - height / 2), .08, height,
                               facecolor=colour, edgecolor="none"))
        ax.text(3.3, center, program.replace("_", " "), ha="left", va="center", fontsize=6)
    for gene, y in gene_positions.items():
        ax.text(.03, y, gene, va="center", ha="left", fontsize=7, fontstyle="italic")
    ax.set(xlim=(0, 6.4), ylim=(len(genes) - .5, -.5))
    ax.set_axis_off()
    ax.set_title("Gene-program membership", fontsize=9, pad=8)


def draw_f(tables, palette):
    """Draw the selected-gene Z heatmap; only display colours saturate at ±2.5."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    z, columns = tables["z"], tables["columns"]
    if list(z.columns) != columns.plaque_id.tolist():
        raise ValueError("Heatmap columns must follow the clustered plaque order")
    if not columns["class"].isin(palette).all():
        raise ValueError("Every heatmap class requires a palette colour")
    if not np.isfinite(z.to_numpy()).all():
        raise ValueError("Heatmap expression must contain finite Z scores")
    fig = plt.figure(figsize=(15, max(8, len(z) * .22 + 2)))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.2, .28, 10, .34],
                           width_ratios=[.9, 6.8, 5.8], hspace=.05, wspace=.025,
                           left=.02, right=.98, bottom=.065, top=.94)
    column_tree = fig.add_subplot(grid[0, 1])
    strip = fig.add_subplot(grid[1, 1])
    row_tree = fig.add_subplot(grid[2, 0])
    heat = fig.add_subplot(grid[2, 1])
    links = fig.add_subplot(grid[2, 2])
    colorbar = fig.add_subplot(grid[3, 1])
    _draw_tree(row_tree, tables["row_linkage"], row=True)
    if tables["row_linkage"] is not None:
        row_tree.set_xlim(tables["row_linkage"][:, 2].max() * 1.04, 0)
    row_tree.set_ylim(len(z) - .5, -.5)
    offset = 0
    boundaries = []
    for label in dict.fromkeys(columns["class"]):
        size = int(columns["class"].eq(label).sum())
        _draw_tree(column_tree, tables["column_linkages"][label], offset=offset)
        strip.text(offset + (size - 1) / 2, 0, f"{label} ({size})", ha="center", va="center", fontsize=7)
        offset += size
        boundaries.append(offset - .5)
    column_tree.set(xlim=(-.5, len(columns) - .5), ylim=(0, max(column_tree.get_ylim()[1], 1)))
    column_tree.set_axis_off()
    strip.imshow(np.array([to_rgba(palette[label]) for label in columns["class"]])[None],
                 aspect="auto", interpolation="nearest")
    strip.set(xlim=(-.5, len(columns) - .5), ylim=(.5, -.5))
    strip.set_axis_off()
    cmap = LinearSegmentedColormap.from_list("expression_z", ["#649FC1", "#FAFBFC", "#C76997"])
    image = heat.imshow(z.to_numpy(), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5,
                        interpolation="nearest")
    for boundary in boundaries[:-1]:
        heat.axvline(boundary, color="white", lw=1)
        strip.axvline(boundary, color="white", lw=1)
    heat.set(xticks=[], yticks=[])
    heat.spines[:].set_visible(False)
    _draw_membership(links, z.index.tolist(), tables["membership"])
    bar = fig.colorbar(image, cax=colorbar, orientation="horizontal", ticks=[-2.5, 0, 2.5], extend="both")
    bar.set_label("Gene row Z score", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(f"f  {len(z)} genes across {len(columns)} plaque neighbourhoods", fontsize=11, y=.975)
    return fig
