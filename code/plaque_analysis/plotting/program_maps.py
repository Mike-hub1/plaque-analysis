"""Figure 7d: neuronal scores in the calibrated Figure 6 orientation."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from plaque_analysis.settings.microenvironment import PRIMARY, INK
from plaque_analysis.plotting.microenvironment_style import heading


def draw(container, tables):
    heading(container, "d", "Spatial neuronal programs")
    axes = container.subplots(2, 1, gridspec_kw={"hspace": .35})
    cmap = LinearSegmentedColormap.from_list("figure7_scores", ["#F4F0D6", "#87BBB0", "#225864"])
    background = tables["panel_d_locations"]
    for ax, code, label in zip(axes, PRIMARY, ("Neuronal system", "UPR")):
        q = tables["panel_d_scores"]
        q = q[q.program.eq(code)]
        unsupported = background.loc[~background.plaque_id.isin(q.plaque_id)]
        ax.scatter(unsupported.display_x_mm, unsupported.display_y_mm, s=7,
                   facecolors="none", edgecolors="#BFC8CD", lw=.4)
        if len(q):
            low, high = float(q.score.min()), float(q.score.max())
            if low == high:
                low, high = max(0, low - .01), min(1, high + .01)
            dots = ax.scatter(q.display_x_mm, q.display_y_mm, c=q.score, s=15,
                              cmap=cmap, vmin=low, vmax=high, edgecolors=INK, lw=.2)
            bar = container.colorbar(dots, ax=ax, fraction=.04, pad=.04)
            bar.set_label("UCell score")
            bar.outline.set_visible(False)
        else:
            ax.text(.5, .5, "No supported plaques", transform=ax.transAxes, ha="center", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        if len(background):
            for column, setter in (("display_x_mm", ax.set_xlim), ("display_y_mm", ax.set_ylim)):
                values = background[column].to_numpy(float)
                span = max(float(np.ptp(values)), .1)
                setter(values.min() - .05 * span, values.max() + .05 * span)
        if tables.get("display_settings", {}).get("invert_y", True):
            ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set(ylabel="y (mm)", title=label)
    axes[1].set_xlabel("x (mm)")
