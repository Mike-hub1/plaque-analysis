"""Figure 7c: current continuous DAM/DAA reference gene-set scores."""

from plaque_analysis.settings.microenvironment import GLIAL_ORDER, COLORS, GLIAL_PROGRAMS, INK
from plaque_analysis.plotting.microenvironment_style import clean, heading
import numpy as np


def draw(container, tables):
    from matplotlib.ticker import MaxNLocator
    heading(container, "c", "Glial reference programs")
    axes = container.subplots(1, 2, gridspec_kw={"wspace": .6})
    data, summary = tables["panel_c_reference_program_scores"], tables["panel_c_summary"]
    for k, (ax, (code, cell, label, _)) in enumerate(zip(axes, GLIAL_PROGRAMS)):
        q = data[data.program.eq(code)]
        counts = []
        for j, group in enumerate(GLIAL_ORDER):
            z = q[q.S_label.eq(group)]
            counts.append(len(z))
            ax.scatter(z.x_drawn, z.score, s=7, color=COLORS[group], alpha=.5, edgecolors="none", zorder=3)
            selected = summary[summary.program.eq(code) & summary.S_label.eq(group)]
            if len(selected) and np.isfinite(selected[["q25", "q75", "median"]].to_numpy(float)).all():
                s = selected.iloc[0]
                ax.plot([j, j], [s.q25, s.q75], color=INK, lw=1.3, zorder=5)
                ax.plot([j-.11, j+.11], [s["median"], s["median"]], color=INK, lw=1.3, zorder=6)
        if len(q):
            span = max(.01, q.score.max() - q.score.min())
            ax.set_ylim(q.score.min() - .05 * span, q.score.max() + .05 * span)
        else:
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "No supported plaques", transform=ax.transAxes, ha="center", fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlim(-.48, 2.48)
        ax.set_xticks(range(3), [f"{s}\nn={n}" for s, n in zip(GLIAL_ORDER, counts)])
        ax.set_title(f"{cell}\n{label}")
        clean(ax)
    axes[0].set_ylabel("UCell score")
