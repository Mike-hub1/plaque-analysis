"""Figure 7b: detected-cell fraction and positive-cell mean expression."""

import numpy as np
from plaque_analysis.settings.microenvironment import CLASS_ORDER, COLORS, GENES
from plaque_analysis.plotting.microenvironment_style import clean, heading


def positive_axis(values):
    from matplotlib.ticker import MaxNLocator
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (0, 1), np.array([0, .5, 1])
    lower = max(0., np.floor((values.min() - .2) * 2) / 2)
    upper = max(lower + .5, float(values.max()) * 1.10)
    ticks = MaxNLocator(nbins=3).tick_values(lower, upper)
    return (lower, upper), ticks[(ticks >= lower) & (ticks <= upper)]


def draw(container, tables):
    heading(container, "b", "Gene detection and positive expression")
    axes = container.subplots(2, 2, gridspec_kw={"hspace": .6, "wspace": .5})
    for ax, (gene, cell) in zip(axes.flat, GENES):
        q = tables["panel_b_positive_points"]
        q = q[q.gene.eq(gene)]
        for group in CLASS_ORDER:
            z = q[q.S_label.eq(group)]
            ax.scatter(z.p_detect, z.mu_positive, s=9, color=COLORS[group], alpha=.58, edgecolors="none")
        limits, ticks = positive_axis(q.mu_positive)
        if q.empty:
            ax.text(.5, .5, "No detected cells", transform=ax.transAxes, ha="center", fontsize=7)
        ax.set_xlim(0, 1.03)
        ax.set_xticks([0, .5, 1])
        ax.set_ylim(*limits)
        ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks])
        ax.set_title(rf"$\mathit{{{gene}}}$ · {cell}", pad=3)
        clean(ax)
    axes[1, 0].set_xlabel("Detected-cell fraction")
    axes[1, 1].set_xlabel("Detected-cell fraction")
    axes[0, 0].set_ylabel("Positive mean\nlog1p(CP10k)")
    axes[1, 0].set_ylabel("Positive mean\nlog1p(CP10k)")
