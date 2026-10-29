"""Figure 7a: fractions and observed-area density."""

import numpy as np
from plaque_analysis.settings.microenvironment import CLASS_ORDER, COLORS, DISPLAY_TYPES, TYPE_LABELS
from plaque_analysis.plotting.microenvironment_style import clean, heading


def draw(container, tables):
    heading(container, "a", "Cell composition and density")
    axes = container.subplots(1, 2, gridspec_kw={"wspace": .22})
    data = tables["panel_a_composition"]
    rng = np.random.default_rng(529)
    for j, (ax, field, label) in enumerate(zip(axes, ("fraction_percent", "log_density"), ("Fraction (%)", "log₁₀(1 + cells/mm²)"))):
        for i, cell in enumerate(DISPLAY_TYPES):
            for k, group in enumerate(CLASS_ORDER):
                values = data.loc[data.display_type.eq(cell) & data.S_label.eq(group), field].to_numpy()
                values = values[np.isfinite(values)]
                y = 5 - i + (1 - k) * .2
                ax.scatter(values, y + rng.uniform(-.045, .045, len(values)), s=3.2,
                           color=COLORS[group], alpha=.28, linewidths=0)
                if len(values):
                    low, median, high = np.quantile(values, [.25, .5, .75])
                    ax.plot([low, high], [y, y], color=COLORS[group], lw=1.2)
                    ax.plot(median, y, "o", color=COLORS[group], ms=2.4, mec="white", mew=.25)
        ax.set_yticks(range(5, -1, -1), TYPE_LABELS if j == 0 else [])
        ax.set_ylim(-.55, 5.55)
        ax.set_xlabel(label)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        clean(ax, "x")
        if j == 0:
            ax.set_xlim(-2, 102)
            ax.set_xticks([0, 50, 100])
        else:
            ax.set_xlim(-.07, max(4.2, float(data.log_density.max()) + .1))
            ax.set_xticks([0, 2, 4])
