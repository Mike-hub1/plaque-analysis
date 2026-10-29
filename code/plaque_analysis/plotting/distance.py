"""Figure 7e: paired differences and calculated spatial-block intervals."""

import numpy as np
from matplotlib.lines import Line2D
from plaque_analysis.settings.microenvironment import PRIMARY, CLASS_ORDER, COLORS, INK
from plaque_analysis.plotting.microenvironment_style import clean, heading


def draw(container, tables):
    heading(container, "e", "Within-plaque far − near")
    axes = container.subplots(2, 1, gridspec_kw={"hspace": .48})
    data, model = tables["panel_e_differences"], tables["panel_e_intervals"]
    for k, (ax, code) in enumerate(zip(axes, PRIMARY)):
        rng = np.random.default_rng(222 + k)
        labels = []
        for j, group in enumerate(CLASS_ORDER):
            ns = []
            for u, window in enumerate(("150-300", ">300")):
                q = data[data.program.eq(code) & data.S_label.eq(group) & data.far_window.eq(window)]
                selected = model[model.program.eq(code) & model.S_label.eq(group) & model.far_window.eq(window)]
                x = j + (u-.5)*.28
                ax.scatter(x + rng.uniform(-.04, .04, len(q)), q.delta_display_x1000,
                           s=5, marker="o" if u == 0 else "s", alpha=.32,
                           facecolors=COLORS[group] if u == 0 else "none", edgecolors=COLORS[group], linewidths=.3)
                if len(selected):
                    r = selected.iloc[0]
                    if np.isfinite([r.low, r.high]).all():
                        ax.plot([x, x], [1000*r.low, 1000*r.high], color=COLORS[group], lw=1)
                    if np.isfinite(r.effect):
                        ax.plot(x, 1000*r.effect, marker="o" if u == 0 else "s", ms=3.6,
                                mfc=COLORS[group] if u == 0 else "white", mec=COLORS[group], mew=.9, ls="none", zorder=5)
                    ns.append(int(r.n))
                else:
                    ns.append(0)
            labels.append(group + f"\nn={ns[0]}/{ns[1]}")
        ax.set_xticks(range(3), labels)
        ax.axhline(0, color=INK, lw=.7, ls=(0, (2, 2)))
        ax.set_xlim(-.48, 2.48)
        ax.set_ylabel(("Neuronal" if k == 0 else "UPR") + " Δ\n(×10⁻³)")
        clean(ax)
    handles = [Line2D([0], [0], marker="o", color=INK, ls="none", ms=3),
               Line2D([0], [0], marker="s", color=INK, markerfacecolor="white", ls="none", ms=3)]
    axes[0].legend(handles, ["150–300 µm", ">300 µm"], fontsize=6, frameon=False, ncol=2)
