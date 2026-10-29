"""Figure 3: plot relations."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_data import empirical_curve
from plaque_analysis.plotting.fluorescence_helpers import draw_channel_scatter, small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, panel_heading, style_axis
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, save_figure
import numpy as np


def render_relationship_plate(plt, data, raw, signed, masks, saturation, spacing, base, dpi):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(183/25.4, 140/25.4))
    figure_text(fig, 3, 136, "Aβ–LAMP1 relationships | spatial co-occurrence is not intensity synchrony", fontsize=9, weight="bold")
    figure_text(fig, 3, 130, "Joint descriptors include LAMP1 and are not Aβ-only predictors. Open symbols denote QC-withheld diagnostic values.", fontsize=6.2)
    pixel = data["pixel"]
    lookup = data["master"].set_index("feature")
    panel_heading(fig, "a", "Binary spatial co-occurrence", 3, 122)
    ax = figure_axis(fig, 15, 88, 35, 30)
    colors = ["#EBEBEF", FIGURE_COLORS["A"], FIGURE_COLORS["L"], FIGURE_COLORS["both"]]
    for a in (0, 1):
        for l in (0, 1):
            count = int(((pixel.abeta_positive == bool(a)) & (pixel.lamp1_positive == bool(l))).sum())
            group = a+2*l
            ax.add_patch(Rectangle((a-.5, l-.5), 1, 1, facecolor=colors[group], edgecolor="white", lw=2))
            ax.text(a, l, f"{count} pixels\n{count*np.prod(spacing):.0f} μm²", fontsize=6.3,
                    ha="center", va="center", color=FIGURE_COLORS["ink"] if group == 0 else "white")
    ax.set(xlim=(-.5, 1.5), ylim=(-.5, 1.5), xlabel="Aβ mask", ylabel="LAMP1 mask")
    ax.set_xticks([0, 1], ["−", "+"]); ax.set_yticks([0, 1], ["−", "+"])
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    figure_text(fig, 9, 78, f"Window support: {len(pixel)} observed pixels", fontsize=5.7)
    panel_heading(fig, "b", "Signals and marginal distributions", 62, 122)
    ax = figure_axis(fig, 76, 87, 31, 25)
    draw_channel_scatter(ax, data, saturation[0], raw[0].astype(float)-signed[0])
    ax.set_xlabel("Aβ signal (a.u.)", fontsize=5.7, labelpad=2)
    ax.set_ylabel("LAMP1 (a.u.)", fontsize=5.7, labelpad=2)
    top = figure_axis(fig, 76, 114, 31, 5)
    right = figure_axis(fig, 109, 87, 6, 25)
    top.hist(pixel.abeta_signed, bins=24, color=FIGURE_COLORS["A"], alpha=.75)
    right.hist(pixel.lamp1_signed, bins=24, orientation="horizontal", color=FIGURE_COLORS["L"], alpha=.75)
    top.set_xlim(ax.get_xlim()); right.set_ylim(ax.get_ylim()); top.axis("off"); right.axis("off")
    figure_text(fig, 75, 77, "All window pixels; no regression fitted", fontsize=5.6)
    panel_heading(fig, "c", "Domain-dependent correlations", 122, 122)
    ax = figure_axis(fig, 151, 90, 25, 26)
    style_axis(ax)
    ax.axvline(0, color="#A6A6AD", lw=.6)
    keys = ["pearson_window", "pearson_foreground_union", "pearson_double_positive_conditional"]
    domains = [np.ones(len(pixel), bool), (pixel.abeta_positive | pixel.lamp1_positive).to_numpy(),
               (pixel.abeta_positive & pixel.lamp1_positive).to_numpy()]
    labels = []
    for y, (key, name, domain) in enumerate(zip(keys, ["Window", "Union", "Double +"], domains)):
        row = lookup.loc[key]
        value = row.value if np.isfinite(row.value) else row.diagnostic_value
        if np.isfinite(value):
            ax.plot([0, value], [y, y], color=FIGURE_COLORS["both"], lw=1)
            ax.plot(value, y, "o", ms=4, mec=FIGURE_COLORS["both"],
                    mfc=FIGURE_COLORS["both"] if np.isfinite(row.value) else "white")
            ax.text(value, y-.22, f"{value:.2f}", fontsize=5.5, ha="center")
        labels.append(f"{name} (n={int(domain.sum())})")
    ax.set_yticks(range(3), labels, fontsize=5.5)
    ax.set(xlim=(-1, 1), ylim=(2.7, -.7), xlabel="Pearson r")
    figure_text(fig, 132, 79, "No CI; n counts pixels, not animals", fontsize=5.5)
    panel_heading(fig, "d", "Overlap metrics answer distinct questions", 3, 65)
    ax = figure_axis(fig, 31, 24, 23, 34)
    style_axis(ax)
    keys = ["dice", "jaccard", "abeta_area_covered_by_lamp1_fraction", "lamp1_area_inside_abeta_fraction",
            "manders_A_in_L", "manders_L_in_A"]
    labels = ["Dice", "Jaccard", "Aβ area in LAMP1", "LAMP1 area in Aβ", "Manders Aβ→L", "Manders L→Aβ"]
    for y, key in enumerate(keys):
        row = lookup.loc[key]
        value = row.value if np.isfinite(row.value) else row.diagnostic_value
        if np.isfinite(value):
            ax.plot([0, value], [y, y], color="#C8C2CC", lw=1)
            ax.plot(value, y, "o", ms=4, mec=FIGURE_COLORS["both"],
                    mfc=FIGURE_COLORS["both"] if np.isfinite(row.value) else "white")
    ax.set_yticks(range(6), labels, fontsize=5.5)
    ax.set(xlim=(0, 1.03), ylim=(5.6, -.6), xlabel="Coefficient / fraction")
    panel_heading(fig, "e", "LAMP1 inside and outside Aβ", 64, 65)
    ax = figure_axis(fig, 77, 24, 37, 33)
    style_axis(ax)
    for inside, color, label in ((True, FIGURE_COLORS["A"], "Inside Aβ"), (False, FIGURE_COLORS["L"], "Outside Aβ")):
        values = pixel.loc[pixel.abeta_positive == inside, "lamp1_signed"]
        empirical_curve(ax, values, color, label=f"{label} (n={len(values)})")
    ax.set(xlabel="LAMP1 signal (a.u.)", ylabel="Pixel cumulative fraction", ylim=(0, 1.03))
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1)
    panel_heading(fig, "f", "Outer-neighbourhood composition", 122, 65)
    size_key = figure_axis(fig, 135, 59.5, 41, 3)
    for j, proportion in enumerate((.1, .25, .5)):
        size_key.scatter(j, .5, s=proportion*150, facecolors="white", edgecolors="#5F5966", linewidths=.4)
        size_key.text(j+.18, .5, f"{proportion:.2f}", fontsize=5.4, va="center")
    size_key.set(xlim=(-.25, 2.8), ylim=(0, 1)); size_key.axis("off")
    ax = figure_axis(fig, 141, 25, 34, 32)
    composition = data["composition"]
    cmap = LinearSegmentedColormap.from_list("composition_delta", [FIGURE_COLORS["A"], "#F3F0EF", FIGURE_COLORS["L"]])
    norm = Normalize(-.5, .5)
    ax.scatter(composition.lamp1_state, composition.abeta_rank, s=composition.proportion*150,
               c=composition.proportion-composition.baseline_proportion, cmap=cmap, norm=norm,
               edgecolors="#5F5966", linewidths=.35)
    for rank, rows in composition.groupby("abeta_rank"):
        ax.text(5.65, rank, f"{int(rows.denominator_pixels.iloc[0])}", fontsize=5.5, va="center")
    n_ranks = int(composition.abeta_rank.max()) if len(composition) else 1
    ax.set(xlim=(-.6, 6.2), ylim=(.4, n_ranks+.6),
           xlabel="LAMP1 state (0 = below threshold)", ylabel="Aβ intensity rank (1 = brightest)")
    ax.set_xticks(range(6)); ax.set_yticks(range(1, n_ranks+1)); ax.tick_params(labelsize=5.5)
    ax.xaxis.label.set_fontsize(5.5); ax.yaxis.label.set_fontsize(5.8)
    ax.text(5.65, n_ranks+.6, "n", fontsize=5.5, ha="center")
    if not len(composition):
        ax.text(.5, .5, "No valid neighbourhoods", transform=ax.transAxes, fontsize=5.5, ha="center")
    small_colorbar(fig, ScalarMappable(norm=norm, cmap=cmap), (143, 12, 30, 1.5), "Proportion − window baseline", ticks=[-.5, 0, .5])
    figure_text(fig, 4, 11, "Filled: computed; open: diagnostic / withheld", fontsize=5.7, panel="d")
    figure_text(fig, 65, 11, "f: bubble area = proportion (0–1); diagnostic ranks", fontsize=5.5, panel=None)
    figure_text(fig, 3, 2, "No colocalization, repulsion or core–halo mechanism is inferred from any single coefficient. Neighbourhoods are static and may share pixels.", fontsize=5.7, panel=None)
    save_figure(fig, base, dpi)

