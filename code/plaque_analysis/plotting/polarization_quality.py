"""Figure 4: plotting qc."""
from __future__ import annotations

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from plaque_analysis.plotting.polarization_style import INK, LIGHT, ORANGE, TEAL, export, map_panel, scalebar, style


def plot_qc(folder, maps, tissue, analysis, config, angle_rows, threshold_rows, loo_rows):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(config["figure_width_mm"]/25.4, 119/25.4))
    fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.10, wspace=.47, hspace=.72)
    labels = ["Tissue and axis support", "Angle hold-out quality", "Held-out intensity prediction",
              "Fourth-harmonic reference", "Orientation quality sensitivity", "Feature sensitivity to one angle"]
    for letter, title, ax in zip("abcdef", labels, axes.flat):
        ax.set_title(title, fontsize=7.5, loc="left", pad=7)
        ax.text(-.19, 1.07, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, color="black")
    a = axes[0, 0]
    a.imshow(maps["A0"], cmap="gray", vmin=np.percentile(maps["A0"], 1), vmax=np.percentile(maps["A0"], 99))
    a.contour(tissue, [.5], colors=["#D69B3A"], linewidths=.5)
    a.contour(analysis["valid_orientation"], [.5], colors=["#22B7C5"], linewidths=.4)
    a.axis("off")
    scalebar(a, (tissue.shape[1], tissue.shape[0]), 2/config["pixel_size_um"], "2 µm", color=INK)
    a.legend(handles=[Line2D([0], [0], color="#D69B3A", label="Tissue mask"),
                      Line2D([0], [0], color="#22B7C5", label="Axis support")],
             loc="lower center", bbox_to_anchor=(.5, -.15), ncol=2, fontsize=5.5)
    a = axes[0, 1]
    im = map_panel(a, maps["angle_holdout_R2"], tissue, config["pixel_size_um"], "RdBu", -1, 1)
    cb = fig.colorbar(im, ax=a, fraction=.045, pad=.025); cb.set_label("Held-out R²", fontsize=6)
    a = axes[0, 2]
    angles = np.array([r["angle_deg"] for r in angle_rows])
    for key, name, colour, dx in [("second", "2nd", TEAL, -1.2), ("second_fourth", "2nd + 4th", ORANGE, 1.2)]:
        m = np.array([r[key+"_median_pct_A0"] for r in angle_rows])
        q1 = np.array([r[key+"_q25_pct_A0"] for r in angle_rows])
        q3 = np.array([r[key+"_q75_pct_A0"] for r in angle_rows])
        a.errorbar(angles+dx, m, yerr=[m-q1, q3-m], fmt="o-", ms=2, lw=.7,
                   elinewidth=.6, color=colour, label=name)
    a.set_xlabel("Omitted angle (°)"); a.set_ylabel("Absolute error / A₀ (%)")
    a.set_ylim(bottom=0); a.legend(fontsize=5.5)
    a = axes[1, 0]
    im = map_panel(a, maps["M4_reference"]*100, tissue, config["pixel_size_um"], "magma", 0, 4)
    cb = fig.colorbar(im, ax=a, fraction=.045, pad=.025); cb.set_label("M₄ reference (%)", fontsize=6)
    a = axes[1, 1]
    x = [r["formal_se_threshold_deg"] for r in threshold_rows]
    for key, name, colour in [("S_global", "Global order", TEAL), ("local_S_median", "Local order", ORANGE)]:
        a.plot(x, [r[key] for r in threshold_rows], "o-", ms=3, color=colour, label=name)
    a.set_xticks(x, [f"{t:g}°\n{r['valid_fraction']:.0%} retained" for t, r in zip(x, threshold_rows)], fontsize=5.5)
    a.set_xlabel("Maximum formal axis SE"); a.set_ylabel("Axial order S")
    a.set_ylim(0, 1.05); a.legend(fontsize=5.5)
    a = axes[1, 2]
    selected = [("optical_m2_median", r"Median $M_2$"),
                ("optical_m2_quartile_dispersion", r"$M_2$ dispersion"),
                ("orientation_S_global", "Global order"),
                ("orientation_local_S_median", "Local order")]
    offsets = np.linspace(-.12, .12, len(loo_rows))
    for index, (key, label) in enumerate(selected):
        baseline = analysis["values"][key]
        change = np.array([(r[key]-baseline)/abs(baseline)*100 for r in loo_rows])
        a.plot([change.min(), change.max()], [index, index], color=LIGHT, lw=3)
        a.scatter(change, index+offsets, s=6, color=TEAL, zorder=3)
    a.axvline(0, color="#9DA8AC", lw=.6, ls="--")
    a.set_yticks(np.arange(len(selected)), [x[1] for x in selected], fontsize=6)
    a.invert_yaxis(); a.set_xlabel("Change from all-angle fit (%)")
    fig.text(.075, .960, "Reconstruction and feature stability", fontsize=10, fontweight="bold")
    export(fig, folder, "Figure4_S1_quality_control", config["raster_dpi"])
