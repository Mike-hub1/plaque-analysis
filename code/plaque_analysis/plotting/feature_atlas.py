"""Figure 3: plot atlas."""
from __future__ import annotations

from plaque_analysis.plotting.fluorescence_data import feature_atlas_data
from plaque_analysis.plotting.fluorescence_helpers import small_colorbar
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_text, plot_value
from plaque_analysis.plotting.fluorescence_style import FIGURE_COLORS, save_figure
import numpy as np


def render_feature_atlas(plt, data, role, base, dpi, source_dir, *, preserve_source_data=False):
    from matplotlib.colors import LinearSegmentedColormap
    master, scenarios, table = feature_atlas_data(data, role)
    n = len(master)
    height = min(170, max(95, 38+2.45*n))
    fig = plt.figure(figsize=(183/25.4, height/25.4))
    title = {"X_A": "Aβ-only candidates", "Y_L": "LAMP1 readouts and window QC", "Z_AL": "Joint-channel descriptors"}[role]
    figure_text(fig, 3, height-5, f"{title} | complete feature and sensitivity atlas", fontsize=9, weight="bold")
    figure_text(fig, 3, height-11, f"{n} fields; baseline values retain their own units. One-factor perturbations of one image, not repeatability estimates.", fontsize=6)
    count = max(1, len(scenarios))
    matrix = np.full((n, count), np.nan)
    annotations = np.full((n, count), "NA", dtype=object)
    if len(table):
        for r in table.itertuples():
            i, j = r.feature_row, r.scenario_column
            if r.display_status == "relative_change":
                matrix[i, j] = r.relative_change_percent
                annotations[i, j] = "0" if abs(r.relative_change_percent) < .05 else f"{r.relative_change_percent:+.2g}"
            elif r.display_status == "unchanged_zero":
                matrix[i, j] = 0
                annotations[i, j] = "="
            elif r.display_status == "zero_baseline_absolute_change":
                annotations[i, j] = "Δ"+f"{r.absolute_change:.2g}"
    cmap = LinearSegmentedColormap.from_list("sensitivity_signed", [FIGURE_COLORS["A"], "#FCFBF9", FIGURE_COLORS["L"]])
    cmap.set_bad("#E1E2E5")
    ax = figure_axis(fig, 113, 24, 64, height-43)
    im = ax.imshow(np.ma.masked_invalid(matrix), origin="upper", aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=-30, vmax=30)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.spines[:].set_visible(False)
    names = {"A_threshold": "Aβ thr", "L_threshold": "L thr", "window_radius": "Radius",
             "ring_width": "Ring", "L_registration_x": "Shift x"}
    for j in range(count):
        if len(scenarios):
            scenario = scenarios.iloc[j]
            suffix = f"{scenario.factor:+g} px" if scenario.parameter == "L_registration_x" else f"×{scenario.factor:g}"
            text = names.get(scenario.parameter, scenario.parameter)+"\n"+suffix
        else:
            text = "Not run"
        ax.text(j, -.8, text, ha="center", va="bottom", fontsize=5.3, linespacing=1.05, clip_on=False)
    role_color = FIGURE_COLORS[{"X_A": "A", "Y_L": "L", "Z_AL": "both"}[role]]
    units = {"detector a.u.": "a.u.", "detector a.u.·pixel": "a.u.·px", "detector a.u.·μm²": "a.u.·μm²",
             "count/(1000 μm²)": "n/1000 μm²", "boolean": "0/1", "count": "n"}
    for i, row in master.iterrows():
        y = 24+(height-43)*(1-(i+.5)/n)
        available = np.isfinite(row.value)
        marker_color = ("#999BA2" if row.measurement_kind != "phenotype" else role_color) if available else FIGURE_COLORS["withheld"]
        figure_text(fig, 3, y, "●" if available else "○", color=marker_color, fontsize=5.5, va="center")
        figure_text(fig, 6, y, row.feature, fontsize=5.5, va="center")
        figure_text(fig, 81, y, plot_value(row.value), fontsize=5.5, va="center", ha="right")
        figure_text(fig, 85, y, units.get(row.units, row.units), fontsize=5.3, va="center", color="#666671")
        for j in range(count):
            color = "white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= 22 else "#46434C"
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=5.1, color=color)
        if i and row.family != master.iloc[i-1].family:
            ax.axhline(i-.5, color="white", lw=.8)
    figure_text(fig, 6, height-17, "Feature code (complete CSV order)", fontsize=6, weight="bold")
    figure_text(fig, 81, height-17, "Value", ha="right", fontsize=6, weight="bold")
    figure_text(fig, 85, height-17, "Unit", fontsize=6, weight="bold")
    small_colorbar(fig, im, (125, 16, 40, 1.8), "Signed relative change (%)", ticks=[-30, 0, 30])
    figure_text(fig, 3, 19, "● Computed field / observed QC    ○ Withheld at baseline    NA: unavailable", fontsize=5.8)
    figure_text(fig, 3, 13, "Colour scale saturates at ±30%; cell text retains the actual change. '=': unchanged zero.", fontsize=5.6)
    figure_text(fig, 3, 7, "Δ: absolute change when baseline is zero; no percentage is defined. Missing baseline values are never replaced by diagnostic values.", fontsize=5.6)
    figure_text(fig, 3, 2, "Threshold/window changes and pixel shifts do not establish biological stability or ICC; all parameters remain subject to independent validation.", fontsize=5.6)
    if not preserve_source_data:
        table.to_csv(source_dir/f"feature_atlas_{role}_display_values.csv", index=False)
    save_figure(fig, base, dpi)

