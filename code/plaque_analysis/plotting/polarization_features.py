"""Figure 4: plotting features."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from plaque_analysis.features.polarization_table import feature_slice, feature_value
from plaque_analysis.plotting.polarization_helpers import DEPTH_COLOURS
from plaque_analysis.plotting.polarization_style import INK, TEAL, export, style


def feature_figure_title(fig, title, subtitle):
    fig.text(.045, .96, title, fontsize=11, weight="bold", va="top")
    fig.text(.045, .923, subtitle, fontsize=6.8, color="#647480", va="top")


def feature_panel_title(ax, letter, text):
    ax.text(-.10, 1.10, letter, transform=ax.transAxes, fontsize=9, weight="bold", va="baseline")
    ax.text(0, 1.10, text, transform=ax.transAxes, fontsize=8, va="baseline")


def plot_complete_feature_atlas(table, folder, dpi):
    """Every numerical mark is read from the exported complete CSV, not new fits."""
    style()
    curated = table.loc[table.curated.eq(True) & table.scenario.eq("baseline")].copy()
    order = ["Optical response", "Orientation", "Morphology", "PA anisotropy", "Optical reference",
             "Conditional axis", "Conditional conversion"]
    curated["family_order"] = curated.family.map({f: k for k, f in enumerate(order)})
    curated = curated.sort_values("family_order", kind="stable").reset_index(drop=True)
    fig = plt.figure(figsize=(183/25.4, 178/25.4))
    feature_figure_title(fig, "Physical feature atlas", "Baseline values in native units and sensitivity to omitting one polarization frame")
    y_top, step = .825, .026
    sensitivity = np.full((len(curated), 11), np.nan)
    for i, row in curated.iterrows():
        for j in range(11):
            candidate = table.loc[(table.feature == row.feature) & (table.scope == "global") &
                                  (table.scenario == f"omit_angle_{j+1:02d}")]
            if (len(candidate) and np.isfinite(row.value) and row.value != 0
                    and row.feature != "orientation_mean_axis_deg"):
                sensitivity[i, j] = 100*(float(candidate.iloc[0].value)-row.value)/abs(row.value)
    x0, x1 = .565, .965
    ax = fig.add_axes([x0, y_top-(len(curated)-.5)*step, x1-x0, len(curated)*step])
    cmap = LinearSegmentedColormap.from_list("feature_sensitivity", ["#327591", "#FAFBFC", "#C8886B"])
    cmap.set_bad("#F0F2F4")
    im = ax.imshow(sensitivity, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-30, vmax=30)
    ax.set_axis_off()
    family_colours = {"Optical response": "#457DAB", "Orientation": TEAL, "Morphology": "#6F7A8D",
                      "PA anisotropy": "#5AA7A7", "Optical reference": "#A38A70",
                      "Conditional axis": "#AB9681", "Conditional conversion": "#AB9681"}
    fig.text(.045, .865, "Feature", fontsize=7, weight="bold")
    fig.text(.438, .865, "Value", fontsize=7, weight="bold", ha="right")
    fig.text(.450, .865, "Unit", fontsize=7, weight="bold")
    fig.text((x0+x1)/2, .883, "Omitted frame", fontsize=7, ha="center", weight="bold")
    for j in range(11):
        fig.text(x0+(j+.5)*(x1-x0)/11, .863, str(j+1), ha="center", fontsize=6.3)
    for i, row in curated.iterrows():
        y = y_top-i*step
        color = family_colours[row.family]
        fig.add_artist(Line2D([.036, .036], [y-.009, y+.009], transform=fig.transFigure,
                             lw=2.1, color=color, solid_capstyle="round"))
        if i and row.family != curated.iloc[i-1].family:
            fig.add_artist(Line2D([.045, .965], [y+step*.5, y+step*.5], transform=fig.transFigure,
                                 lw=.5, color="#D8E0E5"))
        fig.text(.05, y, row.label, fontsize=6.2, va="center")
        conditional = not np.isfinite(row.value) and np.isfinite(row.diagnostic_value)
        value = row.diagnostic_value if conditional else row.value
        formatted = (f"{value:.4g}"+"*" if conditional else f"{value:.4g}") if np.isfinite(value) else "NA"
        fig.text(.438, y, formatted, fontsize=6.5, va="center", ha="right", color=color)
        fig.text(.45, y, {"degree": "°", "1": "—"}.get(row.unit, row.unit), fontsize=6.1, va="center", color="#62707A")
        for j in range(11):
            change = sensitivity[i, j]
            label = "—" if not np.isfinite(change) else "0" if abs(change) < .05 else f"{change:+.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=5.1,
                    color="white" if np.isfinite(change) and abs(change) > 24 else INK)
    cbax = fig.add_axes([.66, .128, .23, .012])
    cb = fig.colorbar(im, cax=cbax, orientation="horizontal", ticks=[-30, 0, 30], extend="both")
    cb.outline.set_visible(False)
    cb.set_label("Relative change (%)", fontsize=6.5, labelpad=2)
    fig.text(.045, .16, "Colour is capped at ±30%; cell text gives the actual change. A dash means untested or unavailable.", fontsize=6.1)
    fig.text(.045, .098, "* Conditional reference only; calibration is unconfirmed. Accepted CSV values remain empty.", fontsize=6.1)
    fig.text(.045, .072, "M4 is an optical reference. PA DOA comes from an independent scan; no optical-frame perturbation is applied to PA.", fontsize=6.1)
    fig.text(.045, .045, "Baseline masks are fixed: zero shape changes follow that choice. One plaque; technical sensitivity, not biological uncertainty.", fontsize=6.1)
    export(fig, folder, "01_physical_feature_atlas", dpi)


def plot_feature_distributions(table, folder, dpi):
    specs = [("M2", "Polarization response", "M2 (%)", 100, "global", TEAL),
             ("local_S", "Local axial order", "S", 1, "global", TEAL),
             ("angular_spread_deg", "Angular spread", "Angular spread (°)", 1, "global", TEAL),
             ("M4_reference", "Optical reference", "M4 (%)", 100, "global", "#A38A70"),
             ("PA_DOA", "Native PA anisotropy", "DOA", 1, "global", "#528F9C"),
             ("PA_pair_abs_difference", "Adjacent-pair contrast", "Absolute DOA difference", 1, "global", "#528F9C"),
             ("PA_point_neighbour_mean", "Point-wise neighbour contrast", "Mean absolute DOA difference", 1, "global", "#528F9C"),
             ("fit_R2", "Harmonic fit quality", "R²", 1, "qc_distribution", "#8797AE"),
             ("phi_formal_se_deg", "Direction fit precision", "Formal SE (°)", 1, "qc_distribution", "#8797AE")]
    fig = plt.figure(figsize=(183/25.4, 162/25.4))
    feature_figure_title(fig, "Distributions within the plaque", "All supported observations contribute to the displayed quantiles and summary statistics")
    for k, (metric, title_text, xlabel, factor, scope, color) in enumerate(specs):
        ax = fig.add_axes([.075+(k%3)*.321, .705-(k//3)*.270, .247, .10])
        sub = feature_slice(table, scope, metric)
        stats = dict(zip(sub.statistic, sub.value))
        n = int(sub.iloc[0].support_n)
        vals = [stats[s]*factor for s in ("min", "q05", "q25", "median", "q75", "q95", "max")]
        ax.plot([vals[0], vals[6]], [.5, .5], color="#D5DFE3", lw=1.0, zorder=1)
        ax.plot([vals[1], vals[5]], [.5, .5], color=color, lw=1.5, zorder=2)
        ax.plot([vals[2], vals[4]], [.5, .5], color=color, lw=6, alpha=.32, solid_capstyle="butt", zorder=3)
        ax.scatter([vals[3]], [.5], s=22, facecolors="white", edgecolors=color, linewidth=1.2, zorder=4)
        ax.scatter([vals[0], vals[6]], [.5, .5], s=8, marker="|", color="#899AA4", zorder=2)
        span = max(vals[6]-vals[0], abs(vals[6])*.05, 1e-6)
        ax.set_xlim(vals[0]-.06*span, vals[6]+.06*span)
        if metric == "phi_formal_se_deg":
            # Show the entire tail while retaining resolution near the bulk.
            ax.set_xscale("symlog", linthresh=5, linscale=1)
            ax.set_xlim(0, vals[6]*1.15)
            ax.set_xticks([0, 5, 50, 500], ["0", "5", "50", "500"])
            xlabel = "Formal SE (°; symlog)"
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel, labelpad=2, fontsize=6.8)
        if metric != "phi_formal_se_deg":
            ax.locator_params(axis="x", nbins=3)
        feature_panel_title(ax, chr(97+k), title_text)
        ax.text(.5, -.58, f"Median {stats['median']*factor:.3g}  ·  IQR {stats['iqr']*factor:.3g}", transform=ax.transAxes, ha="center", fontsize=6.2)
        ax.text(.5, -.84, f"Mean {stats['mean']*factor:.3g}  ·  SD {stats['std']*factor:.3g}  ·  n = {n:,}", transform=ax.transAxes, ha="center", fontsize=5.9, color="#62707A")
    fig.text(.045, .051, "Circle: median; thick interval: IQR; coloured line: 5–95%; pale line: min–max. Panel i uses symlog (linear through 5°).", fontsize=6.2)
    fig.text(.045, .029, "n counts pixels, supported windows or PA scan points/pairs. SD and intervals describe spatial dispersion in one plaque.", fontsize=6.2)
    export(fig, folder, "02_distribution_atlas", dpi)


def plot_feature_depth(table, folder, dpi):
    fig = plt.figure(figsize=(183/25.4, 160/25.4))
    feature_figure_title(fig, "Spatial organization of physical features", "Geometry-defined depth regions and position-azimuth sectors of the same two-dimensional plaque")
    for k, (metric, ylabel, factor, name) in enumerate([("M2", "M2 (%)", 100, "Polarization response"),
                                                       ("local_S", "Local order S", 1, "Axial organization"),
                                                       ("angular_spread_deg", "Angular spread (°)", 1, "Angular heterogeneity")]):
        ax = fig.add_axes([.08+k*.322, .60, .237, .24])
        values = feature_slice(table, "depth", metric).pivot(index="depth_region", columns="statistic", values="value").sort_index()*factor
        x = values.index.to_numpy(float)
        ax.fill_between(x, values.q05, values.q95, color="#A8D1E1", alpha=.21, lw=0)
        ax.fill_between(x, values.q25, values.q75, color="#62A9C8", alpha=.22, lw=0)
        ax.plot(x, values["median"], color=TEAL, lw=1.0, zorder=3)
        for j in range(len(x)):
            ax.scatter(x[j], values["median"].iloc[j], facecolor=DEPTH_COLOURS[j], edgecolor="white", s=24, lw=.6, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "2", "3", "4", "5"])
        ax.set_xlim(.7, 5.3)
        ax.set_xlabel("Depth region: rim → core")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E3E9ED", lw=.5, zorder=0)
        if metric == "local_S":
            ax.set_ylim(0, 1.03)
        else:
            ax.set_ylim(bottom=0)
        feature_panel_title(ax, chr(97+k), name)
    for k, (metric, factor, name, cmap) in enumerate([("M2", 100, "Median M2 (%)", "cividis"),
                                                    ("local_S", 1, "Median local S", "magma")]):
        ax = fig.add_axes([.09+k*.48, .19, .35, .24])
        sub = feature_slice(table, "depth_azimuth", metric, "median")
        vals = sub.pivot(index="depth_region", columns="azimuth_sector", values="value").sort_index().to_numpy()*factor
        low, high = (0, max(10, float(np.nanmax(vals)))) if metric == "M2" else (0, 1)
        im = ax.imshow(vals, origin="lower", aspect="auto", interpolation="nearest", extent=(0, 360, .5, 5.5),
                       cmap=cmap, vmin=low, vmax=high)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                value = vals[i, j]
                color = "white" if (value-low)/(high-low) < .56 else INK
                ax.text((j+.5)*45, i+1, f"{value:.1f}" if metric == "M2" else f"{value:.2f}", fontsize=5.8,
                        ha="center", va="center", color=color)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks(range(1, 6))
        ax.set_xlabel("Position azimuth (°)")
        ax.set_ylabel("Depth region")
        feature_panel_title(ax, chr(100+k), name)
        cax = fig.add_axes([.445+k*.48, .19, .010, .24])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5)
    fig.text(.045, .075, "Lines and points: medians; dark band: IQR; light band: 5–95%. No values above the former 35° display limit are excluded.", fontsize=6.1)
    fig.text(.045, .047, "Position azimuth runs clockwise from image right (0–360°); optical axes remain axial (0–180°).", fontsize=6.2)
    fig.text(.045, .023, "Each heatmap cell requires at least 50 valid observations. Spatial bins and overlapping windows are not biological replicates.", fontsize=6.1)
    export(fig, folder, "03_depth_and_azimuth", dpi)


def plot_feature_scales(table, folder, dpi):
    fig = plt.figure(figsize=(183/25.4, 158/25.4))
    feature_figure_title(fig, "Scale dependence and spatial heterogeneity", "Analysis windows use their actual physical widths; all quantification is performed before plotting")
    ax = fig.add_axes([.09, .59, .35, .25])
    scale = feature_slice(table, "scale", "local_S").pivot(index="window_um", columns="statistic", values="value").sort_index()
    x = scale.index.to_numpy(float)
    ax.fill_between(x, scale.q05, scale.q95, color="#B0D4DC", alpha=.24, lw=0)
    ax.fill_between(x, scale.q25, scale.q75, color="#78B5C3", alpha=.3, lw=0)
    ax.plot(x, scale["median"], "o-", color=TEAL, ms=3, mfc="white", lw=1.2)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, x.max()+.08)
    ax.set_ylabel("Local order S")
    ax.set_xlabel("Window width (µm)")
    feature_panel_title(ax, "a", "Organization across physical scales")
    n = int(feature_slice(table, "scale", "local_S").iloc[0].support_n)
    ax.text(.98, .08, f"{n:,} shared centres", ha="right", transform=ax.transAxes, fontsize=6.5)
    ax = fig.add_axes([.585, .59, .35, .25])
    for metric, color, label in [("C2", TEAL, "C2"), ("G2_connected", "#8B86AD", "Connected G2")]:
        sub = feature_slice(table, "correlation", metric).sort_values("lag_bin")
        ax.plot(sub.lag_um, sub.value, color=color, lw=1.15, label=label)
        if metric == "C2":
            folds = feature_slice(table, "correlation", metric, scenario=None)
            folds = folds.loc[folds.scenario != "baseline"].pivot(index="lag_bin", columns="scenario", values="value")
            ax.fill_between(sub.lag_um, folds.min(axis=1), folds.max(axis=1), color=TEAL, alpha=.17, lw=0)
    half = feature_value(table, "global", "orientation_half_decay_um")
    ax.axhline(.5, color="#B3BBC4", ls=(0, (3, 2)), lw=.6)
    if np.isfinite(half):
        ax.axvline(half, color="#B3BBC4", ls=(0, (3, 2)), lw=.6)
        ax.text(.98, .84, f"Half decay {half:.3f} µm", transform=ax.transAxes, ha="right", fontsize=6.3)
    ax.set_xlabel("Pair distance (µm)")
    ax.set_ylabel("Axial correlation")
    ax.legend(fontsize=6, loc="lower left")
    feature_panel_title(ax, "b", "Pair-weighted spatial correlation")
    ax = fig.add_axes([.09, .19, .35, .25])
    hist = feature_slice(table, "orientation_histogram").sort_values("bin_left")
    ax.bar(hist.bin_left, hist.value*100, width=hist.bin_right-hist.bin_left,
           align="edge", color="#A8D1E1", edgecolor="white", linewidth=.5)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Configured optical axis (°)")
    ax.set_ylabel("Qualified directions (%)")
    feature_panel_title(ax, "c", "Axial orientation distribution")
    ax = fig.add_axes([.585, .19, .35, .25])
    hist = feature_slice(table, "pa_histogram").sort_values("bin_left")
    ax.bar(hist.bin_left, hist.value, width=hist.bin_right-hist.bin_left, align="edge",
           color="#7DAFB7", edgecolor="white", linewidth=.5)
    median = feature_value(table, "global", "PA_DOA", "median")
    q1, q3 = [feature_value(table, "global", "PA_DOA", stat) for stat in ("q25", "q75")]
    ax.axvspan(q1, q3, color="#D5E6E9", alpha=.45, zorder=0)
    ax.axvline(median, ls=(0, (3, 2)), lw=1, color=INK)
    ax.set_xlim(0, hist.bin_right.max())
    ax.set_xticks([0, .04, .08, .12, .16])
    ax.set_xlabel("DOA")
    ax.set_ylabel("PA scan points (%)")
    feature_panel_title(ax, "d", "Independent PA anisotropy")
    ax.text(.98, .92, f"Median {median:.3f}\nIQR {q3-q1:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    fig.text(.045, .075, "a: median, IQR and 5–95%. b: C2 band is the minimum–maximum across 11 omitted-frame fits.", fontsize=6.2)
    fig.text(.045, .047, "c: configured axis sign/zero is uncalibrated. d: dashed line and shading show the median and IQR of native PA DOA.", fontsize=6.1)
    fig.text(.045, .023, "All panels describe one plaque; spatial dispersion and technical sensitivity are not biological confidence intervals.", fontsize=6.2)
    export(fig, folder, "04_scales_and_correlations", dpi)


def plot_feature_qc(table, folder, dpi):
    fig = plt.figure(figsize=(183/25.4, 144/25.4))
    feature_figure_title(fig, "Support and technical sensitivity", "Explicit checks of direction selection, mask boundaries and held-out intensity prediction")
    ax = fig.add_axes([.09, .56, .34, .27])
    for metric, color, label in [("valid_fraction", "#8C9BAF", "Qualified fraction"), ("S_global", TEAL, "Global S"),
                                 ("local_S_median", "#599BB4", "Median local S")]:
        sub = feature_slice(table, "threshold_sensitivity", metric, scenario=None).sort_values("threshold_deg")
        ax.plot(sub.threshold_deg, sub.value, "o-", color=color, ms=3, lw=1, label=label)
    ax.set_xticks(sorted(sub.threshold_deg.unique()))
    ax.set_xlabel("Maximum formal axis SE (°)")
    ax.set_ylabel("Fraction / axial order")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=6)
    feature_panel_title(ax, "a", "Direction-support threshold")
    ax = fig.add_axes([.59, .56, .34, .27])
    for metric, color, label in [("M2_median", TEAL, "Median M2"), ("S_global", "#8C86AA", "Global S"),
                                 ("core_rim_ratio", "#BB987B", "Core/rim M2")]:
        sub = feature_slice(table, "boundary_sensitivity", metric, scenario=None).set_index("scenario")
        val = sub.loc[["boundary_eroded", "boundary_primary", "boundary_dilated"], "value"].to_numpy(float)
        pct = (val-val[1])/abs(val[1])*100 if val[1] else np.full(3, np.nan)
        ax.plot([-1, 0, 1], pct, "o-", color=color, lw=1, ms=3, label=label)
    ax.axhline(0, lw=.6, color="#CAD4DA")
    ax.set_xticks([-1, 0, 1], ["Eroded", "Baseline", "Dilated"])
    ax.set_ylabel("Relative change (%)")
    boundary = feature_slice(table, "boundary_sensitivity", scenario=None)
    radius_px, radius_um = boundary.boundary_radius_px.max(), boundary.boundary_radius_um.max()
    ax.set_xlabel(f"Mask perturbation ({radius_px:g} px = {radius_um:.4g} µm)")
    ax.legend(loc="best", fontsize=6)
    feature_panel_title(ax, "b", "Boundary sensitivity")
    ax = fig.add_axes([.09, .18, .56, .22])
    for model, color, label in [("second", "#98A7B5", "Second harmonic"),
                                ("second_fourth", TEAL, "Second + fourth harmonics")]:
        vals = []
        for stat in ("q25", "median", "q75"):
            sub = feature_slice(table, "angle_prediction", model+"_"+stat+"_pct_A0").sort_values("acquisition_angle_deg")
            vals.append(sub.value.to_numpy())
        angles = sub.acquisition_angle_deg.to_numpy()
        ax.fill_between(angles, vals[0], vals[2], color=color, alpha=.13, lw=0)
        ax.plot(angles, vals[1], "o-", color=color, lw=1, ms=2.5, label=label)
    ax.set_xlabel("Held-out acquisition angle (°)")
    ax.set_ylabel("Absolute error / A0 (%)")
    ax.legend(loc="upper left", fontsize=6)
    feature_panel_title(ax, "c", "Prediction from the remaining frames")
    ax = fig.add_axes([.73, .17, .235, .23])
    ax.set_axis_off()
    ax.text(0, 1.02, "Sampling support", fontsize=8, weight="bold")
    for i, (key, label) in enumerate([("optical_tissue_n", "Optical mask pixels"),
                                      ("orientation_valid_n", "Qualified directions"),
                                      ("local_order_supported_n", "Supported local centres"),
                                      ("multiscale_common_n", "Common scale centres"),
                                      ("PA_finite_n", "Native PA scan points")]):
        value = feature_value(table, "qc", key)
        ax.text(0, .83-i*.185, label, fontsize=6.1)
        ax.text(1, .83-i*.185, f"{value:,.0f}", ha="right", fontsize=6.6, color=TEAL)
    fig.text(.045, .065, "Threshold and boundary changes are sensitivity scenarios. The baseline mask and parameters define the reported features.", fontsize=6.2)
    fig.text(.045, .033, "Prediction curves show the spatial median and IQR. Counts are imaging units from a single plaque.", fontsize=6.2)
    export(fig, folder, "05_support_and_sensitivity", dpi)
