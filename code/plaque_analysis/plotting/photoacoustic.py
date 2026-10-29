"""Figure 4: plotting pa."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.offsetbox import HPacker
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import VPacker
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
from plaque_analysis.plotting.polarization_helpers import BACKGROUND, I_CENTRE_X, axes, colorbar, note, title
from plaque_analysis.plotting.polarization_style import INK, TEAL, scalebar


def draw_pa_distribution(fig, box, distribution):
    """Reference-style bars and annotations, with all heights tied to source data."""
    a = axes(fig, box, [I_CENTRE_X-.43, .10, .86, .23])
    a.set_gid("i_DOA_distribution")
    summary = distribution["summary"]
    edges, percent = distribution["edges"], distribution["percent"]
    width = summary["histogram_bin_width"]
    q1, median, q3 = summary["q25"], summary["median"], summary["q75"]
    ymax = max(20, np.ceil(float(percent.max())/.60/5)*5)
    band = a.axvspan(q1, q3, ymin=0, ymax=.97, color="#8BBAC2", alpha=.16, lw=0, zorder=0)
    band.set_gid("i_DOA_IQR_band")
    left = edges[:-1]+.035*width
    right = edges[1:]-.035*width
    polygons = np.array([[[lo, 0], [hi, 0], [hi, height], [lo, height]]
                         for lo, hi, height in zip(left, right, percent)])
    fill = a.imshow(np.linspace(0, 1, 256)[:, None], extent=[0, edges[-1], 0, percent.max()],
                     origin="lower", interpolation="bilinear", aspect="auto",
                     cmap=LinearSegmentedColormap.from_list("DOA_bar_fill", ["#B0CDD2", "#568E97"]),
                     vmin=0, vmax=1, alpha=.94, zorder=1)
    fill.set_clip_path(MplPath.make_compound_path_from_polys(polygons), a.transData)
    fill.set_gid("i_DOA_bar_gradient")
    bars = a.bar(left, percent, width=right-left, align="edge", facecolor="none",
                  edgecolor="#427B84", linewidth=.32, zorder=2)
    for j, bar in enumerate(bars):
        bar.set_gid(f"i_DOA_bar_{j+1}")
    curve, = a.plot(distribution["curve_x"], distribution["curve_percent_per_bin"],
                    color="#317681", linewidth=.8, zorder=3)
    curve.set_gid("i_DOA_density_guide")
    bracket_y = .66*ymax
    median_line, = a.plot([median, median], [0, bracket_y], color=INK,
                          linewidth=.85, linestyle=(0, (2, 1.2)), zorder=4)
    median_line.set_gid("i_DOA_median")
    bracket, = a.plot([q1, q3], [bracket_y]*2, color="#7AADB6", linewidth=.6, zorder=4)
    bracket.set_gid("i_DOA_IQR_bracket")
    for x in (q1, q3):
        a.plot([x, x], [bracket_y-.035*ymax, bracket_y+.035*ymax],
                color="#7AADB6", linewidth=.6, zorder=4)
    for label, value, x, y, loc in [
            ("Median", median, median, .985, "upper center"),
            ("IQR", q3-q1, q3+.003, .955, "upper left")]:
        content = VPacker(children=[
            TextArea(label, textprops={"color": TEAL, "size": 5.1}),
            TextArea(f"{value:.3f}", textprops={"color": TEAL, "size": 5.4,
                                               "weight": "bold" if label == "Median" else "normal"})],
            align="center" if label == "Median" else "left", pad=0, sep=.15)
        annotation = AnchoredOffsetbox(loc=loc, child=content, frameon=False, pad=0, borderpad=0,
                                       bbox_to_anchor=(x, y), bbox_transform=a.get_xaxis_transform())
        annotation.set_gid(f"i_DOA_{label}_label")
        a.add_artist(annotation)
    a.set_xlim(0, edges[-1]); a.set_ylim(0, ymax)
    a.set_xticks(np.linspace(0, edges[-1], 5))
    a.set_xticklabels([f"{v:.2f}" for v in np.linspace(0, edges[-1], 5)])
    a.set_yticks(np.linspace(0, ymax, 5))
    a.set_ylabel("PA cells (%)", fontsize=6, labelpad=2)
    a.set_xlabel("DOA", fontsize=6, labelpad=2)
    a.tick_params(labelsize=5.5, pad=1)
    chunks = [("Median ", INK, "normal"), (f"{median:.3f}", TEAL, "bold"),
              (" · IQR ", INK, "normal"), (f"{q3-q1:.3f}", TEAL, "bold"),
              (" · ", INK, "normal"), (str(summary["support_n"]), TEAL, "bold"),
              (" cells", INK, "normal")]
    footer = HPacker(children=[TextArea(t, textprops={"size": 5.5, "color": c, "weight": w})
                              for t, c, w in chunks], align="baseline", pad=0, sep=0)
    bx, by, bw, bh = box
    fig.add_artist(AnchoredOffsetbox(loc="lower center", child=footer, frameon=False,
                                     pad=0, borderpad=0, bbox_to_anchor=(bx+I_CENTRE_X*bw, by-.052*bh),
                                     bbox_transform=fig.transFigure))


def draw_i(fig, box, data):
    title(fig, box, "i", "PA anisotropy features")
    pa, visual, config = data["pa"], data["visual"], data["config"]
    dy, dx = config["pa_native_row_pitch_um"], config["pa_native_column_pitch_um"]
    if config["pa_display_rot90"] % 2: dy, dx = dx, dy
    a = axes(fig, box, [I_CENTRE_X-.44, .535, .88, .37])
    a.set_gid("i_PA_neighbour_map")
    cm = plt.get_cmap("inferno").copy(); cm.set_bad(BACKGROUND)
    vmax = max(.05, np.ceil(np.nanmax(visual["PA_neighbour_mean"])*100)/100)
    im = a.imshow(visual["PA_neighbour_mean"], cmap=cm, vmin=0, vmax=vmax, interpolation="nearest",
                  extent=[0, pa.shape[1]*dx, pa.shape[0]*dy, 0], aspect="equal")
    a.axis("off"); scalebar(a, (pa.shape[1]*dx, pa.shape[0]*dy),
                            config["main_scale_bar_um"], f"{config['main_scale_bar_um']:g} µm")
    note(fig, box, I_CENTRE_X, .921, "Mean neighbour |ΔDOA| from c", fontsize=6, ha="center")
    cb = colorbar(fig, box, im, [I_CENTRE_X-.40, .49, .80, .022], "", [0, vmax/2, vmax])
    cb.ax.set_gid("i_PA_neighbour_colourbar")
    mean_delta = np.mean([r["absolute_DOA_difference"] for r in visual["PA_pairs"]])
    note(fig, box, I_CENTRE_X, .398, f"Pair mean {mean_delta:.3f} · {len(visual['PA_pairs'])} pairs", fontsize=5.5, ha="center")
    note(fig, box, I_CENTRE_X, .344, "Original DOA distribution from c", fontsize=6, ha="center")
    draw_pa_distribution(fig, box, visual["PA_distribution"])
