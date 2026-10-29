"""Figure 4: plotting scale."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import numpy as np
from plaque_analysis.plotting.polarization_helpers import axes, colorbar, note, title
from plaque_analysis.plotting.polarization_style import INK, TEAL


def scale_frame_projection(index):
    """Common oblique projection: image x slopes down-right, image y is up."""
    return Affine2D.from_values(.78, -.145, 0, .43, index+1-.39, .08+.075*index)


def draw_h(fig, box, data):
    title(fig, box, "h", "Orientation across scales")
    bx, by, bw, bh = box
    fig.add_artist(Line2D([bx+.062*bw]*2, [by+.992*bh, by+1.043*bh],
                         transform=fig.transFigure, color=INK, linewidth=.35,
                         gid="panel_title_divider"))
    visual, analysis, config = data["visual"], data["analysis"], data["config"]
    count = len(visual["scale_maps"])
    positions = np.arange(count)+1
    limits = (.38, count+.62)
    frames = axes(fig, box, [.105, .515, .785, .440])
    frames.set_gid("h_scale_frames")
    frames.set_xlim(limits); frames.set_ylim(-.21, .98); frames.axis("off")
    x0, y0, x1, y1 = visual["scale_frame_crop_xyxy"]
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    corners = [scale_frame_projection(j).transform(unit) for j in range(count)]
    floor = np.array([corners[0][0], corners[0][1], corners[-1][1], corners[-1][0]])
    ceiling = np.array([corners[0][3], corners[0][2], corners[-1][2], corners[-1][3]])
    rear = np.array([corners[0][0], corners[0][3], corners[-1][3], corners[-1][0]])
    # Walls and depth shading are display context only; measured S pixels remain opaque.
    for offset, alpha in [(.035, .032), (.055, .020), (.08, .010)]:
        frames.add_patch(Polygon(floor+[0, -offset], closed=True, facecolor="#637A89",
                                 edgecolor="none", alpha=alpha, zorder=-4))
    for surface, colour, alpha in [(floor, "#E2E9ED", .24), (rear, "#DBE5EB", .22),
                                   (ceiling, "#E4ECF1", .18)]:
        frames.add_patch(Polygon(surface, closed=True, facecolor=colour,
                                 edgecolor="#BCCCD5", linewidth=.4, alpha=alpha, zorder=-2))
    u, v = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    context = LinearSegmentedColormap.from_list("plane_context", ["#A3B5C1", "#102A3B"])
    backdrop = context(.18+.68*v+.12*u)
    colour = plt.get_cmap("magma").copy(); colour.set_bad((0, 0, 0, 0))
    for j in reversed(range(count)):
        transform = scale_frame_projection(j)
        frames.imshow(backdrop, origin="lower", extent=[0, 1, 0, 1],
                       interpolation="bilinear", aspect="auto", alpha=.86,
                       transform=transform+frames.transData, zorder=1+(count-j)*.1)
        im = frames.imshow(visual["scale_maps"][j, y0:y1, x0:x1], cmap=colour,
                           vmin=0, vmax=1, extent=[0, 1, 0, 1], origin="upper",
                           interpolation="nearest", aspect="auto",
                           transform=transform+frames.transData, zorder=2+(count-j)*.1)
        im.set_gid(f"h_scale_frame_{j+1}")
        frames.add_patch(Polygon(corners[j], closed=True, fill=False,
                                  edgecolor="#D9E3E9", linewidth=.45, zorder=4))
        frames.text(j+1, corners[j][:, 1].max()+.024,
                     f"{analysis['scale_rows'][j]['window_um']:.2f}",
                     color=INK, fontsize=5.5, ha="center", va="bottom", zorder=5)
        if j == 0:
            length = config["main_scale_bar_um"]/((x1-x0)*config["pixel_size_um"])
            xy = transform.transform([[.12, .065], [.12+length, .065]])
            bar, = frames.plot(xy[:, 0], xy[:, 1], color="white", lw=1.3,
                                solid_capstyle="butt", zorder=6)
            bar.set_gid("h_scale_bar")
            endpoints = frames.transData.transform(xy)
            bar_angle = np.degrees(np.arctan2(*(endpoints[1]-endpoints[0])[::-1]))
            p = transform.transform((.12+length/2, .082))
            label = frames.text(*p, f"{config['main_scale_bar_um']:g} µm", fontsize=5.1,
                                color="white", ha="center", va="bottom", rotation=bar_angle,
                                rotation_mode="anchor", zorder=6)
            label.set_gid("h_scale_bar_label")
    for edge in range(4):
        line = np.array([corners[0][edge], corners[-1][edge]])
        # Rear rails are hidden by the opaque image planes; front glass edges
        # are a light outline so they do not dominate the measured maps.
        frames.plot(line[:, 0], line[:, 1], color="#CDDCE4", linewidth=.3, alpha=.8,
                     zorder=5 if edge in (1, 2) else 0)
    start, end = corners[0][1]+[.22, -.075], corners[-1][1]+[.10, -.075]
    frames.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=INK, lw=.65),
                     zorder=6)
    endpoints = frames.transData.transform([start, end])
    angle = np.degrees(np.arctan2(*(endpoints[1]-endpoints[0])[::-1]))
    midpoint = (start+end)/2+[0, -.022]
    frames.text(*midpoint, "Window width (µm)", fontsize=5.3, rotation=angle,
                 rotation_mode="anchor", ha="center", va="top", color=INK)
    # A shared, unshortened origin keeps the tiny coordinate axes connected.
    # The x arm is parallel to the image plane's projected horizontal edge.
    origin = np.array([.035, .075])
    for label, vector, offset, ha, va in [
            ("x", [.31, -.31*.145/.78], (1.5, -.5), "left", "center"),
            ("y", [0, .12], (0, 1.5), "center", "bottom")]:
        tip = origin+vector
        arrow = FancyArrowPatch(origin, tip, transform=frames.transData,
                                arrowstyle="-|>", mutation_scale=5.2, linewidth=.6,
                                shrinkA=0, shrinkB=0, color=INK, clip_on=False,
                                capstyle="round", joinstyle="round", zorder=7)
        arrow.set_gid(f"h_coordinate_axis_{label}")
        frames.add_patch(arrow)
        text = frames.annotate(label, xy=tip, xytext=offset, textcoords="offset points",
                               fontsize=5.3, fontstyle="italic", color=INK, ha=ha, va=va,
                               annotation_clip=False, zorder=7)
        text.set_gid(f"h_coordinate_label_{label}")
    frames.set_xlim(limits); frames.set_ylim(-.21, .98)
    # Centre the shared horizontal colourbar on both the frame stack and the
    # distribution, without changing their common horizontal category axis.
    cb = colorbar(fig, box, im, [.2575, .443, .480, .020], "", [0, .5, 1], "horizontal")
    cb.ax.set_gid("h_shared_colourbar")
    cb.ax.set_xticklabels(["0.0", "0.5", "1.0"])
    note(fig, box, .4975, .478, "Local order S", fontsize=5.1, ha="center", va="bottom",
         gid="h_shared_colourbar_label", bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    a = axes(fig, box, [.105, .13, .785, .245])
    a.set_gid("h_scale_distribution"); a.patch.set_visible(False)
    rows = analysis["scale_rows"]
    median = np.array([r["median"] for r in rows])
    q25, q75 = np.array([r["q25"] for r in rows]), np.array([r["q75"] for r in rows])
    q05, q95 = np.array([r["q05"] for r in rows]), np.array([r["q95"] for r in rows])
    band = a.fill_between(positions, q25, q75, color="#6BACBA", alpha=.22, linewidth=0, zorder=1)
    band.set_gid("h_full_population_IQR")
    for j in range(count):
        dots = [r for r in visual["scale_scatter"] if r["frame"] == j+1]
        artist = a.scatter([r["x_category"] for r in dots], [r["local_S"] for r in dots],
                            s=1.65, color="#73B4C2", alpha=.60, linewidths=0, zorder=2)
        artist.set_gid(f"h_display_scatter_{j+1}")
    line, = a.plot(positions, median, color=TEAL, linewidth=.85, zorder=3)
    line.set_gid("h_full_population_medians")
    error = a.errorbar(positions, median, yerr=[median-q05, q95-median], fmt="o",
                       mfc="white", mec=INK, mew=.85, ms=3.1, ecolor=INK,
                       elinewidth=.9, capsize=1.65, capthick=.75, zorder=4)
    error.lines[0].set_gid("h_median_markers")
    for j in range(count):
        lower = scale_frame_projection(j).transform((.5, 0))
        guide = ConnectionPatch(xyA=lower, coordsA=frames.transData,
                                 xyB=(j+1, 0), coordsB=a.transData,
                                 color="#A7BDC8", linewidth=.45, ls=(0, (2, 2)), zorder=-.5)
        guide.set_gid(f"h_alignment_guide_{j+1}")
        fig.add_artist(guide)
    a.set_xlim(limits); a.set_ylim(0, 1.055); a.set_yticks([0, .5, 1])
    a.set_xticks(positions, [f"{r['window_um']:.2f}" for r in rows], fontsize=5.1)
    a.tick_params(labelsize=5.5, pad=1)
    a.set_ylabel("Local order S", fontsize=6, labelpad=2)
    a.set_xlabel("Window width (µm)", fontsize=6, labelpad=2)
    note(fig, box, .105, -.006, "Median · band: IQR · bars: 5–95%", fontsize=5.1)
    shown = len(visual["scale_scatter"])//count
    note(fig, box, .105, -.052, f"{analysis['scale_common_mask'].sum():,} shared centres · {shown} dots/scale", fontsize=5.1)
