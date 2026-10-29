"""Figure 4: plotting helpers."""
from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Polygon
import numpy as np


DEPTH_COLOURS = ["#A8DFE0", "#86CBCD", "#62A9C8", "#6C8CBF", "#33539E"]


BACKGROUND = "#10141A"


I_CENTRE_X = .55


AXIAL_CMAP = ListedColormap(hsv_to_rgb(np.column_stack([
    np.linspace(0, 1, 361), np.full(361, .60), np.ones(361)])), name="axial_hue")


D_CONTEXT_CMAP = LinearSegmentedColormap.from_list("M2_context_grey", ["#181818", "#8A8A8A"])


D_CONTEXT_CMAP.set_bad(BACKGROUND)


def axes(fig, box, relative):
    x, y, w, h = box
    a, b, c, d = relative
    return fig.add_axes([x+a*w, y+b*h, c*w, d*h])


def note(fig, box, x, y, text, **kwargs):
    bx, by, bw, bh = box
    return fig.text(bx+x*bw, by+y*bh, text, **kwargs)


def title(fig, box, letter, text):
    note(fig, box, 0, .993, letter, fontsize=8, fontweight="bold", va="bottom", color="black")
    note(fig, box, .092, .999, text, fontsize=7, va="bottom")


def colorbar(fig, box, im, relative, label, ticks, orientation="horizontal", extend="neither"):
    cb = fig.colorbar(im, cax=axes(fig, box, relative), orientation=orientation,
                      ticks=ticks, extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.set_label(label, fontsize=6, labelpad=2)
    return cb


def raincloud_plot(ax, groups, trend, colours, factor=1):
    """Reference layout: left rain, central median-pinched box, right density."""
    for k, (group, colour) in enumerate(zip(groups, colours), 1):
        stat = group["summary"]
        yy = group["density_x"]*factor
        density = group["density"]
        width = .32*density/density.max() if density.max() else density
        cloud = ax.fill_betweenx(yy, k, k+width, facecolor=colour, edgecolor=colour,
                                  linewidth=.55, alpha=.26, zorder=1)
        cloud.set_gid(f"f_{stat['family']}_cloud_{k}")
        q1, med, q3 = np.array([stat["q25"], stat["median"], stat["q75"]])*factor
        lo, hi = np.array([stat["whisker_low"], stat["whisker_high"]])*factor
        # The waist is a median glyph, not a sampling-based confidence interval.
        pinch = min(.10*(q3-q1), med-q1, q3-med)
        verts = [(k-.095, q1), (k+.095, q1), (k+.095, med-pinch),
                 (k+.048, med), (k+.095, med+pinch), (k+.095, q3),
                 (k-.095, q3), (k-.095, med+pinch), (k-.048, med), (k-.095, med-pinch)]
        box = Polygon(verts, closed=True, facecolor=colour, edgecolor="none", alpha=.88, zorder=4)
        box.set_gid(f"f_{stat['family']}_IQR_box_{k}"); ax.add_patch(box)
        whisker, = ax.plot([k, k], [lo, hi], color=colour, lw=1.05, zorder=3)
        whisker.set_gid(f"f_{stat['family']}_whisker_{k}")
        ax.hlines([lo, hi], k-.052, k+.052, color=colour, lw=1.0, zorder=4)
        line, = ax.plot([k-.048, k+.048], [med, med], color="#152A35", lw=1.15, zorder=6)
        line.set_gid(f"f_{stat['family']}_median_{k}")
        rain = ax.scatter([r["x_category"] for r in group["scatter"]],
                           [r["plotted_value"] for r in group["scatter"]],
                           s=3.4, facecolor=colour, edgecolor="white", linewidth=.24,
                           alpha=.82, zorder=5)
        rain.set_gid(f"f_{stat['family']}_rain_{k}")
    # Shape-preserving interpolation only joins the five measured medians.
    xy = np.column_stack([trend["x_category"], trend["median_guide"]*factor])
    curve_colours = LinearSegmentedColormap.from_list("depth_median_guide", colours)
    curve = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1),
                           cmap=curve_colours, norm=Normalize(1, len(groups)),
                           linewidth=1.05, alpha=.95, zorder=2.5, capstyle="round")
    curve.set_array((xy[:-1, 0]+xy[1:, 0])/2)
    curve.set_gid(f"f_{groups[0]['summary']['family']}_median_curve")
    ax.add_collection(curve)
    ax.set_xlim(.5, len(groups)+.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#BAC4CA", linestyle=(0, (2, 2)), linewidth=.35, alpha=.7)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    for spine in ax.spines.values(): spine.set_linewidth(.55)
    ax.tick_params(labelsize=5.5, pad=1, length=2)
