"""Shared plotting conventions, applied only on explicit rendering."""

from plaque_analysis.settings.microenvironment import COLORS, CLASS_ORDER, GRID, INK


def configure():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": .55, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "figure7-modules", "figure.facecolor": "white",
        "savefig.facecolor": "white", "text.color": INK, "axes.labelcolor": INK})


def clean(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=.4)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)


def legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="none", color=COLORS[s], markersize=4) for s in CLASS_ORDER]
    fig.legend(handles, CLASS_ORDER, loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(.99, .998), fontsize=8)


def heading(container, letter, title=None):
    container.suptitle(letter if title is None else f"{letter}   {title}", x=.01,
                       ha="left", fontsize=10, fontweight="bold")
