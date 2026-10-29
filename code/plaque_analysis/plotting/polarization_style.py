"""Figure 4: plotting style."""
from __future__ import annotations

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


TEAL = "#236C78"


ORANGE = "#BB6A38"


INK = "#25323B"


LIGHT = "#D8E5E8"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": .6, "lines.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "legend.frameon": False, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "savefig.facecolor": "white",
    })


def export(fig, folder, name, dpi=600, *, close=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if name == "Figure4_final" and (folder / "Figure4_final.svg").exists():
        print("  保留已排版的 Figure4_final.svg；跳过主图写入。", flush=True)
        if close:
            plt.close(fig)
        return
    # Arial lacks several Unicode subscript glyphs; MathText preserves proper subscripts.
    for text in fig.findobj(matplotlib.text.Text):
        value = text.get_text()
        for before, after in [("M₂", r"$M_2$"), ("M₄", r"$M_4$"),
                              ("C₂", r"$C_2$"), ("A₀", r"$A_0$")]:
            value = value.replace(before, after)
        text.set_text(value)
    # Fixed canvas dimensions preserve intended journal width and alignment.
    fig.savefig(folder/f"{name}.svg", dpi=dpi)
    fig.savefig(folder/f"{name}.png", dpi=dpi)
    if close:
        plt.close(fig)


def hide_figure_text(fig):
    """Hide labels on the existing canvas, retaining scale bars and all graphics."""
    # Materialize ticks before hiding them, and keep the original axes positions.
    fig.canvas.draw()
    for artist in fig.findobj(matplotlib.text.Text):
        if isinstance(artist, matplotlib.text.Annotation) and artist.arrow_patch is not None:
            # Annotation owns its arrow: hiding it would also remove the arrow.
            artist.set_text("")
        else:
            artist.set_visible(False)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
    # A title divider belongs to the heading, not to the scientific graphics.
    for artist in fig.findobj():
        if artist.get_gid() == "panel_title_divider":
            artist.set_visible(False)


def panel(fig, letter, title, x, y):
    fig.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom", color="black")
    fig.text(x+.024, y+.001, title, fontsize=8, va="bottom")


def scalebar(ax, extent, length, text, color="white"):
    width, height = extent
    x, y = .065*width, .90*height
    ax.plot([x, x+length], [y, y], color=color, lw=2, solid_capstyle="butt")
    ax.text(x+length/2, y-.025*height, text, ha="center", va="bottom", fontsize=6, color=color)


def map_panel(ax, image, mask, pixel_um, cmap, vmin, vmax, bar_um=2):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#11181C")
    h, w = image.shape
    extent = (w*pixel_um, h*pixel_um)
    im = ax.imshow(np.where(mask, image, np.nan), cmap=cm, vmin=vmin, vmax=vmax,
                   extent=[0, extent[0], extent[1], 0], interpolation="nearest")
    ax.axis("off")
    if bar_um:
        scalebar(ax, extent, bar_um, f"{bar_um:g} µm")
    return im


def hbar(fig, image, bounds, ticks, label, extend="neither"):
    cax = fig.add_axes(bounds)
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks, extend=extend)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6, length=2, pad=1)
    bar.set_label(label, fontsize=6, labelpad=1)
    return bar
