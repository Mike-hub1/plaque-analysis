"""Eight modular panels and an automatically assembled reference page."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import plaque_analysis.plotting.composition as plot_composition; import plaque_analysis.plotting.genes as plot_genes; import plaque_analysis.plotting.glial as plot_glial; import plaque_analysis.plotting.program_maps as plot_spatial; import plaque_analysis.plotting.distance as plot_distance; import plaque_analysis.plotting.human as plot_human
from plaque_analysis.plotting.microenvironment_style import configure, legend


DRAWERS = {"a": plot_composition.draw, "b": plot_genes.draw, "c": plot_glial.draw,
           "d": plot_spatial.draw, "e": plot_distance.draw,
           "f": plot_human.draw_images, "g": plot_human.draw_structure, "h": plot_human.draw_lamp1}


def save_figure(fig, stem, dpi):
    for extension in ("svg", "png", "pdf"):
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(Path(stem).with_suffix("." + extension), dpi=dpi, metadata=metadata)
    plt.close(fig)


def render_panels(tables, output, dpi=300, panels="abcdefgh", assemble=True):
    configure()
    destination = Path(output) / "panels"
    destination.mkdir(parents=True, exist_ok=True)
    for letter in panels:
        width, height = {"a": (6.2, 3.5), "b": (5.4, 4.6), "c": (4.6, 3.4),
                         "d": (4.0, 5.0), "e": (4.6, 4.5), "f": (4.0, 7.0),
                         "g": (5.4, max(4.0, len(tables["panel_g_human_structure"]) * .18 + 1.2)),
                         "h": (4.8, max(4.0, len(tables["panel_g_human_structure"]) * .18 + 1.2))}[letter]
        fig = plt.figure(figsize=(width, height), layout="constrained")
        DRAWERS[letter](fig, tables)
        save_figure(fig, destination / f"panel_{letter}", dpi)
    if assemble:
        render_preview(destination, Path(output) / "Figure7_overview.png", dpi)


def render_preview(panel_dir, destination, dpi):
    """Make a raster overview; editable panel SVGs remain separate."""
    fig = plt.figure(figsize=(10.5, 14.85), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=[1., 1.15, 1.65])
    spans = [(0, 0, 3), (0, 3, 6), (1, 0, 2), (1, 2, 4), (1, 4, 6),
             (2, 0, 2), (2, 2, 4), (2, 4, 6)]
    for letter, (row, first, last) in zip("abcdefgh", spans):
        ax = fig.add_subplot(grid[row, first:last])
        with Image.open(Path(panel_dir) / f"panel_{letter}.png") as image:
            ax.imshow(image.convert("RGB"))
        ax.set_axis_off()
    legend(fig)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)
