"""Figure 4: plotting main."""
from __future__ import annotations

from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.pyplot as plt
from plaque_analysis.plotting.photoacoustic import draw_i
from plaque_analysis.plotting.polarization_panels import draw_d, draw_e, draw_f, draw_g, draw_top
from plaque_analysis.plotting.polarization_scale import draw_h
from plaque_analysis.plotting.polarization_style import export, hide_figure_text, style


DRAW = dict(zip("defghi", [draw_d, draw_e, draw_f, draw_g, draw_h, draw_i]))


def plot_visual_story(folder, maps, tissue, analysis, pa_display, config, visual, artwork):
    style()
    folder = Path(folder)
    data = {"maps": maps, "tissue": tissue, "analysis": analysis, "pa": pa_display,
            "config": config, "visual": visual, "artwork": artwork}
    fig = plt.figure(figsize=(config["figure_width_mm"]/25.4, 204/25.4))
    columns = [.036, .367, .698]
    rows = [.686, .381, .058]
    boxes = {key: (columns[k%3], rows[k//3], .272, .266) for k, key in enumerate("abcdefghi")}
    for y, label in [(.978, "MEASURED PARAMETERS"), (.674, "SPATIAL FEATURE EXTRACTION"),
                      (.356, "SPATIAL AND SCALE DESCRIPTORS")]:
        fig.text(.036, y, label, fontsize=6, color="#6A7983", va="center")
        start = .34 if "SPATIAL" in label else .285
        fig.add_artist(Line2D([start, .975], [y, y], transform=fig.transFigure,
                             color="#CED6DB", lw=.45))
    for key in "abc": draw_top(fig, boxes[key], key, data)
    for key, draw in DRAW.items(): draw(fig, boxes[key], data)
    fig.text(.036, .014,
             "One plaque · Panel c retains the final SVG image and the author-confirmed field calibration · Spatial distributions are descriptive",
             fontsize=5.3, color="#697780")
    export(fig, folder, "Figure4_regenerated", config["raster_dpi"])
    # Every panel has a labelled and text-free copy on exactly the same canvas.
    panel_folder = folder/"panels"
    no_text_folder = folder/"panels_no_text"
    for key in "abcdefghi":
        size, panel_box = ((89, 82), (.035, .075, .92, .84)) if key == "h" else ((61, 65), (.115, .10, .82, .825))
        fig = plt.figure(figsize=(size[0]/25.4, size[1]/25.4))
        if key in "abc":
            draw_top(fig, panel_box, key, data)
        else:
            DRAW[key](fig, panel_box, data)
        if key == "h":
            for text in fig.findobj(plt.Text):
                if text.get_gid() != "h_scale_bar_label":
                    text.set_fontsize(text.get_fontsize()*1.25)
        export(fig, panel_folder, f"panel_{key}", config["raster_dpi"], close=False)
        hide_figure_text(fig)
        export(fig, no_text_folder, f"panel_{key}_no_text", config["raster_dpi"])
