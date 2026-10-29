"""Figure 3: plotting."""
from __future__ import annotations

from plaque_analysis.plotting.abeta import render_abeta_plate
from plaque_analysis.plotting.feature_atlas import render_feature_atlas
from plaque_analysis.plotting.fluorescence_data import load_saved_visual_data, prepare_visual_data
from plaque_analysis.plotting.fluorescence_guides import write_chinese_figure_guides, write_figures_index
from plaque_analysis.plotting.fluorescence_helpers import add_scale_bar
from plaque_analysis.plotting.lamp1 import render_lamp1_plate
from plaque_analysis.plotting.fluorescence_layout import figure_axis, figure_bundle_base, figure_text
from plaque_analysis.plotting.fluorescence_overview import render_main_plate
from plaque_analysis.plotting.fluorescence_publication import render_publication_figure
from plaque_analysis.plotting.colocalization import render_relationship_plate
from plaque_analysis.plotting.fluorescence_style import configure_plotting, save_figure
import numpy as np


def render_figures(raw, signed, masks, valid, geometry, profiles, angular, relations_qc,
                   features, outcomes, ranks, stratum_rows, saturation, spacing,
                   config, out_dir, plaque_id, dpi=600, *, preserve_source_data=False):
    """Presentation layer only: nine complementary plates with traceable source data."""
    plt = configure_plotting()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig_dir = out_dir/"figures"
    fig_dir.mkdir(exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in plaque_id)
    plaque_dir = fig_dir/safe_id
    plaque_dir.mkdir(exist_ok=True)
    source_dir = out_dir/"plot_source_data"/safe_id
    source_dir.mkdir(parents=True, exist_ok=True)
    data = (load_saved_visual_data(out_dir, plaque_id, geometry["expected"], masks[1]) if preserve_source_data else
            prepare_visual_data(raw, signed, masks, valid, geometry, profiles, spacing, config, out_dir, plaque_id))
    data["strata"] = stratum_rows
    if not preserve_source_data:
        for key, suffix in (("pixel", "pixels"), ("rings", "radial_summary"),
                            ("cells", "radial_angular"), ("patches", "lamp1_patches")):
            data[key].to_csv(source_dir/f"{suffix}.csv", index=False)
    base = figure_bundle_base(plaque_dir, "boundary_phenotyping")
    render_main_plate(plt, data, raw, signed, masks, geometry, saturation, spacing, config, base, dpi)
    render_abeta_plate(plt, data, signed, masks, geometry, saturation, spacing, config,
                       figure_bundle_base(plaque_dir, "abeta_organization"), dpi)
    render_lamp1_plate(plt, data, masks, angular, spacing, config,
                       figure_bundle_base(plaque_dir, "lamp1_spatial_architecture"), dpi)
    render_relationship_plate(plt, data, raw, signed, masks, saturation, spacing,
                             figure_bundle_base(plaque_dir, "channel_relationships"), dpi)
    for role in ("X_A", "Y_L", "Z_AL"):
        render_feature_atlas(plt, data, role, figure_bundle_base(plaque_dir, f"feature_atlas_{role}"), dpi, source_dir,
                             preserve_source_data=preserve_source_data)
    for ci, (tag, color_map) in enumerate((("blue_Aβ", "Blues"), ("red_LAMP1", "Oranges"))):
        rows = stratum_rows[ci]
        k = max(len(rows), 1)
        colors = plt.get_cmap(color_map)(np.linspace(.25, .95, k))
        cmap = ListedColormap(colors); cmap.set_bad("#14141A")
        fig = plt.figure(figsize=(100/25.4, 85/25.4))
        ax = figure_axis(fig, 3, 17, 70, 57)
        image = np.ma.masked_where(ranks[ci] < 1, ranks[ci])
        sy, sx = spacing; h, w = image.shape
        im = ax.imshow(image, cmap=cmap, norm=BoundaryNorm(np.arange(.5, k+1.5), k),
                       interpolation="nearest", extent=(-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2))
        ax.axis("off"); add_scale_bar(ax, raw[ci].shape, spacing)
        title = "Aβ intensity strata" if ci == 0 else "LAMP1 strata in the fixed window"
        figure_text(fig, 3, 80, title, fontsize=8, weight="bold")
        cb = fig.colorbar(im, cax=figure_axis(fig, 77, 27, 2, 35), ticks=np.arange(1, k+1))
        cb.ax.invert_yaxis()  # Rank 1 is brightest: put the highest signal at the top of the key.
        if rows:
            cb.ax.set_yticklabels([f"{r['rank']}: {r['min']:.3g}–{r['max']:.3g}" for r in rows])
        cb.ax.tick_params(labelsize=5.8, length=2, pad=2); cb.outline.set_linewidth(.4)
        figure_text(fig, 75, 70, "Rank: signal range", fontsize=5.6)
        figure_text(fig, 75, 66, "(corrected a.u.)", fontsize=5.6)
        figure_text(fig, 3, 11, "Rank 1 = brightest; intensity ranks are not biological subtypes.", fontsize=5.8)
        figure_text(fig, 3, 5, "Ceiling-affected: diagnostic only." if saturation[ci]["mask"].any()
                    else "Native pixels; no interpolation or additional spatial information.", fontsize=5.8)
        save_figure(fig, figure_bundle_base(plaque_dir, f"{tag}_intensity_strata"), dpi)
    paper_base = figure_bundle_base(plaque_dir, "manuscript_figure")
    render_publication_figure(plt, data, raw, signed, masks, valid, geometry, saturation,
        spacing, config, ranks, stratum_rows, paper_base, dpi)
    render_publication_figure(plt, data, raw, signed, masks, valid, geometry, saturation,
        spacing, config, ranks, stratum_rows, paper_base.parent/"A4校样", dpi, a4=True)
    write_chinese_figure_guides(data, raw, saturation, spacing, config, plaque_dir, dpi)
    write_figures_index(fig_dir)
    return str(base)+".png"

