"""Figure 4: plotting panels."""
from __future__ import annotations

from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from plaque_analysis.plotting.polarization_helpers import AXIAL_CMAP, BACKGROUND, DEPTH_COLOURS, D_CONTEXT_CMAP, axes, colorbar, note, raincloud_plot, title
from plaque_analysis.plotting.polarization_style import INK, map_panel, scalebar


def draw_top(fig, box, key, data):
    config, maps, tissue = data["config"], data["maps"], data["tissue"]
    titles = {"a": "Retardance proxy", "b": "Principal optic axis", "c": "Photoacoustic DOA"}
    title(fig, box, key, titles[key])
    width, height = config["approved_main_field_width_um"], config["approved_main_field_height_um"]
    a = axes(fig, box, [0, .14, .87, .78])
    if key == "c":
        # Final SVG embedded pixels, oriented by the source transform and using its display aspect.
        a.imshow(data["artwork"]["panels"]["c"], extent=[0, width, height, 0],
                 interpolation="nearest", aspect="equal")
        im = ScalarMappable(norm=Normalize(np.nanmin(data["pa"]), np.nanmax(data["pa"])),
                            cmap=ListedColormap(data["artwork"]["colour_ramps"]["c"]))
        colorbar(fig, box, im, [.91, .15, .032, .76], "", [.02, .06, .1, .14], "vertical")
    else:
        # The optical ROI is centred in a common black display frame without
        # changing its sampling or stretching the image. Extra margins are padding.
        optical_height = tissue.shape[0]*config["pixel_size_um"]
        pad_y = (height-optical_height)/2
        a.add_patch(Rectangle((0, 0), width, height, color=BACKGROUND, zorder=-2, lw=0))
        if key == "a":
            im = map_panel(a, maps["M2"], tissue, config["pixel_size_um"], "jet", 0, .2, bar_um=0)
            ticks, label = [0, .1, .2], r"$M_2$"
        else:
            im = map_panel(a, maps["phi_deg"], data["analysis"]["valid_orientation"],
                           config["pixel_size_um"], "hsv", 0, 180, bar_um=0)
            ticks, label = [0, 90, 180], "Axis (°)"
        im.set_extent([0, width, pad_y+optical_height, pad_y])
        colorbar(fig, box, im, [.91, .15, .032, .76], label, ticks, "vertical")
    a.set_xlim(0, width); a.set_ylim(height, 0); a.set_aspect("equal"); a.axis("off")
    # Identical data range, axes box and styling guarantee identical physical
    # bar lengths in a/b/c under the author's confirmed field calibration.
    length = config["main_scale_bar_um"]
    x, y = .065*width, .970*height
    a.plot([x, x+length], [y, y], color="white", lw=2, solid_capstyle="butt")
    a.text(x+length/2, y-.030*height, f"{length:g} µm", color="white", fontsize=6,
           ha="center", va="bottom")


def draw_d(fig, box, data):
    title(fig, box, "d", "Optic-axis organization")
    maps, tissue, config = data["maps"], data["tissue"], data["config"]
    h, w = tissue.shape; pixel = config["pixel_size_um"]
    a = axes(fig, box, [0, .19, .99, .75])
    # Style-only inheritance from Axer et al., Fig. 5: coloured axial rods over
    # a grey retardance map. Here M2 remains a proxy and all rods have equal
    # length; the reference's 3D inclination encoding is deliberately absent.
    a.imshow(np.where(tissue, maps["M2"], np.nan), cmap=D_CONTEXT_CMAP, vmin=0, vmax=.2,
             extent=[0, w*pixel, h*pixel, 0], interpolation="nearest", aspect="equal")
    segments, angles = [], []
    for glyph in data["visual"]["glyphs"]:
        x, y, angle = glyph["x_um"], glyph["y_um"], np.deg2rad(glyph["phi_deg"])
        dx, dy = glyph["length_um"]/2*np.cos(angle), -glyph["length_um"]/2*np.sin(angle)
        # Image rows increase downward; the minus sign preserves the existing
        # optical-angle convention when plotting Cartesian axial directions.
        segments.append([(x-dx, y-dy), (x+dx, y+dy)])
        angles.append(glyph["phi_deg"])
    rods = LineCollection(segments, cmap=AXIAL_CMAP, norm=Normalize(0, 180),
                           linewidths=.8, capstyle="round", zorder=3)
    rods.set_array(np.asarray(angles))
    a.add_collection(rods)
    a.set_xlim(0, w*pixel); a.set_ylim(h*pixel, 0)
    a.axis("off")
    scalebar(a, (w*pixel, h*pixel), config["main_scale_bar_um"],
              f"{config['main_scale_bar_um']:g} µm")
    colorbar(fig, box, rods, [.06, .143, .88, .025], "", [0, 45, 90, 135, 180])
    note(fig, box, .50, .045, "Local axis φ (°) · φ ≡ φ + 180°", fontsize=6, ha="center")
    note(fig, box, .50, -.008, f"Grey: M₂ · Local average: {data['analysis']['window_actual_um']:.2f} µm", fontsize=5.5, ha="center")


def draw_e(fig, box, data):
    title(fig, box, "e", "Spatial feature coordinates")
    tissue, analysis, config = data["tissue"], data["analysis"], data["config"]
    pixel = config["pixel_size_um"]; h, w = tissue.shape
    a = axes(fig, box, [0, .19, .99, .75])
    cm = ListedColormap(DEPTH_COLOURS); cm.set_bad("#F2F4F5")
    im = a.imshow(np.where(tissue, analysis["depth_labels"]+1, np.nan), cmap=cm,
                  norm=BoundaryNorm(np.arange(.5, 6.5), 5), interpolation="nearest",
                  extent=[0, w*pixel, h*pixel, 0])
    cy, cx = data["visual"]["spatial"]["centroid_row_col"]
    x, y = (cx+.5)*pixel, (cy+.5)*pixel
    yy, xx = np.indices(tissue.shape)
    a.contour((xx+.5)*pixel, (yy+.5)*pixel, tissue.astype(float), [.5], colors=[INK], linewidths=.4)
    radius = .37*min(h, w)*pixel
    for theta in np.linspace(0, 2*np.pi, config["spatial_sectors"], endpoint=False):
        a.plot([x, x+radius*np.cos(theta)], [y, y+radius*np.sin(theta)], color="white", lw=.75, ls=(0, (2, 1)))
        a.plot([x, x+radius*np.cos(theta)], [y, y+radius*np.sin(theta)], color=INK, lw=.3, ls=(0, (2, 1)))
    a.plot(x, y, "o", ms=2.5, mfc="white", mec=INK, mew=.5)
    a.text(x+radius+.1, y, "0°", fontsize=5.5, va="center")
    a.text(x, min(y+radius+.15, h*pixel-.2), "90°", fontsize=5.5, va="top", ha="center")
    a.axis("off")
    scalebar(a, (w*pixel, h*pixel), 2, "2 µm", color=INK)
    colorbar(fig, box, im, [.06, .143, .88, .030], "", [1, 2, 3, 4, 5])
    note(fig, box, .06, .043, "Rim", fontsize=6)
    note(fig, box, .94, .043, "Core", fontsize=6, ha="right")
    note(fig, box, .50, .043, "Depth region", fontsize=6, ha="center")
    note(fig, box, .50, -.008, "Position azimuth: clockwise from image right", fontsize=5.5, ha="center")


def draw_f(fig, box, data):
    title(fig, box, "f", "Regional optical distributions")
    visual = data["visual"]
    a = axes(fig, box, [.12, .57, .86, .34])
    a.set_gid("f_M2_distribution")
    raincloud_plot(a, visual["rainclouds"]["M2"], visual["median_trends"]["M2"], DEPTH_COLOURS, 100)
    a.set_ylim(0, visual["m2_hist_edges"][-1]*100)
    a.set_yticks([0, 8, 16]); a.set_xticks(range(1, 6), [])
    a.set_ylabel(r"$M_2$ (%)", fontsize=6, labelpad=2)
    b = axes(fig, box, [.12, .13, .86, .32])
    b.set_gid("f_angular_spread_distribution")
    raincloud_plot(b, visual["rainclouds"]["angular_spread_deg"],
                   visual["median_trends"]["angular_spread_deg"], DEPTH_COLOURS)
    view = visual["angular_spread_view"]
    upper = view["limits_deg"][1]
    b.set_ylim(view["limits_deg"])
    ticks = np.arange(0, upper+.001, 10) if upper >= 20 else np.linspace(0, upper, 4)
    b.set_yticks(np.unique(np.append(ticks, upper)))
    b.set_xticks(range(1, 6))
    b.set_ylabel("Angular spread (°)", fontsize=6, labelpad=2)
    b.set_xlabel("Depth region: rim → core", fontsize=6, labelpad=2)
    if view["above_display_max_n"]:
        note(fig, box, .98, .477,
             f"{view['above_display_max_n']:,} values > {upper:g}° ({view['above_display_max_percent']:.2f}%)",
             fontsize=5.1, ha="right", gid="f_spread_display_limit_note")
    ratio = data["analysis"]["values"]["optical_m2_core_rim_ratio"]
    note(fig, box, .14, -.014, "Box: IQR · whiskers: 1.5×IQR", fontsize=5.1)
    note(fig, box, .14, -.063, f"Curve: medians · core/rim {ratio:.2f}×", fontsize=5.1)


def draw_g(fig, box, data):
    title(fig, box, "g", "Depth–azimuth feature maps")
    spatial = data["visual"]["spatial"]
    for key, y, colour, lo, hi, label, ticks in [
        ("M2", .62, "viridis", 0, .10, r"Median $M_2$ (%)", [0, 5, 10]),
        ("S", .19, "magma", .5, 1, "Median local S", [.5, .75, 1])]:
        a = axes(fig, box, [.12, y, .77, .26])
        values = spatial[key]*(100 if key == "M2" else 1)
        vmin, vmax = (lo*100, hi*100) if key == "M2" else (lo, hi)
        cm = plt.get_cmap(colour).copy(); cm.set_bad("#E9ECEF")
        im = a.imshow(values, origin="lower", extent=[0, 360, .5, 5.5], aspect="auto",
                      cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        a.set_yticks([1, 3, 5]); a.set_xticks([0, 180, 360])
        a.tick_params(labelsize=5.5, length=2, pad=1)
        a.set_ylabel("Depth region", fontsize=6, labelpad=2)
        if key == "S": a.set_xlabel("Position azimuth (°)", fontsize=6, labelpad=2)
        else: a.set_xticklabels([])
        for r, c in np.argwhere(~np.isfinite(values)):
            a.text((c+.5)*360/values.shape[1], r+1, "×", ha="center", va="center", fontsize=6)
        note(fig, box, .12, y+.285, label, fontsize=6)
        below, above = np.any(values < vmin), np.any(values > vmax)
        extension = "both" if below and above else "min" if below else "max" if above else "neither"
        colorbar(fig, box, im, [.925, y, .032, .26], "", ticks, "vertical", extension)
    missing = sum(not r["local_S_displayed"] or not r["M2_displayed"] for r in spatial["rows"])
    note(fig, box, .12, .045, f"5 depth regions × 8 angular sectors", fontsize=5.7)
    if missing:
        note(fig, box, .12, -.010, f"× fewer than {data['config']['spatial_cell_min_pixels']} supported pixels", fontsize=5.5)
