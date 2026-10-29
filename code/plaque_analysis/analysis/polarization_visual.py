"""Figure 4: visual features."""
from __future__ import annotations

from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde
import numpy as np
from plaque_analysis.imaging.orientation import distribution, local_axial
from plaque_analysis.imaging.photoacoustic import pa_doa_distribution, pa_neighbours
from plaque_analysis.imaging.reconstruction import odd_window


def spatial_cells(m2, local_s, tissue, depth_labels, sectors=8, min_n=50):
    """Geometry-defined depth/azimuth partition, independent of signal magnitude."""
    yy, xx = np.indices(tissue.shape)
    cy, cx = np.argwhere(tissue).mean(axis=0)
    # Position azimuth follows Figure 3: zero to image right, clockwise positive.
    azimuth = np.degrees(np.arctan2(yy-cy, xx-cx)) % 360
    sector = np.floor(azimuth/(360/sectors)).astype(int)
    n_depth = int(depth_labels[tissue].max())+1
    amplitude = np.full((n_depth, sectors), np.nan)
    order = np.full_like(amplitude, np.nan)
    rows = []
    for depth in range(n_depth):
        for angle in range(sectors):
            mask = tissue & (depth_labels == depth) & (sector == angle)
            a, s = m2[mask], local_s[mask]
            a, s = a[np.isfinite(a)], s[np.isfinite(s)]
            am = float(np.median(a)) if a.size else np.nan
            sm = float(np.median(s)) if s.size else np.nan
            if a.size >= min_n:
                amplitude[depth, angle] = am
            if s.size >= min_n:
                order[depth, angle] = sm
            rows.append({"depth_region": depth+1, "azimuth_sector": angle+1,
                         "azimuth_start_deg": angle*360/sectors,
                         "azimuth_end_deg": (angle+1)*360/sectors,
                         "tissue_n": int(mask.sum()), "M2_n": int(a.size),
                         "local_S_n": int(s.size), "M2_median_unmasked": am,
                         "local_S_median_unmasked": sm, "M2_displayed": a.size >= min_n,
                         "local_S_displayed": s.size >= min_n})
    return {"M2": amplitude, "S": order, "rows": rows, "sector_map": sector,
            "azimuth_deg": azimuth, "centroid_row_col": [cy, cx]}


def histogram_rows(values, edges, family, group):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    counts, _ = np.histogram(a, edges)
    if counts.sum() != a.size:
        raise ValueError(f"Histogram range would omit {family}/{group} observations.")
    density = counts/(max(1, a.size)*np.diff(edges))
    rows = [{"family": family, "group": group, "bin_left": float(lo),
             "bin_right": float(hi), "count": int(n), "density": float(d),
             "support_n": int(a.size)}
            for lo, hi, n, d in zip(edges[:-1], edges[1:], counts, density)]
    return rows


def raincloud_data(values, native_rows, native_cols, family, region, config):
    """Full-population density/box statistics and traceable display-only samples."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raincloud groups require finite, nonnegative observations.")
    if len(values) != len(native_rows) or len(values) != len(native_cols):
        raise ValueError("Raincloud source positions do not match their observations.")
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    iqr = q3-q1
    inside = values[(values >= q1-1.5*iqr) & (values <= q3+1.5*iqr)]
    low, high = float(inside.min()), float(inside.max())
    grid = np.linspace(float(values.min()), float(values.max()), 512)
    if len(values) > 1 and values.max() > values.min():
        kde = gaussian_kde(values, bw_method="scott")
        density = kde(grid)+kde(-grid)
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    else:
        density = np.zeros_like(grid)
        bandwidth = 0.0
    family_index = 0 if family == "M2" else 1
    rng = np.random.default_rng(config["f_scatter_seed"]+100*family_index+region)
    selected = np.sort(rng.choice(len(values), size=min(config["f_scatter_count"], len(values)),
                                  replace=False))
    # Both variables retain their own valid populations. Only x is jittered.
    x = region-.17-np.abs(rng.normal(0, .068, len(selected)))
    x = np.clip(x, region-.38, region-.17)
    factor = 100 if family == "M2" else 1
    scatter = [{"family": family, "depth_region": region, "display_id": j+1,
                "population_index": int(index), "roi_row": int(native_rows[index]),
                "roi_column": int(native_cols[index]), "raw_value": float(values[index]),
                "plotted_value": float(values[index]*factor), "x_category": float(x[j])}
               for j, index in enumerate(selected)]
    summary = {"family": family, "depth_region": region, "n": len(values),
               "q25": float(q1), "median": float(median), "q75": float(q3),
               "IQR": float(iqr), "whisker_low": low, "whisker_high": high,
               "observed_min": float(values.min()), "observed_max": float(values.max()),
               "outside_whiskers_n": int(np.sum((values < low) | (values > high))),
               "KDE_bandwidth_raw_units": bandwidth, "display_n": len(selected),
               "display_seed": config["f_scatter_seed"]+100*family_index+region,
               "box_notch_is_confidence_interval": False}
    return {"summary": summary, "density_x": grid, "density": density, "scatter": scatter}


def extract_visual_features(maps, tissue, analysis, pa_display, config):
    pixel = config["pixel_size_um"]
    spatial = spatial_cells(maps["M2"], analysis["S_local"], tissue,
                            analysis["depth_labels"], config["spatial_sectors"],
                            config["spatial_cell_min_pixels"])
    depth_groups, summaries, histograms = [], [], []
    rainclouds = {"M2": [], "angular_spread_deg": []}
    # Histogram-only display aggregation: every observation is counted exactly once.
    m2_upper = max(.16, np.ceil(np.max(maps["M2"][tissue])*100)/100)
    m2_edges = np.linspace(0, m2_upper, 65)
    s_edges = np.linspace(0, 1, 201)
    spread_upper = max(70, np.ceil(np.nanmax(analysis["local_angular_spread_deg"])/10)*10)
    spread_edges = np.linspace(0, spread_upper, 71)
    for k in range(config["depth_quantiles"]):
        valid = tissue & (analysis["depth_labels"] == k)
        a, s = maps["M2"][valid], analysis["S_local"][valid]
        s = s[np.isfinite(s)]
        spread = analysis["local_angular_spread_deg"][valid]
        spread = spread[np.isfinite(spread)]
        depth_groups.append({"M2": a, "S": s, "spread": spread})
        for family, data, edges in [("M2", a, m2_edges), ("local_S", s, s_edges),
                                    ("angular_spread_deg", spread, spread_edges)]:
            summaries.append({"depth_region": k+1, "family": family, **distribution(data)})
            histograms.extend(histogram_rows(data, edges, family, k+1))
            if family in rainclouds:
                source_map = maps["M2"] if family == "M2" else analysis["local_angular_spread_deg"]
                rows, cols = np.where(valid & np.isfinite(source_map))
                rainclouds[family].append(raincloud_data(data, rows, cols, family, k+1, config))
    median_trends = {}
    for family, groups in rainclouds.items():
        x = np.arange(1, len(groups)+1, dtype=float)
        y = np.array([g["summary"]["median"] for g in groups])
        trend_x = np.linspace(x[0], x[-1], 401)
        median_trends[family] = {"x_category": trend_x, "median_guide": PchipInterpolator(x, y)(trend_x)}
    # The lower f axis is a display zoom only. Keep all observations in the
    # density, box statistics, median guide and previously exported sources.
    spread_limit = config["f_spread_display_max_deg"]
    spread_view_groups = [
        {"depth_region": k+1, "n": len(group["spread"]),
         "above_display_max_n": int(np.sum(group["spread"] > spread_limit)),
         "observed_max_deg": float(np.max(group["spread"]))}
        for k, group in enumerate(depth_groups)]
    spread_n = sum(row["n"] for row in spread_view_groups)
    spread_above = sum(row["above_display_max_n"] for row in spread_view_groups)
    spread_view = {"limits_deg": [0, spread_limit], "n": spread_n,
                   "above_display_max_n": spread_above,
                   "above_display_max_percent": 100*spread_above/spread_n,
                   "affects_statistics": False, "groups": spread_view_groups}
    scale_maps, scale_values, scale_histograms = [], [], []
    for requested in config["scale_windows_um"]:
        w = odd_window(requested, pixel)
        s, _, _ = local_axial(maps["phi_deg"], analysis["valid_orientation"], w,
                              config["minimum_window_coverage"])
        keep = analysis["scale_common_mask"]
        if not np.isfinite(s[keep]).all():
            raise ValueError("Multiscale display support differs from numerical support.")
        s = np.where(keep, s, np.nan)
        scale_maps.append(s)
        scale_values.append(s[keep])
        scale_histograms.extend(histogram_rows(s[keep], s_edges, "local_S", w*pixel))
    neighbour, neighbour_n, pairs = pa_neighbours(pa_display)
    # Remove only unused display margins, preserving every common centre.
    yy, xx = np.where(analysis["scale_common_mask"])
    if not yy.size:
        raise ValueError("Multiscale frames require nonempty common support.")
    pad = 12
    scale_crop = [max(0, int(xx.min())-pad), max(0, int(yy.min())-pad),
                  min(tissue.shape[1], int(xx.max())+1+pad),
                  min(tissue.shape[0], int(yy.max())+1+pad)]
    # Author-authorized display sampling. Quantiles still use the entire common
    # population; reusing the same centres preserves pairing across scales.
    rng = np.random.default_rng(config["h_scatter_seed"])
    selected = np.sort(rng.choice(yy.size, size=min(config["h_scatter_count"], yy.size), replace=False))
    jitter = rng.uniform(-.17, .17, len(selected))
    scatter = [
        {"frame": j+1, "window_um": analysis["scale_rows"][j]["window_um"],
         "display_id": k+1, "common_center_index": int(index),
         "roi_row": int(yy[index]), "roi_column": int(xx[index]),
         "x_category": float(j+1+jitter[k]), "jitter_category": float(jitter[k]),
         "local_S": float(s[yy[index], xx[index]])}
        for j, s in enumerate(scale_maps) for k, index in enumerate(selected)]
    glyphs = []
    step = max(1, round(config["direction_spacing_um"]/pixel))
    for r in range(step//2, tissue.shape[0], step):
        for c in range(step//2, tissue.shape[1], step):
            angle = analysis["local_phi_deg"][r, c]
            if np.isfinite(angle):
                glyphs.append({"roi_row": r, "roi_column": c, "x_um": (c+.5)*pixel,
                               "y_um": (r+.5)*pixel, "phi_deg": float(angle),
                               "local_S": float(analysis["S_local"][r, c]),
                               "length_um": config["direction_length_um"]})
    return {"spatial": spatial, "depth_groups": depth_groups, "depth_summaries": summaries,
            "rainclouds": rainclouds, "median_trends": median_trends,
            "angular_spread_view": spread_view,
            "depth_histograms": histograms, "m2_hist_edges": m2_edges, "s_hist_edges": s_edges,
            "spread_hist_edges": spread_edges,
            "scale_maps": np.asarray(scale_maps), "scale_values": scale_values,
            "scale_frame_crop_xyxy": scale_crop,
            "scale_scatter": scatter,
            "scale_histograms": scale_histograms,
            "PA_distribution": pa_doa_distribution(pa_display, config["pa_histogram_bins"]),
            "PA_neighbour_mean": neighbour,
            "PA_neighbour_n": neighbour_n, "PA_pairs": pairs, "glyphs": glyphs}
