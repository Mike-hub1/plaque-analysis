"""Figure 4: photoacoustic."""
from __future__ import annotations

from scipy.stats import gaussian_kde
import numpy as np
from plaque_analysis.imaging.orientation import distribution


def pa_grid_statistics(pa):
    valid = np.isfinite(pa)
    if not valid.any() or np.any(pa[valid] < 0) or np.any(pa[valid] > 1):
        raise ValueError("PA DOA must contain finite values in [0,1]; missing values must be NaN.")
    differences = []
    for axis in (0, 1):
        left = [slice(None), slice(None)]; right = left.copy()
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        a, b = pa[tuple(left)], pa[tuple(right)]
        ok = np.isfinite(a) & np.isfinite(b)
        differences.extend(np.abs(a[ok]-b[ok]).tolist())
    stats = distribution(pa[valid])
    stats["adjacent_pair_n"] = len(differences)
    stats["adjacent_abs_difference_mean"] = float(np.mean(differences)) if differences else np.nan
    return stats


def display_pa(pa, rotate=1, mirror=True):
    r, c = np.where(np.isfinite(pa))
    if not len(r):
        raise ValueError("PA matrix has no finite cells.")
    bbox = [int(r.min()), int(r.max())+1, int(c.min()), int(c.max())+1]
    transformed = np.rot90(pa[bbox[0]:bbox[1], bbox[2]:bbox[3]], rotate)
    if mirror:
        transformed = np.fliplr(transformed)
    return transformed, bbox


def pa_neighbours(pa):
    """Count each undirected horizontal/vertical pair exactly once, retaining 0."""
    pa = np.asarray(pa, float)
    total = np.zeros(pa.shape)
    count = np.zeros(pa.shape, int)
    pairs = []
    for dr, dc in [(0, 1), (1, 0)]:
        for r in range(pa.shape[0]-dr):
            for c in range(pa.shape[1]-dc):
                r2, c2 = r+dr, c+dc
                if not (np.isfinite(pa[r, c]) and np.isfinite(pa[r2, c2])):
                    continue
                delta = abs(pa[r, c]-pa[r2, c2])
                pairs.append({"row_1": r, "column_1": c, "row_2": r2,
                              "column_2": c2, "absolute_DOA_difference": float(delta)})
                for y, x in [(r, c), (r2, c2)]:
                    total[y, x] += delta
                    count[y, x] += 1
    mean = np.full(pa.shape, np.nan)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, count, pairs


def pa_doa_distribution(pa, bins=26):
    """All-cell histogram and a reflected, Scott-bandwidth density guide."""
    values = np.asarray(pa, float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0):
        raise ValueError("The DOA distribution requires nonnegative finite observations.")
    edges = np.linspace(0, max(.16, float(values.max())), bins+1)
    counts, _ = np.histogram(values, edges)
    if counts.sum() != values.size:
        raise ValueError("The DOA histogram must retain every finite PA observation.")
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    width = float(edges[1]-edges[0])
    bandwidth = np.nan
    grid, density = np.array([]), np.array([])
    if values.size > 1 and np.ptp(values) > 0:
        kde = gaussian_kde(values, bw_method="scott")
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))
        # f(x)+f(-x) puts the Gaussian negative tail back on the DOA >= 0 domain.
        # Retain the full numerical tail in Source Data, beyond the displayed axis.
        grid = np.linspace(0, max(edges[-1], values.max()+6*bandwidth), 1601)
        density = kde(grid)+kde(-grid)
    summary = {"support_n": int(values.size), "histogram_bins": int(bins),
               "histogram_bin_width": width, "axis_max": float(edges[-1]),
               "q25": float(q25), "median": float(median), "q75": float(q75),
               "IQR": float(q75-q25), "KDE_bandwidth": bandwidth,
               "KDE_method": "Gaussian, Scott bandwidth, reflection at zero",
               "DOA_units": "dimensionless"}
    return {"values": values, "edges": edges, "counts": counts,
            "percent": counts/values.size*100, "summary": summary,
            "curve_x": grid, "curve_density": density,
            "curve_percent_per_bin": density*width*100}
