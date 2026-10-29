"""Cell-type expression, program means and matched near/far differences."""

import numpy as np
import pandas as pd
from plaque_analysis.settings.microenvironment import CLASS_ORDER, GENES, GLIAL_PROGRAMS, PRIMARY


def gene_points(cells, plaques, counts, genes, minimum=5):
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    records, support, complete = [], [], []
    for gene, _ in GENES:
        cell_type = {"Hspa5": "Excitatory neuron", "Gfap": "Astrocyte", "P2ry12": "Microglia", "Trem2": "Microglia"}[gene]
        radius = 150 if cell_type == "Excitatory neuron" else 300
        mask = cells.state_eligible & cells.broad_celltype.eq(cell_type) & cells.boundary_distance_um.lt(radius)
        q = cells.loc[mask, ["cell_id", "plaque_id"]].copy()
        if gene in genes:
            raw = counts[mask.to_numpy(), genes.index(gene)].toarray().ravel()
            q["detected"] = raw > 0
            q["expression"] = np.log1p(10000 * raw / libraries[mask.to_numpy()])
        else:
            q["detected"], q["expression"] = False, np.nan
        aggregated = q.groupby("plaque_id").agg(n_cells=("cell_id", "size"), n_detected=("detected", "sum"), expression_sum=("expression", lambda x: x.sum(min_count=1)))
        d = plaques[["plaque_id", "S_label"]].merge(aggregated, on="plaque_id", how="left", validate="one_to_one")
        d[["n_cells", "n_detected"]] = d[["n_cells", "n_detected"]].fillna(0).astype(int)
        d["gene"], d["cell_type"] = gene, cell_type
        d["window_actual"] = f"d < {radius} um; includes Inside"
        d["gene_available"] = gene in genes
        d["eligible"] = d.n_cells.ge(minimum) & d.gene_available
        d["p_detect"] = d.n_detected / d.n_cells.replace(0, np.nan)
        d["mu_positive"] = d.expression_sum / d.n_detected.replace(0, np.nan)
        d["mean_log1p_cp10k"] = d.expression_sum / d.n_cells.replace(0, np.nan)
        complete.append(d)
        for label in CLASS_ORDER:
            group = d[d.S_label.eq(label) & d.eligible]
            positive = group[group.n_detected.gt(0)]
            records.append(positive)
            support.append({"gene": gene, "S_label": label, "gene_available": gene in genes,
                            "n_eligible": len(group), "n_positive_plaques": len(positive),
                            "n_undetected_plaques": int(group.n_detected.eq(0).sum())})
    return pd.concat(records, ignore_index=True), pd.DataFrame(support), pd.concat(complete, ignore_index=True)


def aggregate_program_scores(cells, plaques, cell_scores, minimum=5):
    programs = {**{code: "Excitatory neuron" for code in PRIMARY},
                **{code: cell_type for code, cell_type, _, _ in GLIAL_PROGRAMS}}
    joined = cell_scores.merge(cells[["cell_id", "plaque_id", "broad_celltype", "boundary_distance_um", "state_eligible"]],
                               on="cell_id", how="left", validate="many_to_one")
    if joined.duplicated(["cell_id", "program"]).any():
        raise ValueError("UCell output requires one score per cell and program")
    result = []
    windows = {"0-150": (0, 150), "0-300": (0, 300), "150-300": (150, 300), ">300": (300, np.inf)}
    for code, cell_type in programs.items():
        base = joined[joined.program.eq(code) & joined.state_eligible & joined.broad_celltype.eq(cell_type)]
        for window, (lower, upper) in windows.items():
            mask = base.boundary_distance_um.ge(lower) & base.boundary_distance_um.lt(upper)
            if window == ">300":
                mask &= base.boundary_distance_um.gt(300)
            q = base.loc[mask].copy()
            q["score"] = q.score.where(np.isfinite(q.score.to_numpy(float)) & q.status.eq("ok"))
            d = q.groupby("plaque_id").agg(n_cells=("cell_id", "size"), n_scored=("score", "count"), score=("score", "mean"))
            d = plaques.merge(d, on="plaque_id", how="left", validate="one_to_one")
            d[["n_cells", "n_scored"]] = d[["n_cells", "n_scored"]].fillna(0).astype(int)
            d["program"], d["cell_type"], d["window"], d["variant"] = code, cell_type, window, "standard"
            d["eligible"] = d.n_scored.ge(minimum)
            result.append(d)
    return pd.concat(result, ignore_index=True)


def paired_scores(scores):
    result = []
    for code in PRIMARY:
        q = scores[scores.program.eq(code) & scores.eligible]
        near = q[q.window.eq("0-150")].copy()
        for window in ("150-300", ">300"):
            far = q[q.window.eq(window)][["plaque_id", "score", "n_scored"]]
            pairs = near.merge(far, on="plaque_id", suffixes=("_near", "_far"), validate="one_to_one")
            pairs["far_window"] = window
            pairs["delta"] = pairs.score_far - pairs.score_near
            result.append(pairs)
    return pd.concat(result, ignore_index=True)
