"""Compute mouse panels from cell counts, reviewed labels and supplied geometry."""

import numpy as np
import pandas as pd
from plaque_analysis.settings.microenvironment import CLASS_ORDER, GLIAL_ORDER, GLIAL_PROGRAMS, PRIMARY
from plaque_analysis.analysis.mouse_neighborhoods import validate_mouse_inputs, composition_points
from plaque_analysis.analysis.mouse_expression import gene_points, aggregate_program_scores, paired_scores


def glial_points(scores, seed=42):
    selected, summaries = [], []
    rng = np.random.default_rng(seed)
    for code, cell_type, _, _ in GLIAL_PROGRAMS:
        q = scores[scores.program.eq(code) & scores.window.eq("0-300") & scores.eligible].copy()
        q["x_drawn"] = q.S_label.map({label: i for i, label in enumerate(GLIAL_ORDER)}).to_numpy() + rng.uniform(-.17, .17, len(q))
        selected.append(q)
        for label in CLASS_ORDER:
            values = q.loc[q.S_label.eq(label), "score"].to_numpy(float)
            low, median, high = np.quantile(values, [.25, .5, .75]) if len(values) else (np.nan,) * 3
            summaries.append({"program": code, "cell_type": cell_type, "S_label": label,
                              "n": len(values), "median": median, "q25": low, "q75": high})
    return pd.concat(selected, ignore_index=True), pd.DataFrame(summaries)


def spatial_points(scores, plaques, display):
    matrix = np.asarray(display.get("coordinate_transform", np.eye(3)), dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-8)):
        raise ValueError("display.coordinate_transform must be a rigid 2-D homogeneous transform")
    background = plaques.copy()
    xy = np.column_stack((plaques[["x_um", "y_um"]].to_numpy(float), np.ones(len(plaques)))) @ matrix.T
    background["display_x_mm"], background["display_y_mm"] = (xy[:, :2] / 1000).T
    q = scores[scores.program.isin(PRIMARY) & scores.window.eq("0-150") & scores.eligible].copy()
    q = q.merge(background[["plaque_id", "display_x_mm", "display_y_mm"]],
                on="plaque_id", how="left", validate="many_to_one")
    return q, background


def prepare_mouse(source, config=None):
    """Return a–e panel tables; counts follow cells/genes input row order.

    Cells carry one upstream nearest-plaque assignment and signed boundary
    distance (negative inside). Areas are measured, observable exterior annuli.
    Plaques failing neighborhood eligibility are excluded without reassigning
    their cells. UCell uses raw counts and externally specified gene sets.
    """
    from plaque_analysis.analysis.ucell import score_ucell
    from plaque_analysis.spatial.statistics import bootstrap_spatial_contrasts

    config = config or {}
    cells, plaques, counts, genes, areas = validate_mouse_inputs(source)
    minimum = config.get("mouse", {}).get("min_cells", 5)
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("mouse.min_cells must be a positive integer")
    gene_table, gene_support, gene_all = gene_points(cells, plaques, counts, genes, minimum)
    programs = {*PRIMARY, *(code for code, _, _, _ in GLIAL_PROGRAMS)}
    memberships = source["memberships"]
    if "program" not in memberships or programs - set(memberships.program):
        raise ValueError(f"Gene-set membership must supply all four displayed programs: {sorted(programs)}")
    score_mask = cells.state_eligible & cells.broad_celltype.isin(["Excitatory neuron", "Microglia", "Astrocyte"])
    settings = {"max_rank": 1500, "missing_genes": "impute", **config.get("ucell", {})}
    if score_mask.any():
        cell_scores = score_ucell(counts[score_mask.to_numpy()], genes, memberships,
                                 cell_ids=cells.loc[score_mask, "cell_id"].to_numpy(), **settings)
    else:
        cell_scores = pd.DataFrame({"cell_id": pd.Series(dtype=cells.cell_id.dtype),
                                    "program": pd.Series(dtype=str), "score": pd.Series(dtype=float),
                                    "status": pd.Series(dtype=str)})
    scores = aggregate_program_scores(cells, plaques, cell_scores, minimum)
    glial, glial_summary = glial_points(scores, int(config.get("mouse", {}).get("jitter_seed", 42)))
    display = dict(config.get("display", {}))
    if "display_invert_y" in source:
        display["invert_y"] = bool(source["display_invert_y"])
    spatial, locations = spatial_points(scores, plaques, display)
    pairs = paired_scores(scores)
    intervals = bootstrap_spatial_contrasts(pairs, **config.get("bootstrap", {}))
    differences = pairs.copy()
    differences["delta_display_x1000"] = 1000 * differences.delta
    return {"panel_a_composition": composition_points(cells, plaques, areas),
            "panel_b_positive_points": gene_table, "panel_b_detection_support": gene_support,
            "panel_c_reference_program_scores": glial, "panel_c_summary": glial_summary,
            "panel_d_scores": spatial, "panel_d_locations": locations,
            "panel_e_differences": differences, "panel_e_intervals": intervals,
            "cell_program_scores": cell_scores, "plaque_program_scores": scores,
            "plaque_gene_expression": gene_all, "paired_scores": pairs, "display_settings": display}
