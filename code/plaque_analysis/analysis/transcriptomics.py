"""Run the manuscript methods from supplied measurements and fixed models."""

from plaque_analysis.io.transcriptomics import load_inputs
from plaque_analysis.analysis.rna import prepare_rna
from plaque_analysis.spatial.registration import register_cells, summarize_registration
from plaque_analysis.models.classification import assign_fixed_classes
from plaque_analysis.spatial.neighborhoods import assign_neighborhoods, aggregate_counts
from plaque_analysis.analysis.expression import prepare_expression
from plaque_analysis.spatial.regions import infer_reviewed_regions


def prepare_analysis(data_dir, config_file):
    source = load_inputs(data_dir, config_file)
    config = source["config"]
    annotated, embedding, annotation_scores = prepare_rna(
        source["counts"], source["genes"], source["cells"], source["annotations"], config)
    cells, landmark_errors = register_cells(annotated, source["landmarks"],
        source["tissue_mask"], config["registration"])
    qc = summarize_registration(cells)
    classes = assign_fixed_classes(source["features"], source["model_path"])
    plaques = source["plaques"].merge(classes, on="plaque_id", how="left", validate="one_to_one")
    assignments, neighborhoods = assign_neighborhoods(cells, source["plaque_mask"], plaques,
        pixel_size_um=config["registration"]["target_pixel_size_um"], **config["neighborhood"])
    counts, libraries = aggregate_counts(source["counts"], source["cells"].cell_id,
        source["genes"].gene, assignments, neighborhoods)
    expression = prepare_expression(counts, libraries, classes, source["membership"])
    spatial = prepare_spatial(cells, embedding, neighborhoods, config["region"])
    tables = {
        "cells_processed.csv.gz": cells,
        "rna_annotation_candidates.csv.gz": annotation_scores,
        "panel_b_umap.csv": embedding,
        "landmark_leave_one_out.csv": landmark_errors,
        "panel_d_registration_summary.csv": qc,
        "plaque_fixed_model_assignments.csv": classes,
        "cell_neighborhood_assignments.csv.gz": assignments,
        "plaque_neighborhoods.csv": neighborhoods,
        "panel_e_cells.csv.gz": spatial["e_cells"],
        "panel_e_plaques.csv": spatial["e_plaques"],
        "neighborhood_library_sizes.csv": libraries.rename("library_size_all_genes").rename_axis("plaque_id").reset_index(),
        "panel_f_gene_selection.csv": expression["selection_stats"],
        "panel_f_gene_row_order.csv": expression["rows"],
        "panel_f_plaque_column_order.csv": expression["columns"],
        "panel_f_log2_CPM_plus_1.csv": expression["log_expression"].rename_axis("gene").reset_index(),
        "panel_f_gene_row_Z.csv": expression["z"].rename_axis("gene").reset_index(),
        "panel_f_gene_program_membership.csv": expression["membership"],
    }
    report = {"status": "completed", "panels": {letter: {} for letter in "abcdef"},
        "cell_counts": {"all": len(cells), "rna_qc_pass": int(cells.rna_qc_pass.sum()),
                        "mapping_available": int(cells.valid_for_spatial_sampling.sum())},
        "neighborhoods": len(expression["z"].columns), "display_genes": len(expression["z"]),
        "registration_error": "interpolated landmark leave-one-out Euclidean error in target micrometres",
        "class_assignment": "fixed Figure 5 transform and GMM posterior argmax; no refitting",
        "expression": "all-gene pseudobulk CPM; log2(CPM+1); gene-wise sample-SD Z; no composition adjustment",
        "clustering": "Euclidean distance and Ward linkage; columns within fixed classes; unclipped Z",
        "configuration": config}
    prepared = {"spatial": spatial, "qc": qc, "expression": expression, "config": config}
    return report, tables, prepared


def prepare_spatial(cells, embedding, neighborhoods, region_config):
    """Use the same RNA-qualified nuclei for a/b and one coordinate frame for a/c/e."""
    a = cells.loc[cells.rna_qc_pass].copy()
    b = a.merge(embedding, on="cell_id", how="left", validate="one_to_one")
    if b[["UMAP1", "UMAP2"]].isna().any().any() or len(b) != len(embedding):
        raise ValueError("Panels a and b must contain the same RNA-qualified cell identifiers")
    e_cells = cells.loc[cells.valid_for_spatial_sampling].copy()
    e_plaques = neighborhoods.loc[neighborhoods.neighborhood_eligible].copy()
    for frame in (e_cells, e_plaques):
        regions = infer_reviewed_regions(frame[["x_um", "y_um"]].to_numpy(float), cells, **region_config)
        for column in regions:
            frame[column] = regions[column].to_numpy()
    return {"a": a, "b": b, "c": cells, "e_cells": e_cells, "e_plaques": e_plaques}
