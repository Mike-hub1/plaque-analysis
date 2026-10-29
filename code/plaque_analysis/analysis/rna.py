"""RNA quality control and expression analysis for a shared cell cohort."""

import numpy as np
from scipy import sparse


def mad_threshold(values, side, multiplier):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    spread = 1.4826 * np.median(np.abs(values - median))
    if spread <= np.finfo(float).eps:
        return float(np.quantile(values, .01 if side == "lower" else .99))
    return float(median + (-1 if side == "lower" else 1) * multiplier * spread)


def validate_counts(counts, genes, cells):
    if not sparse.issparse(counts):
        raise ValueError("RNA counts must be a sparse cell-by-gene matrix")
    matrix = counts.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape != (len(cells), len(genes)) or not min(matrix.shape):
        raise ValueError("RNA matrix rows and columns must match cells and genes")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or (matrix.data != np.floor(matrix.data)).any():
        raise ValueError("RNA counts must be finite nonnegative integer observations")
    for frame, columns in ((genes, ("gene", "is_mitochondrial")),
                           (cells, ("cell_id", "sample_id", "x_dapi", "y_dapi"))):
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"Missing RNA input fields: {sorted(missing)}")
    if genes.gene.isna().any() or genes.gene.duplicated().any() or cells.cell_id.isna().any() or cells.cell_id.duplicated().any():
        raise ValueError("Gene names and cell identifiers must be present and unique")
    if cells.sample_id.isna().any() or not np.isfinite(cells[["x_dapi", "y_dapi"]].to_numpy(dtype=float)).all():
        raise ValueError("Cells require sample identifiers and finite DAPI coordinates")
    return matrix


def rna_quality_control(counts, genes, cells, settings):
    from plaque_analysis.analysis.rna_annotation import boolean_values

    result = cells.reset_index(drop=True).copy()
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    detected = np.asarray(counts.getnnz(axis=1)).ravel()
    mito = boolean_values(genes.is_mitochondrial, "is_mitochondrial")
    mito_counts = np.asarray(counts[:, mito].sum(axis=1)).ravel()
    percent = np.divide(100 * mito_counts, totals, out=np.zeros_like(totals), where=totals > 0)
    result["n_counts"], result["n_genes"], result["mito_percent"] = totals, detected, percent
    low, high, mt = np.zeros((3, len(result)), dtype=bool)
    for indices in result.groupby("sample_id", sort=False).indices.values():
        lc, lg = np.log10(totals[indices] + 1), np.log10(detected[indices] + 1)
        low[indices] = (lc < mad_threshold(lc, "lower", settings.get("lower_mad_counts", 3))) | (
            lg < mad_threshold(lg, "lower", settings.get("lower_mad_genes", 3)))
        high[indices] = (lc > mad_threshold(lc, "upper", settings.get("upper_mad_counts", 4))) | (
            lg > mad_threshold(lg, "upper", settings.get("upper_mad_genes", 4)))
        mt[indices] = percent[indices] > mad_threshold(percent[indices], "upper", settings.get("upper_mad_mito", 3))
    hard = (detected < settings.get("min_genes", 200)) | (percent > settings.get("max_mito_percent", 8.0)) | (totals <= 0)
    result["severe_low_quality"] = hard | (low & mt)
    result["low_quality_flag"] = result.severe_low_quality | low | high | mt
    result["doublet_like_flag"] = high
    result["rna_qc_pass"] = ~result.severe_low_quality
    return result


def prepare_rna(counts, genes, cells, annotations, config):
    """Return QC/annotation metadata, one global UMAP and lineage candidates.

    Counts follow the input cell and gene table order. Reviewed annotations and
    UCell candidate evidence are external inputs; this function does not create
    manual review decisions or use plaque classes in RNA analysis.
    """
    from plaque_analysis.analysis.rna_annotation import attach_reviewed_annotations
    from plaque_analysis.analysis.rna_embedding import build_embeddings

    settings = config.get("rna", {})
    matrix = validate_counts(counts, genes, cells)
    enriched = rna_quality_control(matrix, genes, cells, settings)
    enriched, scores = attach_reviewed_annotations(enriched, annotations, settings)
    passed = enriched.rna_qc_pass.to_numpy()
    if passed.sum() < 4:
        raise ValueError("At least four RNA-QC-passing cells are required for a shared UMAP")
    embedding, clusters = build_embeddings(matrix[passed], genes, enriched.loc[passed].copy(), settings)
    scores = scores.merge(clusters, on="cell_id", how="left", validate="one_to_one", sort=False)
    return enriched, embedding, scores
