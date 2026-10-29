"""Positive-signature UCell scores from cell-by-gene expression.

This independent implementation follows the rank normalization used by UCell
2.16: descending average-tie ranks, capped at maxRank, and a normalized
Mann–Whitney statistic. Algorithm reference:
https://github.com/carmonalab/UCell/blob/master/R/HelperFunctions.R
Negative signatures and nearest-neighbour score smoothing are not applied.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def _names(values, name, *, unique=True):
    names = pd.Index(values)
    if names.hasnans or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError(f"{name} must contain nonempty gene/program names")
    if unique and names.has_duplicates:
        raise ValueError(f"{name} must be unique")
    return names


def _expression_input(counts, genes, cell_ids, max_rank):
    if not sparse.issparse(counts) or counts.ndim != 2 or min(counts.shape) == 0:
        raise ValueError("counts must be a nonempty sparse cell-by-gene matrix")
    if np.iscomplexobj(counts.data):
        raise ValueError("Expression values must be real, finite and nonnegative")
    matrix = counts.tocsr(copy=True).astype(float)
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Expression values must be real, finite and nonnegative")
    genes = _names(genes, "genes")
    if matrix.shape[1] != len(genes):
        raise ValueError("Gene names must follow the expression matrix column order")
    ids = pd.Index(range(matrix.shape[0]) if cell_ids is None else cell_ids)
    if len(ids) != matrix.shape[0] or ids.has_duplicates or ids.hasnans:
        raise ValueError("cell_ids must uniquely identify expression matrix rows")
    if isinstance(max_rank, bool) or not isinstance(max_rank, (int, np.integer)) or max_rank < 2:
        raise ValueError("max_rank must be an integer of at least two")
    cutoff = min(int(max_rank), len(genes))
    if cutoff < 2:
        raise ValueError("At least two assayed genes are required for UCell normalization")
    return matrix, genes, ids, cutoff


def _signatures(gene_sets, genes, cutoff, missing_genes):
    if missing_genes not in ("skip", "impute"):
        raise ValueError("missing_genes must be 'skip' or 'impute'")
    if (not isinstance(gene_sets, pd.DataFrame) or not gene_sets.columns.is_unique
            or not {"program", "gene"}.issubset(gene_sets)):
        raise ValueError("gene_sets must contain program and gene columns")
    if gene_sets.empty or gene_sets.duplicated(["program", "gene"]).any():
        raise ValueError("Gene sets must be nonempty with unique program/gene pairs")
    _names(gene_sets.program, "program", unique=False)
    _names(gene_sets.gene, "signature genes", unique=False)
    if gene_sets.gene.str.endswith(("+", "-")).any():
        raise ValueError("Use plain gene symbols for positive signatures; signed suffixes are unsupported")
    signatures = []
    for program, frame in gene_sets.groupby("program", sort=False):
        requested = len(frame)
        if requested > cutoff:
            raise ValueError(f"{program}: signature size exceeds effective max_rank={cutoff}")
        indices = genes.get_indexer(frame.gene)
        matched = indices[indices >= 0]
        scoring = len(matched) if missing_genes == "skip" else requested
        signatures.append((program, matched, requested, scoring))
    return signatures


def _score_matrix(matrix, signatures, cutoff, missing_genes):
    scores = np.full((matrix.shape[0], len(signatures)), np.nan)
    for cell in range(matrix.shape[0]):
        ranks = rankdata(-matrix.getrow(cell).toarray().ravel(), method="average")
        ranks = np.minimum(ranks, cutoff)
        for column, (_, indices, requested, scoring) in enumerate(signatures):
            if scoring == 0:
                continue
            imputed = (requested - len(indices)) * cutoff if missing_genes == "impute" else 0
            rank_sum = ranks[indices].sum() + imputed
            minimum = scoring * (scoring + 1) / 2
            scores[cell, column] = 1 - (rank_sum - minimum) / (scoring * cutoff - minimum)
    return scores


def score_ucell(counts, genes, gene_sets, *, cell_ids=None, max_rank=1500,
                missing_genes="impute"):
    """Return one score per cell/program without fitting any model.

    Matrix columns follow ``genes``; rows follow ``cell_ids`` (or their integer
    positions when omitted). Signatures use ``program``/``gene`` columns and
    plain positive gene symbols. All assayed genes, including zero-expression
    genes, participate in ranking; maxRank is min(max_rank, number of genes).
    ``skip`` removes unassayed genes; ``impute`` assigns them the rank cutoff.
    Entirely unmatched signatures return NaN and an explicit status under
    skip, or score zero under impute. No cell filtering occurs here.
    """
    matrix, genes, ids, cutoff = _expression_input(counts, genes, cell_ids, max_rank)
    signatures = _signatures(gene_sets, genes, cutoff, missing_genes)
    scores = _score_matrix(matrix, signatures, cutoff, missing_genes)
    result = []
    for column, (program, indices, requested, scoring) in enumerate(signatures):
        result.append(pd.DataFrame({"cell_id": ids, "program": program,
            "score": scores[:, column], "n_signature_genes": requested,
            "n_matched_genes": len(indices), "n_scoring_genes": scoring,
            "effective_max_rank": cutoff, "missing_genes": missing_genes,
            "status": "no_matched_genes" if not len(indices) else "ok"}))
    return pd.concat(result, ignore_index=True)
