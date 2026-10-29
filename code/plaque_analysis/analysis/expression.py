"""Program-based gene selection and neighbourhood expression clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


CLASS_ORDER = ("S1", "S2", "S3")


def normalise_gene_counts(counts: pd.DataFrame, library_sizes: pd.Series):
    """Compute log2(CPM + 1) and sample-SD gene Z using all-gene libraries.

    Constant rows have undefined Z scores and remain NaN. Normalization is
    performed across all qualifying neighbourhoods, without class information.
    """
    if counts.index.has_duplicates or counts.columns.has_duplicates:
        raise ValueError("Gene and plaque identifiers must be unique")
    if counts.index.isna().any() or counts.columns.isna().any():
        raise ValueError("Gene and plaque identifiers must be nonmissing")
    if library_sizes.index.has_duplicates or set(counts.columns) - set(library_sizes.index):
        raise ValueError("Each plaque requires exactly one all-gene library total")
    values = counts.to_numpy(dtype=float)
    totals = library_sizes.reindex(counts.columns).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Library totals must be finite and positive")
    if (values.sum(axis=0) > totals + 1e-8).any():
        raise ValueError("Gene counts cannot exceed the all-gene library total")
    log = pd.DataFrame(np.log2(1 + values / totals * 1e6), index=counts.index,
                       columns=counts.columns)
    standard_deviation = log.std(axis=1, ddof=1)
    standard_deviation = standard_deviation.where(log.max(axis=1) > log.min(axis=1))
    z = log.sub(log.mean(axis=1), axis=0).div(standard_deviation.replace(0, np.nan), axis=0)
    return log, z


def _programs(program_membership):
    required = {"gene", "program", "selection_program"}
    if required - set(program_membership):
        raise ValueError(f"Missing membership columns: {sorted(required - set(program_membership))}")
    membership = program_membership[list(sorted(required))].copy()
    if membership.isna().any().any() or membership.duplicated(["gene", "program"]).any():
        raise ValueError("Gene-program memberships must be nonmissing and unique")
    if not membership.selection_program.isin([True, False, 0, 1]).all():
        raise ValueError("selection_program must contain Boolean values")
    membership["selection_program"] = membership.selection_program.astype(bool)
    if membership.groupby("program").selection_program.nunique().gt(1).any():
        raise ValueError("Every program must have one consistent selection_program flag")
    if membership.loc[membership.selection_program, "program"].nunique() != 16:
        raise ValueError("Exactly 16 custom programs must be marked selection_program=True")
    return membership


def select_program_genes(counts, library_sizes, log, membership, additional_genes):
    """Rank eligible genes independently within each custom program.

    Eligibility requires at least 10 total counts, detection in at least 30
    neighbourhoods, and nonconstant log expression. Ties are resolved by mean
    CPM, then gene symbol. Additional genes bypass abundance thresholds but
    must be present and variable. Class labels are not inputs to this function.
    """
    cpm = counts.div(library_sizes.reindex(counts.columns), axis=1) * 1e6
    statistics = pd.DataFrame({"gene": counts.index,
                               "total_count": counts.sum(axis=1).to_numpy(),
                               "detected_neighborhoods": counts.gt(0).sum(axis=1).to_numpy(),
                               "mean_CPM": cpm.mean(axis=1).to_numpy(),
                               "log_expression_sd": log.std(axis=1, ddof=1).to_numpy(),
                               "variable": (log.max(axis=1) > log.min(axis=1)).to_numpy()})
    stats = membership.loc[membership.selection_program, ["gene", "program"]].merge(
        statistics, how="left", on="gene", validate="many_to_one")
    stats["present"] = stats.total_count.notna()
    stats["selection_eligible"] = (stats.present & stats.total_count.ge(10)
                                    & stats.detected_neighborhoods.ge(30)
                                    & stats.log_expression_sd.gt(0) & stats.variable.eq(True))
    stats = stats.sort_values(["program", "selection_eligible", "detected_neighborhoods", "mean_CPM", "gene"],
                              ascending=[True, False, False, False, True], na_position="last")
    stats["selection_rank"] = stats.groupby("program", sort=False).cumcount() + 1
    stats["chosen_for_program"] = stats.selection_eligible & stats.selection_rank.le(3)
    selected = set(stats.loc[stats.chosen_for_program, "gene"])
    if not selected:
        raise ValueError("No custom-program genes satisfy the count, detection and variation thresholds")
    variable = statistics.set_index("gene").variable
    for gene in additional_genes:
        if gene not in counts.index:
            raise ValueError(f"Additional gene is absent from the RNA matrix: {gene}")
        if gene not in set(membership.gene):
            raise ValueError(f"Supply the gene-program membership for additional gene: {gene}")
        if not variable.loc[gene]:
            raise ValueError(f"Additional gene has constant neighbourhood expression: {gene}")
        selected.add(gene)
    return sorted(selected), stats.reset_index(drop=True)


def _class_series(classes, plaque_ids):
    if isinstance(classes, pd.DataFrame):
        classes = classes.set_index("plaque_id")["class"] if "plaque_id" in classes else classes["class"]
    if not isinstance(classes, pd.Series) or classes.index.has_duplicates:
        raise ValueError("Classes must supply one indexed class per plaque")
    aligned = classes.reindex(plaque_ids)
    if not aligned.isin(CLASS_ORDER).all():
        raise ValueError("Each analysed plaque requires a class in S1, S2 or S3")
    return aligned


def cluster_expression(z, classes):
    """Euclidean/Ward hierarchy for genes and separately within each class.

    SciPy's Ward linkage uses Euclidean observations, corresponding to R's
    Ward.D2 criterion. Unclipped gene Z values are used for every distance.
    """
    if not np.isfinite(z.to_numpy()).all():
        raise ValueError("Selected genes must have finite, nonconstant Z scores")
    row_linkage = linkage(z.to_numpy(), method="ward", metric="euclidean") if len(z) > 1 else None
    row_order = leaves_list(row_linkage) if row_linkage is not None else np.array([0])
    column_linkages, column_inputs, ordered_columns = {}, {}, []
    for label in CLASS_ORDER:
        ids = classes.index[classes.eq(label)].tolist()
        if not ids:
            continue
        column_inputs[label] = ids
        tree = linkage(z.loc[:, ids].T.to_numpy(), method="ward", metric="euclidean") if len(ids) > 1 else None
        column_linkages[label] = tree
        order = leaves_list(tree) if tree is not None else np.array([0])
        ordered_columns.extend([ids[index] for index in order])
    return z.index[row_order].tolist(), ordered_columns, row_linkage, column_linkages, column_inputs


def prepare_expression(counts_gene_by_plaque, library_sizes, classes, program_membership,
                       additional_genes=("Hspa5",)):
    """Select program genes, standardize expression and compute heatmap orders.

    Counts are gene-by-neighbourhood raw sums. Libraries include all genes;
    cell composition is not regressed out. Auxiliary programs (for example UPR)
    have selection_program=False and contribute membership links only.
    """
    counts = counts_gene_by_plaque.copy()
    if counts.shape[1] < 2 or counts.empty:
        raise ValueError("Expression analysis requires genes and at least two neighbourhoods")
    if not all(isinstance(gene, str) and gene for gene in counts.index):
        raise ValueError("RNA rows require nonempty gene symbols")
    membership = _programs(program_membership)
    log, z = normalise_gene_counts(counts, library_sizes)
    genes, statistics = select_program_genes(counts, library_sizes, log, membership, additional_genes)
    classes = _class_series(classes, counts.columns)
    rows, columns, row_tree, column_trees, column_inputs = cluster_expression(z.loc[genes], classes)
    ordered_z, ordered_log = z.loc[rows, columns], log.loc[rows, columns]
    displayed_membership = membership.loc[membership.gene.isin(genes), ["gene", "program"]].copy()
    displayed_membership = displayed_membership.sort_values(["program", "gene"]).reset_index(drop=True)
    row_table = pd.DataFrame({"position": np.arange(1, len(rows) + 1), "gene": rows})
    column_table = pd.DataFrame({"position": np.arange(1, len(columns) + 1), "plaque_id": columns,
                                 "class": classes.reindex(columns).to_numpy()})
    return {"log_expression": ordered_log, "z": ordered_z, "rows": row_table,
            "columns": column_table, "membership": displayed_membership,
            "row_linkage": row_tree, "column_linkages": column_trees,
            "row_inputs": genes, "column_inputs": column_inputs,
            "selection_stats": statistics, "library_sizes": library_sizes.reindex(columns),
            "counts": counts.loc[rows, columns]}
