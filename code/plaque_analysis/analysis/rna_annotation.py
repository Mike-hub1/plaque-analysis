"""Apply external reviewed annotations and their UCell candidate evidence."""

import numpy as np
import pandas as pd


def boolean_values(values, name):
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{name} must contain explicit boolean values")
    return normalized.isin(["true", "1"]).to_numpy()


def attach_reviewed_annotations(cells, annotations, settings):
    """Apply reviewed broad labels; confidence refers to that broad annotation.

    UCell margins and positive-marker counts are externally computed subtype
    evidence. Their thresholds govern subtype use, not reviewed broad labels.
    """
    annotations = annotations.rename(columns={
        name: target for name, target in (("marker_count", "positive_marker_count"), ("margin", "ucell_margin"))
        if name in annotations and target not in annotations}).copy()
    required = {"cell_id", "broad_celltype", "reviewed_subtype", "confidence", "annotation_eligible", "reviewed_region"}
    if required - set(annotations):
        raise ValueError(f"Missing reviewed annotation fields: {sorted(required - set(annotations))}")
    if annotations.cell_id.isna().any() or annotations.cell_id.duplicated().any():
        raise ValueError("Reviewed annotations require unique cell identifiers")
    if not annotations.cell_id.isin(cells.cell_id).all():
        raise ValueError("An annotation cell identifier is absent from the count matrix")
    if not cells.loc[cells.rna_qc_pass, "cell_id"].isin(annotations.cell_id).all():
        raise ValueError("Every RNA-QC-passing cell requires an external annotation record")
    annotations["annotation_eligible"] = boolean_values(annotations.annotation_eligible, "annotation_eligible")
    annotations["confidence"] = annotations.confidence.astype(str).str.strip().str.lower()
    if not annotations.confidence.isin(["high", "medium", "low"]).all():
        raise ValueError("Annotation confidence must be high, medium or low")
    for field in ("broad_celltype", "reviewed_subtype", "reviewed_region"):
        annotations[field] = annotations[field].fillna("").astype(str).str.strip()
    needs_evidence = annotations.reviewed_subtype.ne("") | annotations.reviewed_region.ne("")
    evidence = ["positive_marker_count", "ucell_margin"]
    if needs_evidence.any() and set(evidence) - set(annotations):
        raise ValueError("Reviewed subtypes or regions require positive_marker_count and ucell_margin from external UCell candidates")
    for field in evidence:
        if field not in annotations:
            annotations[field] = np.nan
        annotations[field] = pd.to_numeric(annotations[field], errors="raise")
    values = annotations.loc[needs_evidence, evidence].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any() or (values[:, 0] != np.floor(values[:, 0])).any():
        raise ValueError("Reviewed subtypes or regions require finite UCell margins and nonnegative integer marker counts")
    annotations["candidate_supported"] = (
        annotations.positive_marker_count.ge(settings.get("min_positive_markers", 2)) &
        annotations.ucell_margin.ge(settings.get("min_ucell_margin", .05)))
    annotations["annotation_eligible"] &= annotations.confidence.isin(["high", "medium"])
    annotations["subtype_eligible"] = (
        annotations.annotation_eligible & annotations.candidate_supported & annotations.reviewed_subtype.ne(""))
    return merge_annotations(cells, annotations, required, evidence)


def merge_annotations(cells, annotations, required, evidence):
    fields = sorted(required - {"cell_id"}) + evidence + ["candidate_supported", "subtype_eligible"]
    result = cells.drop(columns=[field for field in fields if field in cells]).merge(
        annotations[["cell_id", *fields]], on="cell_id", how="left", validate="one_to_one", sort=False)
    result["annotation_reviewed"] = result.cell_id.isin(annotations.cell_id)
    result["annotation_eligible"] = result.annotation_eligible.fillna(False).astype(bool)
    result["candidate_supported"] = result.candidate_supported.fillna(False).astype(bool)
    result["subtype_eligible"] = result.subtype_eligible.fillna(False).astype(bool)
    for field in ("broad_celltype", "reviewed_subtype", "reviewed_region"):
        result[field] = result[field].fillna("").astype(str).str.strip()
    if result.loc[result.annotation_eligible, "broad_celltype"].eq("").any():
        raise ValueError("Eligible annotations require a reviewed broad cell type")
    result["confidence"] = result.confidence.fillna("low")
    result["reviewed_broad_celltype"] = result.broad_celltype
    result.loc[~result.annotation_eligible, "broad_celltype"] = "Low-confidence / ambiguous"
    scores = result[["cell_id", "reviewed_broad_celltype", "confidence", *evidence,
                     "candidate_supported", "annotation_eligible", "subtype_eligible"]].copy()
    return result, scores
