"""Validate cell geometry and compute observable-area composition summaries."""

import numpy as np
import pandas as pd
from scipy import sparse
from plaque_analysis.settings.microenvironment import DISPLAY_TYPES, NAMED_TYPES


def explicit_boolean(values, field):
    values = values.astype(str).str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError(f"{field} requires explicit True/False values")
    return values.eq("true")


def validate_mouse_inputs(source):
    cells, plaques, areas = (source[name].reset_index(drop=True).copy() for name in ("cells", "plaques", "areas"))
    required = {"cell_id", "broad_celltype", "rna_qc_pass", "valid_for_spatial_sampling",
                "annotation_eligible", "x_um", "y_um", "plaque_id", "boundary_distance_um"}
    for name, frame, fields, identifier in (
        ("cells", cells, required, "cell_id"),
        ("plaques", plaques, {"plaque_id", "class", "x_um", "y_um", "neighborhood_eligible"}, "plaque_id"),
        ("areas", areas, {"plaque_id", "area_0_50_um2", "area_50_150_um2"}, "plaque_id")):
        if fields - set(frame):
            raise ValueError(f"{name}: missing columns {sorted(fields - set(frame))}")
        if frame[identifier].isna().any() or frame[identifier].astype(str).str.strip().eq("").any() or frame[identifier].duplicated().any():
            raise ValueError(f"{name}: {identifier} must be unique and nonempty")
    if not len(plaques) or not plaques["class"].isin(["S1", "S2", "S3"]).all():
        raise ValueError("Plaques require fixed S1/S2/S3 class labels")
    if not np.isfinite(plaques[["x_um", "y_um"]].to_numpy(float)).all():
        raise ValueError("Plaque coordinates must be finite physical micrometres")
    if set(areas.plaque_id) != set(plaques.plaque_id):
        raise ValueError("Every plaque requires one measured exterior-area record")
    observed = areas[["area_0_50_um2", "area_50_150_um2"]].to_numpy(float)
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Observed annulus areas must be finite and nonnegative")
    genes = list(source["genes"])
    if not genes or len(set(genes)) != len(genes) or any(pd.isna(g) or not str(g).strip() for g in genes):
        raise ValueError("Gene names must be unique and nonempty")
    if not sparse.issparse(source["counts"]):
        raise ValueError("Counts must be a sparse cell-by-gene matrix")
    counts = source["counts"].tocsr(copy=True)
    counts.sum_duplicates(); counts.eliminate_zeros()
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("Count matrix axes must follow cells and genes in the supplied order")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or (counts.data != np.floor(counts.data)).any():
        raise ValueError("Counts must be finite nonnegative integers")
    for flag in ("rna_qc_pass", "valid_for_spatial_sampling", "annotation_eligible"):
        cells[flag] = explicit_boolean(cells[flag], flag)
    assigned = cells.plaque_id.notna() & cells.plaque_id.astype(str).str.strip().ne("")
    if not cells.loc[assigned, "plaque_id"].isin(plaques.plaque_id).all():
        raise ValueError("Cell plaque assignments must refer to the supplied plaque table")
    plaques["neighborhood_eligible"] = explicit_boolean(plaques.neighborhood_eligible, "neighborhood_eligible")
    plaques = plaques.loc[plaques.neighborhood_eligible].copy()
    if not len(plaques):
        raise ValueError("No plaques pass the upstream neighborhood inclusion criteria")
    cells["qc_eligible"] = cells.rna_qc_pass & cells.valid_for_spatial_sampling & cells.plaque_id.isin(plaques.plaque_id)
    if not np.isfinite(cells.loc[cells.qc_eligible, ["x_um", "y_um", "boundary_distance_um"]].to_numpy(float)).all():
        raise ValueError("Eligible cells require finite coordinates and signed boundary distances")
    cells["broad_celltype"] = cells.broad_celltype.fillna("").astype(str).str.strip()
    resolved = ~cells.broad_celltype.str.lower().isin(["", "amb", "unknown", "unresolved", "low-confidence / ambiguous"])
    cells["state_eligible"] = cells.qc_eligible & cells.annotation_eligible & resolved
    if (np.asarray(counts.sum(axis=1)).ravel()[cells.qc_eligible.to_numpy()] <= 0).any():
        raise ValueError("RNA-qualified cells require positive all-gene library sizes")
    plaques = plaques.rename(columns={"class": "S_label"})
    return cells, plaques, counts, genes, areas


def composition_points(cells, plaques, areas):
    q = cells.loc[cells.qc_eligible & cells.boundary_distance_um.ge(0) & cells.boundary_distance_um.lt(150)].copy()
    q["display_type"] = q.broad_celltype.where(q.annotation_eligible & q.broad_celltype.isin(NAMED_TYPES), "Other / unresolved")
    counts = q.groupby(["plaque_id", "display_type"], sort=False).size()
    index = pd.MultiIndex.from_product([plaques.plaque_id, DISPLAY_TYPES], names=["plaque_id", "display_type"])
    result = counts.reindex(index, fill_value=0).rename("n_cells").reset_index()
    result["total"] = result.plaque_id.map(q.groupby("plaque_id").size()).fillna(0).astype(int)
    area = areas.set_index("plaque_id")[["area_0_50_um2", "area_50_150_um2"]].sum(axis=1)
    result["area_um2"] = result.plaque_id.map(area)
    if ((result.total > 0) & (result.area_um2 <= 0)).any():
        raise ValueError("Observed exterior cells cannot have zero observed annulus area")
    result["fraction_percent"] = 100 * result.n_cells / result.total.replace(0, np.nan)
    result["density_per_mm2"] = 1e6 * result.n_cells / result.area_um2.replace(0, np.nan)
    result["log_density"] = np.log10(1 + result.density_per_mm2)
    return result.merge(plaques[["plaque_id", "S_label"]], on="plaque_id", validate="many_to_one")
