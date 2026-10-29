"""Shared global UMAP and independent lineage clustering from RNA counts."""

import numpy as np
import pandas as pd


def preprocess(counts, genes, cell_ids, settings):
    import anndata as ad
    import scanpy as sc

    requested = int(settings.get("n_highly_variable_genes", 3000))
    if not 2000 <= requested <= 4000:
        raise ValueError("n_highly_variable_genes must be between 2000 and 4000")
    data = ad.AnnData(counts.astype(np.float32),
                     obs=pd.DataFrame(index=pd.Index(cell_ids.astype(str))),
                     var=pd.DataFrame(index=pd.Index(genes.gene.astype(str))))
    data.layers["counts"] = data.X.copy()
    sc.pp.normalize_total(data, target_sum=float(settings.get("target_sum", 10000)))
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=min(requested, data.n_vars), flavor="seurat_v3", layer="counts")
    data = data[:, data.var.highly_variable.to_numpy()].copy()
    components = min(int(settings.get("pca_components", 50)), data.n_obs - 1, data.n_vars - 1)
    if components < 2:
        raise ValueError("RNA PCA requires at least three cells and three variable genes")
    sc.pp.scale(data, zero_center=True)
    sc.pp.pca(data, n_comps=components, random_state=int(settings.get("seed", 42)), svd_solver="arpack")
    pcs = min(int(settings.get("n_pcs", 30)), components)
    neighbors = min(int(settings.get("umap_neighbors", 30)), data.n_obs - 1)
    if pcs < 1 or neighbors < 2:
        raise ValueError("RNA neighbors require positive PCs and at least two neighbors")
    sc.pp.neighbors(data, n_neighbors=neighbors, n_pcs=pcs, random_state=int(settings.get("seed", 42)))
    return data


def lineage_clusters(counts, genes, cells, settings):
    import scanpy as sc

    grid = [float(value) for value in settings.get("leiden_resolutions", [.1, .2, .3, .4, .5, .6, .8, 1., 1.2])]
    selected = float(settings.get("selected_resolution", .4))
    if not grid or any(not np.isfinite(value) or not .1 <= value <= 1.2 for value in grid) or selected not in grid:
        raise ValueError("Leiden resolutions must lie in [0.1, 1.2] and include selected_resolution")
    result = cells[["cell_id", "broad_celltype"]].reset_index(drop=True).rename(columns={"broad_celltype": "lineage"})
    result["lineage_clustering_status"] = "insufficient_cells_or_unresolved"
    for field in [*(f"lineage_leiden_{value:g}" for value in grid), "lineage_cluster"]:
        result[field] = pd.Series(pd.NA, index=result.index, dtype="string")
    # Four observations support the PCA/neighbour operations; this is not a
    # biological inclusion criterion. Sparse lineages retain external review.
    minimum = int(settings.get("min_lineage_cells", 4))
    if minimum < 4:
        raise ValueError("min_lineage_cells must be at least four for PCA and neighbors")
    for lineage, indices in result.groupby("lineage", sort=False).indices.items():
        if lineage == "Low-confidence / ambiguous" or len(indices) < minimum:
            continue
        lineage_settings = {**settings, **settings.get("lineage_overrides", {}).get(lineage, {})}
        data = preprocess(counts[indices], genes, result.loc[indices, "cell_id"], lineage_settings)
        for resolution in grid:
            field = f"lineage_leiden_{resolution:g}"
            sc.tl.leiden(data, resolution=resolution, random_state=int(settings.get("seed", 42)),
                         key_added=field, flavor="leidenalg")
            result.loc[indices, field] = data.obs[field].astype(str).to_numpy()
        result.loc[indices, "lineage_cluster"] = result.loc[indices, f"lineage_leiden_{selected:g}"]
        result.loc[indices, "lineage_clustering_status"] = "clustered"
    return result


def build_embeddings(counts, genes, cells, settings):
    import scanpy as sc

    data = preprocess(counts, genes, cells.cell_id, settings)
    sc.tl.umap(data, min_dist=float(settings.get("umap_min_dist", .3)),
               random_state=int(settings.get("seed", 42)))
    embedding = pd.DataFrame(data.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    embedding.insert(0, "cell_id", cells.cell_id.to_numpy())
    return embedding, lineage_clusters(counts, genes, cells.reset_index(drop=True), settings)
