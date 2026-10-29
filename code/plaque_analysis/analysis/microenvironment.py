"""Combine mouse transcriptomic and human imaging methods."""

from plaque_analysis.analysis.mouse import prepare_mouse
from plaque_analysis.analysis.human import prepare_human


def prepare_tables(inputs):
    tables = prepare_mouse(inputs, inputs["config"])
    human, representative = prepare_human(inputs["data_dir"], inputs["config"].get("human", {}))
    tables.update(human)
    tables["panel_f_image"] = representative
    tables["cell_spatial_assignments"] = inputs["cells"]
    tables["observed_annular_areas"] = inputs["areas"]
    return tables
