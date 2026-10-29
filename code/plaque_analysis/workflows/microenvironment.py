"""Execute the method modules and write their numerical and graphical outputs."""

import json
import platform
from pathlib import Path
import pandas as pd
from plaque_analysis.analysis.microenvironment import prepare_tables
from plaque_analysis.io.microenvironment import load_inputs, write_tables
from plaque_analysis.common.paths import CONFIG_ROOT
from plaque_analysis.common.output import validate_output_path


def run(data_dir, output_dir, artwork_dir, *, config_file=None, no_figures=False,
        dpi=300, panels="abcdefgh", export_final=False):
    source, artwork = Path(data_dir).resolve(), Path(artwork_dir).resolve()
    output = validate_output_path(output_dir, (source, artwork, config_file))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose another directory")
    inputs = load_inputs(source, config_file or CONFIG_ROOT / "microenvironment.json")
    tables = prepare_tables(inputs)
    output.mkdir(parents=True, exist_ok=True)
    write_tables(tables, output / "source_data")
    report = {"python": platform.python_version(),
              "table_rows": {name: len(frame) for name, frame in tables.items() if isinstance(frame, pd.DataFrame)},
              "configuration": inputs["config"],
              "classes": "fixed Figure 5 classes supplied by Figure 6; no refitting",
              "ucell": "per-cell gene ranks; arithmetic means within eligible plaque windows",
              "distance_contrast": "far minus near; plaque-weighted mean and spatial-block percentile interval",
              "human": "optical harmonic fits, independent DOA measurements and measured LAMP1 intensity ratios"}
    if not no_figures:
        from plaque_analysis.plotting.microenvironment import render_panels
        render_panels(tables, output, dpi=dpi, panels=panels, assemble=panels == "abcdefgh")
    if export_final:
        from plaque_analysis.io.microenvironment_artwork import copy_final
        report["existing_artwork_copy"] = copy_final(artwork, output).name
    import importlib.metadata
    report["software_versions"] = {name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "matplotlib", "Pillow")}
    (output / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
