"""Orchestrate numerical source preparation and optional figure exports."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from plaque_analysis.common.paths import CONFIG_ROOT
from plaque_analysis.common.output import validate_output_path


def run(data_dir, master_svg, output_dir, *, config_file=None, export=False, replot=False, dpi=300, layout_file=None):
    data_dir, master_svg = Path(data_dir).resolve(), Path(master_svg).resolve()
    output_dir = validate_output_path(output_dir, (data_dir, master_svg.parent, config_file, layout_file))
    from plaque_analysis.analysis.transcriptomics import prepare_analysis
    config_file = config_file or CONFIG_ROOT / "transcriptomics.json"
    layout_file = layout_file or CONFIG_ROOT / "transcriptomics_layout.json"
    report, tables, prepared = prepare_analysis(data_dir, config_file)
    report["software_versions"] = {name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "lxml", "scanpy", "anndata", "scikit-learn", "joblib")}
    output_dir.mkdir(parents=True, exist_ok=True)
    numerical = output_dir/"source_data"
    numerical.mkdir(exist_ok=True)
    for name, table in tables.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        table.to_csv(numerical/name, index=False, compression=compression)
    if export:
        from plaque_analysis.io.transcriptomics_artwork import export_final
        report["final_exports"] = export_final(master_svg, output_dir/"final", layout_file)
    if replot:
        from plaque_analysis.plotting.transcriptomics import replot_panels
        report["regenerated"] = replot_panels(prepared, output_dir, dpi=dpi)
    (output_dir/"analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report
