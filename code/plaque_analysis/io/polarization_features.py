"""Figure 4: feature exports."""
from __future__ import annotations
from plaque_analysis.common.provenance import code_sha256

from datetime import datetime
from datetime import timezone
from pathlib import Path
from plaque_analysis.features.polarization_table import complete_feature_table
from plaque_analysis.io.polarization import sha256, write_json
from plaque_analysis.plotting.polarization_features import plot_complete_feature_atlas, plot_feature_depth, plot_feature_distributions, plot_feature_qc, plot_feature_scales
from plaque_analysis.plotting.polarization_style import style
from plaque_analysis.common.polarization_version import __version__


def export_complete_features(out, maps, tissue, analysis, pa, visual, features, qc, config,
                             source_meta, angle_rows, threshold_rows, boundary_rows,
                             loo_rows, loo_curves, loo_scales, *, figures=True):
    folder = Path(out)/"features"
    folder.mkdir(parents=True, exist_ok=True)
    table = complete_feature_table(maps, tissue, analysis, pa, visual, features, qc, config, source_meta,
                                   angle_rows, threshold_rows, boundary_rows, loo_rows, loo_curves, loo_scales)
    path = folder/"all_features_complete.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig", na_rep="", float_format="%.15g")
    baseline = table.loc[table.scenario.eq("baseline")]
    report = {"version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
              "sample_id": config["sample_id"], "csv": path.name, "csv_sha256": sha256(path),
              "rows": len(table), "baseline_rows": len(baseline), "scenario_rows": len(table)-len(baseline),
              "curated_scalar_features": int(table.curated.sum()),
              "rows_by_scope": table.groupby("scope", sort=False).size().to_dict(),
              "rows_by_status": table.groupby("status", sort=False).size().to_dict(),
              "source": source_meta, "configuration": {k: v for k, v in config.items() if k != "config_root"},
              "script_sha256": code_sha256(), "plot_backend": "python/matplotlib",
              "plot_data_source": "in-memory complete feature table",
              "biological_n": 1, "sampling": "No downsampling in table statistics or new feature plots.",
              "conditional_policy": "Accepted value is missing; diagnostic_value retains conditional calculation; no zero substitution.",
              "figures": []}
    if figures:
        plot_folder = folder/"visualization"
        plot_folder.mkdir(exist_ok=True)
        style()
        for plot in (plot_complete_feature_atlas, plot_feature_distributions, plot_feature_depth,
                     plot_feature_scales, plot_feature_qc):
            plot(table, plot_folder, config["raster_dpi"])
        report["figures"] = [str(p.relative_to(folder)) for p in sorted(plot_folder.iterdir())
                              if p.suffix in (".svg", ".png")]
    write_json(folder/"complete_features_audit.json", report)
    return report
