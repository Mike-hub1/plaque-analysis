"""Record identity, missing-value policy, and provenance for complete features."""
import hashlib
import json
import numpy as np
from plaque_analysis.common.provenance import code_sha256
from plaque_analysis.common.polarization_version import __version__
from plaque_analysis.imaging.orientation import distribution
from plaque_analysis.features.polarization_schema import STATISTIC_DEFINITIONS


def record_writer(config, source_meta, analysis):
    rows, identities = [], set()
    clean_config = {k: v for k, v in config.items() if k != "config_root"}
    config_hash = hashlib.sha256(json.dumps(clean_config, sort_keys=True).encode()).hexdigest()
    source_hash = hashlib.sha256(json.dumps({"frames": source_meta["frame_sources"],
                                             "pa": source_meta["pa_source_sha256"]}, sort_keys=True).encode()).hexdigest()
    common = {"sample_id": config["sample_id"], "biological_n": 1,
              "source_sha256": source_hash, "configuration_sha256": config_hash,
              "script_sha256": code_sha256(), "analysis_version": __version__,
              "optical_pixel_size_um": config["pixel_size_um"]}
    measurement_definitions = {
        "M2": "sqrt(A2²+B2²)/A0 from I(theta)=A0+A2 cos(2theta)+B2 sin(2theta)+A4 cos(4theta)+B4 sin(4theta).",
        "M4_reference": "sqrt(A4²+B4²)/A0; optical harmonic reference, not calibrated linear dichroism or PA DOA.",
        "local_S": "abs(mean(exp(2i phi))) over qualified directions in the supported square window.",
        "angular_spread_deg": "degrees(sqrt(-2 ln(S_local))/2), defined for S_local>0.",
        "PA_DOA": "Independent native PA DOA measurement supplied in the raw scan matrix; dimensionless.",
        "PA_pair_abs_difference": "abs(DOA_i-DOA_j) for each unique finite horizontal/vertical native pair.",
        "PA_point_neighbour_mean": "At each PA point, arithmetic mean of abs(DOA_i-DOA_j) over its finite horizontal/vertical neighbours.",
        "OPD_conditional": "arcsin(M2)*wavelength_nm/(2*pi), for M2 in [0,1]; conditional low-retardance single-retarder branch.",
        "delta_n_conditional": "OPD_conditional_nm/(section_thickness_um*1000); effective conditional value, not intrinsic fibre birefringence.",
    }

    def add(feature, label, value, unit="1", *, family, metric=None, statistic="value",
            scope="global", support_n=0, sampling_unit="optical_pixel", role="descriptor",
            scenario="baseline", status="available", reason="", definition="", source_data="",
            source_field="", curated=False, default_model_candidate=False, feature_name_zh="",
            notes="", **dimensions):
        metric = metric or feature
        if metric.startswith("PA_"):
            population_rule = "All finite native PA observations; zero values retained; no optical mask or interpolation."
        elif metric in ("local_S", "angular_spread_deg"):
            population_rule = (f"Qualified centre and at least {config['minimum_window_coverage']:.0%} qualified directions in the full square window; "
                               f"default width {analysis['window_actual_um']:.6g} µm. Finite values only.")
        else:
            population_rule = "See the definition and source mask; finite supported observations, with zero values retained."
        if scope in ("depth", "depth_geometry", "depth_azimuth"):
            population_rule += " Intersect with the specified geometry-defined depth/azimuth partition."
        if scope == "scale":
            population_rule += " Replace default width by window_um; retain the intersection of supported centres across all six windows."
        identity_dimensions = {k: v for k, v in dimensions.items() if k != "bin_count"}
        identity = (scope, metric, statistic, scenario, tuple(sorted(identity_dimensions.items())))
        if identity in identities:
            return
        identities.add(identity)
        diagnostic = float(value) if value is not None else np.nan
        if np.isinf(diagnostic):
            raise ValueError(f"Infinite feature: {identity}")
        accepted = diagnostic
        if status in ("conditional_uncalibrated", "insufficient_support"):
            accepted = np.nan
        if not np.isfinite(accepted) and not reason:
            reason = "No finite supported observations or a mathematically undefined statistic."
        if not np.isfinite(diagnostic) and status == "available":
            status = "undefined"
        code = ":".join([scope, scenario, metric, statistic] +
                        [f"{k}={v}" for k, v in sorted(identity_dimensions.items())])
        rows.append({**common, "record_id": code, "scope": scope, "role": role, "family": family,
                     "feature": feature, "metric": metric, "statistic": statistic, "label": label,
                     "feature_name_zh": feature_name_zh or label, "value": accepted,
                     "diagnostic_value": diagnostic, "unit": unit, "support_n": int(support_n),
                     "sampling_unit": sampling_unit, "scenario": scenario, "status": status,
                     "reason": reason, "curated": curated,
                     "default_model_candidate": bool(default_model_candidate and np.isfinite(accepted)),
                     "definition": definition, "source_data": source_data, "source_field": source_field,
                     "measurement_definition": measurement_definitions.get(metric, definition),
                     "population_rule": population_rule, "notes": notes, **dimensions})
    return rows, add


def distribution_writer(add):
    def add_distribution(metric, label, values=None, summary=None, unit="1", family="Optical response",
                         feature_name_zh="", **options):
        if values is not None:
            x = np.asarray(values, float)
            x = x[np.isfinite(x)]
            summary = distribution(x) | {"min": float(x.min()) if x.size else np.nan,
                                          "max": float(x.max()) if x.size else np.nan}
        n = summary["n"]
        for stat, (zh, formula) in STATISTIC_DEFINITIONS.items():
            if stat not in summary:
                continue
            add(f"{metric}_{stat}", f"{label}: {stat}", summary[stat],
                "1" if stat == "quartile_dispersion" else unit, family=family,
                metric=metric, statistic=stat, support_n=n, definition=formula,
                feature_name_zh=f"{feature_name_zh or label}：{zh}", **options)
    return add_distribution
