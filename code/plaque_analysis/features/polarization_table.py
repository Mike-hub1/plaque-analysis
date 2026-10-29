"""Complete feature table orchestration and table selection helpers."""
import numpy as np
import pandas as pd
from plaque_analysis.features.polarization_records import record_writer, distribution_writer
from plaque_analysis.features.polarization_groups import add_curated, add_global_distributions, add_spatial_profiles, add_correlations, add_histograms, add_quality, add_technical_scenarios


def complete_feature_table(maps, tissue, analysis, pa, visual, features, qc, config,
                           source_meta, angle_rows, threshold_rows, boundary_rows,
                           loo_rows, loo_curves, loo_scales):
    """Long-form measurements, explicit populations and technical scenarios.

    A repeated summary already present among the 24 curated features is kept once.
    Uncalibrated conversions have an empty accepted value and a diagnostic value.
    No scalar fields are generated from display sampling, KDE vertices or artwork.
    """
    rows, add = record_writer(config, source_meta, analysis)
    add_distribution = distribution_writer(add)
    add_curated(add, features)
    add_global_distributions(add, add_distribution, maps, tissue, analysis, pa, visual, config)
    add_spatial_profiles(add, add_distribution, analysis, visual, config)
    add_correlations(add, analysis, loo_rows, loo_curves)
    add_histograms(add, analysis, visual)
    add_quality(add, add_distribution, qc, maps, tissue)
    add_technical_scenarios(add, features, analysis, loo_rows, loo_scales, threshold_rows, boundary_rows, angle_rows)
    return pd.DataFrame(rows)


def feature_slice(table, scope=None, metric=None, statistic=None, scenario="baseline"):
    keep = np.ones(len(table), bool)
    for key, value in (("scope", scope), ("metric", metric), ("statistic", statistic), ("scenario", scenario)):
        if value is not None:
            keep &= table[key].eq(value).to_numpy()
    return table.loc[keep]


def feature_value(table, scope, metric, statistic="value", **dimensions):
    selected = feature_slice(table, scope, metric, statistic)
    for key, value in dimensions.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one feature record: {scope}/{metric}/{statistic}/{dimensions}; got {len(selected)}")
    return float(selected.iloc[0].value)
