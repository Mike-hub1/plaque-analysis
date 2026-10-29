"""Stable feature labels, definitions, and curated metric identities."""

FEATURE_NAMES_ZH = {
    "optical_m2_median": "二阶调制幅值中位数", "optical_m2_quartile_dispersion": "二阶调制幅值四分位离散系数",
    "optical_m2_core_rim_ratio": "二阶调制幅值深部/边缘比", "orientation_S_global": "全局轴向有序度",
    "orientation_entropy": "归一化轴向角熵", "orientation_local_S_median": "局部轴向有序度中位数",
    "orientation_local_S_iqr": "局部轴向有序度四分位距", "orientation_local_spread_median_deg": "局部角度离散度中位数",
    "orientation_C2_at_1um": "1微米间距的轴向相关", "orientation_half_decay_um": "连通轴向相关半衰减距离",
    "orientation_centroid_texture_coherence": "质心参照取向纹理一致性", "orientation_mean_axis_deg": "平均面内主轴参考角",
    "optical_M4_reference_median": "四阶调制参考值中位数", "conditional_OPD_median_nm": "条件性光程差中位数",
    "conditional_delta_n_median": "条件性有效双折射中位数", "morphology_area_um2": "二维斑块掩膜面积",
    "morphology_eccentricity": "二维掩膜偏心率", "morphology_solidity": "二维掩膜实心度",
    "morphology_perimeter_um": "二维掩膜Crofton周长", "pa_doa_median": "原生光声DOA中位数",
    "pa_doa_quartile_dispersion": "光声DOA四分位离散系数", "pa_doa_iqr": "光声DOA四分位距",
    "pa_doa_q95": "光声DOA第95百分位", "pa_adjacent_abs_difference_mean": "不重复邻接对DOA绝对差均值",
}

FEATURE_METRICS = {
    "optical_m2_median": ("M2", "median"), "optical_m2_quartile_dispersion": ("M2", "quartile_dispersion"),
    "orientation_local_S_median": ("local_S", "median"), "orientation_local_S_iqr": ("local_S", "iqr"),
    "orientation_local_spread_median_deg": ("angular_spread_deg", "median"),
    "optical_M4_reference_median": ("M4_reference", "median"),
    "conditional_OPD_median_nm": ("OPD_conditional", "median"),
    "conditional_delta_n_median": ("delta_n_conditional", "median"),
    "pa_doa_median": ("PA_DOA", "median"), "pa_doa_quartile_dispersion": ("PA_DOA", "quartile_dispersion"),
    "pa_doa_iqr": ("PA_DOA", "iqr"), "pa_doa_q95": ("PA_DOA", "q95"),
    "pa_adjacent_abs_difference_mean": ("PA_pair_abs_difference", "mean"),
}

STATISTIC_DEFINITIONS = {
    "mean": ("均值", "Arithmetic mean of all finite supported observations."),
    "median": ("中位数", "50th percentile; NumPy linear quantile interpolation."),
    "q05": ("第5百分位", "5th percentile; NumPy linear quantile interpolation."),
    "q25": ("第25百分位", "25th percentile; NumPy linear quantile interpolation."),
    "q75": ("第75百分位", "75th percentile; NumPy linear quantile interpolation."),
    "q95": ("第95百分位", "95th percentile; NumPy linear quantile interpolation."),
    "std": ("空间标准差", "Population standard deviation (ddof=0); spatial dispersion, not a standard error."),
    "iqr": ("四分位距", "75th minus 25th percentile."),
    "quartile_dispersion": ("四分位离散系数", "(Q75-Q25)/(Q75+Q25); undefined when the denominator is zero."),
    "min": ("最小值", "Minimum of all finite supported observations."),
    "max": ("最大值", "Maximum of all finite supported observations."),
}
