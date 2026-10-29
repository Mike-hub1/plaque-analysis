"""Figure 3: definitions."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import TEXTURE_PROPERTIES, VERSION


# Public schema: names, units, roles and interpretation limits only. Exact
# measurement formulas and domains are documented with the licensed core.
FORMULA_REFERENCE = "licensed core feature reference"


FEATURE_DEFINITIONS = {
    "abeta_area_um2": ("μm²", "baseline load"),
    "equivalent_radius_um": ("μm", "deterministic transform of area"),
    "perimeter_um": ("μm", "pixel-cell perimeter, grid dependent"),
    "circularity": ("1", "pixel-cell shape, not molecular packing"),
    "solidity": ("1", "2D contour concavity"),
    "eccentricity": ("1", "ellipse shape"),
    "intensity_centroid_offset_norm": ("1", "saturation sensitive"),
    "intensity_gyration_radius_norm": ("1", "saturation sensitive"),
    "core_edge_signal_enrichment": ("1", "not intensity quantile core"),
    "mean_intensity": ("detector a.u.", "brightness control"),
    "std_intensity": ("detector a.u.", "histogram descriptor"),
    "median_intensity": ("detector a.u.", "histogram descriptor"),
    "iqr_over_median": ("1", "histogram, not spatial arrangement"),
    "cv": ("1", "histogram descriptor; fraction not percent"),
    "skewness": ("1", "undefined constant signal"),
    "kurtosis_excess": ("1", "undefined constant signal"),
    "integrated_signal": ("detector a.u.·pixel", "area-collinear candidate"),
    "glcm_contrast": ("level²", "ordered relative gray levels by default"),
    "glcm_homogeneity": ("1", "same quantization"),
    "glcm_entropy": ("bit", "not entropy of pooled directions"),
    "glcm_ASM": ("1", "not Energy"),
    "glcm_energy": ("1", "redundant with ASM"),
    "glcm_correlation": ("1", "undefined zero variance"),
    "lamp1_area_um2": ("μm²", "primary target; Fig2 definition compatibility pending"),
    "log1p_lamp1_area_um2": ("1", "monotone transform of primary target"),
    "lamp1_inside_area_um2": ("μm²", "part of total area"),
    "lamp1_outside_area_um2": ("μm²", "part of total area"),
    "lamp1_outside_d50_um": ("μm", "not invasion depth"),
    "lamp1_outside_d90_um": ("μm", "not full halo beyond fixed window"),
    "lamp1_window_mean_signed_intensity": ("detector a.u.", "not cross-channel protein concentration"),
    "angular_coverage": ("1", "empty/unobserved sectors are missing"),
    "angular_asymmetry": ("1", "not migration/force direction"),
    "lamp1_area_observed_um2": ("μm²", "diagnostic; not a replacement for complete primary target"),
    "window_observed_area_um2": ("μm²", "QC"),
    "window_expected_area_um2": ("μm²", "QC; no extrapolated pathology"),
    "observable_window_fraction": ("1", "QC"),
    "window_complete": ("boolean", "QC"),
    "pearson_window": ("1", "joint descriptor, not Aβ-only input"),
    "pearson_double_positive_conditional": ("1", "conditional diagnostic"),
    "pearson_foreground_union": ("1", "distinct analysis domain"),
    "manders_A_in_L": ("1", "intensity-weighted overlap"),
    "manders_L_in_A": ("1", "intensity-weighted overlap"),
    "dice": ("1", "deterministically redundant with Jaccard; not reference segmentation accuracy"),
    "jaccard": ("1", "deterministically redundant with Dice"),
    "lamp1_centroid_offset_um": ("μm", "joint position"),
    "lamp1_centroid_offset_norm": ("1", "normalised joint position"),
}


ROLE_FEATURES = {
    "X_A": tuple(list(FEATURE_DEFINITIONS)[:list(FEATURE_DEFINITIONS).index("lamp1_area_um2")]),
    "Y_L": tuple(list(FEATURE_DEFINITIONS)[list(FEATURE_DEFINITIONS).index("lamp1_area_um2"):
                                          list(FEATURE_DEFINITIONS).index("pearson_window")]),
    "Z_AL": tuple(list(FEATURE_DEFINITIONS)[list(FEATURE_DEFINITIONS).index("pearson_window"):]),
}


EXTRA_FEATURES = {
    # label_zh, unit, interpretation limit
    "X_A": {
        "shape_topology": {
            "major_axis_length_um": ("等效椭圆长轴", "μm", "second-moment equivalent ellipse, not longest chord"),
            "minor_axis_length_um": ("等效椭圆短轴", "μm", "second-moment equivalent ellipse"),
            "axis_ratio_major_minor": ("长短轴比", "1", "redundant with eccentricity; not molecular anisotropy"),
            "bounding_box_extent": ("包围框填充率", "1", "rotation/grid dependent"),
            "mask_component_count_8": ("Aβ 掩膜八连通域数", "count", "normally 1; checks disconnected instance labels, not animal count"),
            "mask_hole_count_4": ("掩膜内部孔隙数", "count", "threshold/grid holes, not physical pores"),
            "mask_hole_area_fraction": ("掩膜孔隙面积比例", "1", "2D mask voids, not porosity at molecular scale"),
            "euler_number_8": ("八连通欧拉数", "count", "redundant with component and hole count; grid dependent"),
            "geometric_gyration_radius_norm": ("归一化几何回转半径", "1", "shape dispersion relative to area"),
            "mean_inward_distance_um": ("平均内部边界距离", "μm", "boundary depth, not 3D thickness"),
            "max_inward_distance_um": ("最大采样内部边界距离", "μm", "centre-sampled maximum, not subpixel maximum-inscribed radius"),
            "eroded_area_fraction": ("几何内区保留比例", "1", "depends on specified μm edge width, not intensity quantile"),
        },
        "intensity_distribution": {
            "intensity_p10": ("Aβ 强度第10百分位", "detector a.u.", "histogram, saturation sensitive"),
            "intensity_p90": ("Aβ 强度第90百分位", "detector a.u.", "histogram, saturation sensitive"),
            "mad_over_median": ("中位绝对偏差比", "1", "unscaled MAD; histogram not spatial arrangement"),
            "positive_signal_gini": ("非负信号 Gini 系数", "1", "concentration inequality, no spatial-location information"),
            "effective_signal_pixel_fraction": ("有效信号参与像素比例", "1", "inverse participation fraction, not signal-positive area fraction"),
            "brightest_decile_signal_fraction": ("最亮10%像素信号份额", "1", "signal share, NOT a quantile-defined core area"),
        },
        "internal_organization": {
            "signal_weighted_depth_norm": ("信号加权归一化内部深度", "1", "geometric depth-weighted intensity, no invasion claim"),
            "intensity_depth_spearman": ("强度与内部深度秩相关", "1", "descriptive coefficient only, no pixel-independent p-value"),
            "core_rim_signal_contrast": ("几何核心边缘强度对比", "1", "redundant with core_edge_signal_enrichment; not proof of core-halo mechanism"),
            "geometric_core_signal_fraction": ("几何核心非负信号份额", "1", "geometrically defined core, saturation sensitive"),
            "interior_gradient_rms_norm": ("归一化内部梯度均方根", "1", "no smoothing; image gradient, not molecular arrangement"),
            "interior_gradient_coherence": ("内部梯度方向一致性", "1", "global gradient tensor; undefined for zero gradient; not birefringence"),
        },
        "multiscale_texture": {
            "glcm_lag2_contrast": ("2像素位移纹理对比度", "level²", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
            "glcm_lag2_homogeneity": ("2像素位移纹理同质性", "1", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
            "glcm_lag2_entropy": ("2像素位移纹理熵", "bit", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
            "glcm_lag3_contrast": ("3像素位移纹理对比度", "level²", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
            "glcm_lag3_homogeneity": ("3像素位移纹理同质性", "1", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
            "glcm_lag3_entropy": ("3像素位移纹理熵", "bit", "no resampling; actual dx/dy/distance and pair count in glcm_directions.csv"),
        },
    },
    "Y_L": {
        "boundary_burden": {
            "lamp1_window_positive_fraction": ("固定窗 LAMP1 阳性比例", "1", "area-normalized target, window-dependent"),
            "lamp1_inside_density": ("斑块内 LAMP1 密度", "1", "within-mask density, not protein concentration"),
            "lamp1_outside_density": ("斑块外 LAMP1 密度", "1", "fixed-window exterior, not full halo"),
            "lamp1_outside_area_fraction": ("LAMP1 斑块外面积份额", "1", "complement of lamp1_area_inside_abeta_fraction"),
            "lamp1_outside_mean_distance_um": ("外部 LAMP1 平均边界距离", "μm", "not full halo extent"),
            "lamp1_outside_distance_sd_um": ("外部 LAMP1 距离标准差", "μm", "spatial dispersion, not measurement uncertainty"),
            "lamp1_near_boundary_density": ("近边界带 LAMP1 密度", "1", "coarse band, not physical contact"),
            "lamp1_far_window_density": ("外缘带 LAMP1 密度", "1", "far band is tied to the prespecified window"),
            "lamp1_near_minus_far_density": ("近远边界密度差", "1", "descriptive contrast, not diffusion or leakage"),
            "lamp1_outermost_band_area_fraction": ("外缘带 LAMP1 面积份额", "1", "warns of boundary accumulation; cannot recover unobserved halo"),
            "lamp1_geometric_core_density": ("Aβ 几何核心内 LAMP1 密度", "1", "Aβ-only geometric core; not strict molecular exclusion"),
            "lamp1_geometric_rim_density": ("Aβ 几何边缘内 LAMP1 密度", "1", "coarse boundary-relative occupancy"),
            "lamp1_core_minus_rim_density": ("核心边缘 LAMP1 密度差", "1", "static pattern, no physical repulsion inference"),
        },
        "angular_organization": {
            "lamp1_angular_entropy_norm": ("角向密度归一化熵", "1", "equal-sector density normalization, not raw signal count"),
            "lamp1_angular_density_cv": ("角向密度变异系数", "1", "angular nonuniformity, not repeatability CV"),
            "lamp1_angular_second_harmonic": ("角向双极分布幅度", "1", "descriptive twofold organization, grid/orientation sensitive"),
        },
        "signal_patch_topology": {
            "lamp1_window_patch_count_8": ("窗内八连通信号斑片数", "count", "window-clipped image patches, NOT lysosome/spheroid counts"),
            "lamp1_window_patch_count_4": ("窗内四连通信号斑片数", "count", "compare with 8-connectivity for grid sensitivity; not independent objects"),
            "lamp1_patch_density_per_1000_um2": ("每千平方微米斑片数", "count/(1000 μm²)", "threshold/grid/window-dependent image fragmentation"),
            "lamp1_largest_patch_area_fraction": ("最大信号斑片面积份额", "1", "dominance within window; does not identify a single axonal swelling"),
            "lamp1_patch_median_area_um2": ("信号斑片面积中位数", "μm²", "window clipping can split a connected biological structure"),
            "lamp1_patch_p90_area_um2": ("信号斑片面积第90百分位", "μm²", "not organelle size; unstable with few patches"),
            "lamp1_single_pixel_patch_area_fraction": ("单像素斑片面积份额", "1", "sampling/segmentation warning, not isolated lysosome fraction"),
        },
        "lamp1_intensity": {
            "lamp1_positive_mean_intensity": ("LAMP1 阳性区域平均强度", "detector a.u.", "conditional intensity, cannot compare raw a.u. across channels"),
            "lamp1_positive_median_intensity": ("LAMP1 阳性区域强度中位数", "detector a.u.", "saturation sensitive"),
            "lamp1_positive_intensity_cv": ("LAMP1 阳性区域强度变异系数", "1", "histogram variability, not biological-replicate CV"),
            "lamp1_positive_intensity_p90": ("LAMP1 阳性区域强度第90百分位", "detector a.u.", "saturation sensitive"),
            "lamp1_window_integrated_signed_signal_um2": ("LAMP1 窗内面积积分信号", "detector a.u.·μm²", "physical-area integral, not absolute protein amount"),
        },
    },
    "Z_AL": {
        "joint_geometry": {
            "abeta_area_covered_by_lamp1_fraction": ("Aβ 面积被 LAMP1 覆盖的比例", "1", "binary area overlap, not intensity-weighted Manders"),
            "lamp1_area_inside_abeta_fraction": ("LAMP1 面积位于 Aβ 内的比例", "1", "complement of outside_area_fraction; joint descriptor"),
            "lamp1_to_abeta_area_ratio": ("LAMP1 与 Aβ 面积比", "1", "contains the LAMP1 target; never an Aβ-only input"),
            "binary_phi_window": ("固定窗二值 Phi 相关", "1", "2x2 occupancy association; no independent-pixel significance test"),
            "lamp1_d50_over_abeta_radius": ("归一化 LAMP1 外部距离中位数", "1", "joint scale ratio, not penetration depth"),
        },
        "joint_intensity": {
            "spearman_window": ("固定窗强度 Spearman 相关", "1", "descriptive monotone dependence, no pixel-level p-value"),
            "spatial_signal_hellinger_distance": ("信号空间分布 Hellinger 距离", "1", "each channel normalized separately; 0 same mass distribution, 1 disjoint; not protein concentration or mechanism"),
        },
    },
}


FEATURE_LABELS_ZH = {
    "abeta_area_um2": "Aβ 面积", "equivalent_radius_um": "等效圆半径", "perimeter_um": "像素单元边缘周长",
    "circularity": "圆度", "solidity": "实心度", "eccentricity": "偏心率",
    "intensity_centroid_offset_norm": "归一化强度质心偏移", "intensity_gyration_radius_norm": "归一化强度回转半径",
    "core_edge_signal_enrichment": "几何核心边缘信号富集比", "mean_intensity": "Aβ 平均强度",
    "std_intensity": "Aβ 强度标准差", "median_intensity": "Aβ 强度中位数", "iqr_over_median": "四分位距中位数比",
    "cv": "Aβ 强度变异系数", "skewness": "Aβ 强度偏度", "kurtosis_excess": "Aβ 强度超额峰度",
    "integrated_signal": "Aβ 像素积分信号", "lamp1_area_um2": "完整固定窗 LAMP1 面积",
    "log1p_lamp1_area_um2": "LAMP1 面积 log1p", "lamp1_inside_area_um2": "斑块内 LAMP1 面积",
    "lamp1_outside_area_um2": "斑块外 LAMP1 面积", "lamp1_outside_d50_um": "外部 LAMP1 距离中位数",
    "lamp1_outside_d90_um": "外部 LAMP1 距离第90百分位", "lamp1_window_mean_signed_intensity": "窗内 LAMP1 带符号平均强度",
    "angular_coverage": "角向覆盖率", "angular_asymmetry": "角向不对称性",
    "lamp1_area_observed_um2": "已观测部分 LAMP1 面积", "window_observed_area_um2": "已观测窗口面积",
    "window_expected_area_um2": "理论归属窗口面积", "observable_window_fraction": "窗口可观测比例", "window_complete": "窗口完整标记",
    "pearson_window": "固定窗 Pearson 相关", "pearson_double_positive_conditional": "双阳性条件 Pearson 相关",
    "pearson_foreground_union": "前景并集 Pearson 相关", "manders_A_in_L": "Aβ 信号在 LAMP1 内的 Manders 系数",
    "manders_L_in_A": "LAMP1 信号在 Aβ 内的 Manders 系数", "dice": "两通道掩膜 Dice", "jaccard": "两通道掩膜 Jaccard",
    "lamp1_centroid_offset_um": "LAMP1 相对 Aβ 几何质心位移", "lamp1_centroid_offset_norm": "归一化 LAMP1 质心位移",
    **{"glcm_"+p: "一像素位移 GLCM "+p for p in TEXTURE_PROPERTIES},
}


EXTRA_FEATURE_METADATA = {}


for _role, _families in EXTRA_FEATURES.items():
    for _family, _entries in _families.items():
        for _name, (_zh, _unit, _limit) in _entries.items():
            if _name in FEATURE_DEFINITIONS:
                raise RuntimeError(f"Duplicate feature definition: {_name}")
            FEATURE_DEFINITIONS[_name] = (_unit, _limit)
            FEATURE_LABELS_ZH[_name] = _zh
            EXTRA_FEATURE_METADATA[_name] = {"role": _role, "family": _family}
    ROLE_FEATURES[_role] += tuple(name for entries in _families.values() for name in entries)


LAMP1_INTENSITY_FEATURES = {"lamp1_window_mean_signed_intensity", *EXTRA_FEATURES["Y_L"]["lamp1_intensity"]}


JOINT_INTENSITY_FEATURES = {"pearson_window", "pearson_foreground_union", "pearson_double_positive_conditional",
                          "manders_A_in_L", "manders_L_in_A", *EXTRA_FEATURES["Z_AL"]["joint_intensity"]}


WINDOW_QC_FEATURES = {"lamp1_area_observed_um2", "window_observed_area_um2", "window_expected_area_um2",
                      "observable_window_fraction", "window_complete"}


def feature_dictionary(tables, config):
    rows = []
    function = {"X_A": "compute_abeta_features", "Y_L": "compute_lamp1_outcomes", "Z_AL": "compute_colocalization"}
    for role, records in tables.items():
        columns = ROLE_FEATURES[role]
        for name in columns:
            units, note = FEATURE_DEFINITIONS[name]
            family = EXTRA_FEATURE_METADATA.get(name, {}).get("family")
            if family is None:
                family = ("window_qc" if name in WINDOW_QC_FEATURES else
                          "texture" if name.startswith("glcm_") else
                          "shape" if name in ROLE_FEATURES["X_A"][:6] else
                          "intensity" if role == "X_A" else
                          "boundary_burden" if role == "Y_L" else "joint_relationship")
            source = ("https://scikit-image.org/docs/stable/api/skimage.measure.html" if family in ("shape", "shape_topology", "signal_patch_topology") else
                      "https://scikit-image.org/docs/stable/api/skimage.feature.html" if "texture" in family or family == "internal_organization" else
                      "https://ibsi.readthedocs.io/en/latest/03_Image_features.html" if "intensity" in family else "")
            implementation = function[role]
            if name in EXTRA_FEATURE_METADATA:
                implementation = ({"shape_topology": "extended_abeta_geometry",
                    "intensity_distribution": "extended_abeta_intensity", "internal_organization": "extended_abeta_intensity",
                    "multiscale_texture": "multiscale_abeta_texture"}[family] if role == "X_A" else
                    "extended_lamp1_features" if role == "Y_L" else "extended_joint_features")
            if name in ("lamp1_centroid_offset_um", "lamp1_centroid_offset_norm"):
                implementation = "analyse_arrays"
            if name in ("spearman_window", "intensity_depth_spearman"):
                source = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"
            elif name == "spatial_signal_hellinger_distance":
                source = ""  # Project-defined probability distance, not an IBSI feature.
            rows.append({"feature": name, "role": role, "formula": FORMULA_REFERENCE, "units": units, "mask": FORMULA_REFERENCE,
                         "feature_name_zh": FEATURE_LABELS_ZH[name], "family": family,
                         "measurement_kind": "QC_or_observed_diagnostic" if name in WINDOW_QC_FEATURES else "phenotype",
                         "implementation": implementation, "version": VERSION, "interpretation_redundancy": note,
                         "scale": "per-offset physical dx,dy for GLCM; calibrated μm for geometry",
                         "thresholds": "per-channel preprocessing metadata; independent of LAMP1 outcome for X_A",
                         "missing_conditions": "see feature_qc: saturation, unknown bit depth, insufficient pixels/pairs, truncation or undefined denominator",
                         "cross_sample_fit": "none; relative per-object bins or pre-supplied fixed bins",
                         "source_url": source,
                         "source_scope": "general method reference; exact project formula in the licensed core reference, not an IBSI-certified implementation",
                         "model_use": "Aβ-only candidate; validate stability and remove redundancy in training folds" if role == "X_A" else
                                      "outcome_or_QC_not_input_to_its_own_prediction" if role == "Y_L" else "joint_descriptor_excluded_from_Aβ_only_prediction",
                         "parameter_status": "user_frozen" if config.parameters_frozen else "provisional"})
    return rows

