"""Figure 4: pipeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from plaque_analysis.analysis.polarization import analyse_optical, feature_records
from plaque_analysis.io.polarization_artwork import apply_pa_field_calibration, load_final_artwork
from plaque_analysis.io.polarization_tables import write_sources, write_visual_sources
from plaque_analysis.io.polarization_features import export_complete_features
from plaque_analysis.io.polarization import read_acquisition, resolve, validate_input_paths, write_csv, write_json
from plaque_analysis.imaging.photoacoustic import display_pa
from plaque_analysis.plotting.polarization import plot_visual_story
from plaque_analysis.plotting.polarization_quality import plot_qc
from plaque_analysis.imaging.reconstruction import fit_harmonics, tissue_mask
from plaque_analysis.io.polarization_report import build_analysis_report
from plaque_analysis.analysis.polarization_sensitivity import angle_validation, boundary_sensitivity, leave_angle_sensitivity, threshold_sensitivity
from plaque_analysis.analysis.polarization_visual import extract_visual_features
from plaque_analysis.common.output import validate_output_path


def run_pipeline(config, out, *, no_figures=False, features_only=False):
    """Run the preserved Figure 4 workflow into an explicitly selected directory."""
    validate_input_paths(config)
    protected = [resolve(config, key) for key in ("raw_image_dir", "pa_csv", "mask_npy", "final_svg")
                 if config.get(key)]
    out = validate_output_path(out, protected)
    out.mkdir(exist_ok=True, parents=True)
    qa_dir = out/"quality_control"
    artwork = None
    if not features_only:
        qa_dir.mkdir(exist_ok=True)
        if not no_figures:
            artwork = load_final_artwork(config, out)
        apply_pa_field_calibration(config, artwork)
    print("[1/5] 读取11张原生图像、元数据和光声DOA…", flush=True)
    stack, pa, source_meta = read_acquisition(config)
    maps, coefficients, heldout4 = fit_harmonics(stack, config["angles_deg"],
                                                phi_sign=config["phi_sign"], phi_offset_deg=config["phi_offset_deg"])
    _, _, heldout2 = fit_harmonics(stack, config["angles_deg"], fourth=False)
    if config.get("mask_npy"):
        tissue = np.load(resolve(config, "mask_npy"))
        if tissue.shape != stack.shape[1:] or not np.isin(tissue, [0, 1]).all():
            raise ValueError("Provided mask must match the native ROI and contain only 0/1 values.")
        tissue = tissue.astype(bool)
        mask_record = {"method": "user-supplied native ROI mask", "threshold": None}
    else:
        tissue, threshold = tissue_mask(stack.mean(axis=0))
        mask_record = {"method": "A0 Otsu inverse threshold; largest connected component; hole filling",
                       "threshold_gray": threshold}
    if not tissue.any():
        raise ValueError("Tissue mask is empty.")
    print("[2/5] 计算方向场、组织异质性和空间相关…", flush=True)
    analysis = analyse_optical(maps, tissue, config)
    features, qc = feature_records(analysis, pa, maps, tissue, config)
    angle_rows = angle_validation(heldout2, heldout4, maps["A0"], tissue, config["angles_deg"])
    threshold_rows = threshold_sensitivity(maps, tissue, config)
    pa_display, pa_bbox = display_pa(pa, config["pa_display_rot90"], config["pa_display_fliplr"])
    visual = extract_visual_features(maps, tissue, analysis, pa_display, config)
    print("[3/5] 留一角度重拟合与物理特征敏感性（11次）…", flush=True)
    loo_rows, loo_curves, loo_scales = leave_angle_sensitivity(
        stack, config["angles_deg"], tissue, config, analysis,
        progress=lambda k, n: print(f"  已完成 {k}/{n}", flush=True) if k in (1, 6, 11) else None)
    print("[4/5] 汇总并导出完整物理特征…", flush=True)
    boundary_rows = boundary_sensitivity(maps, tissue, config)
    complete_report = export_complete_features(
        out, maps, tissue, analysis, pa, visual, features, qc, config, source_meta,
        angle_rows, threshold_rows, boundary_rows, loo_rows, loo_curves, loo_scales,
        figures=not no_figures)
    if features_only:
        complete_report["run_mode"] = "features_only"
        write_json(out/"features/complete_features_audit.json", complete_report)
        print(f"[5/5] 完成：{complete_report['rows']:,}行完整特征。", flush=True)
        print(f"输出：{out/'features/all_features_complete.csv'}", flush=True)
        return complete_report
    write_sources(out, stack, maps, coefficients, tissue, analysis, pa, pa_display,
                  features, angle_rows, threshold_rows, loo_rows, loo_curves, loo_scales, config)
    write_visual_sources(out, visual, analysis, pa_display)
    write_csv(out/"source_data/boundary_mask_sensitivity.csv", boundary_rows)
    resolved_config = {k: v for k, v in config.items() if k != "config_root"}
    for key in ("raw_image_dir", "pa_csv", "mask_npy", "final_svg"):
        if config.get(key):
            resolved_config[key] = str(resolve(config, key).resolve())
    resolved_config["output_dir"] = str(out.resolve())
    write_json(qa_dir/"resolved_config.json", resolved_config)
    report = build_analysis_report(
        config, out, source_meta, mask_record, qc, visual, analysis, tissue, pa,
        pa_bbox, stack, maps, features, complete_report, resolved_config,
        artwork_used=artwork is not None)
    write_json(qa_dir/"analysis_report.json", report)
    if not no_figures:
        print("[5/5] 绘制主图、a–i 带文字/无文字子图及补充质量控制图，导出 SVG/PNG…", flush=True)
        plot_visual_story(out, maps, tissue, analysis, pa_display, config, visual, artwork)
        plot_qc(qa_dir, maps, tissue, analysis, config, angle_rows, threshold_rows, loo_rows)
    print(f"完成：{out}", flush=True)
    print(f"组织像素 {tissue.sum():,}；方向有效 {analysis['valid_orientation'].sum():,}；PA格点 {np.isfinite(pa).sum()}。", flush=True)
    return report
