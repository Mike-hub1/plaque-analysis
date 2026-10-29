"""Figure 3: redraw."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import DEFAULT_TAU_CHANNEL, FIGURE_FONT, MeasurementConfig, VERSION
from plaque_analysis.io.czi import load_czi_input_channels
from plaque_analysis.plotting.fluorescence_data import load_saved_visual_data
from plaque_analysis.plotting.fluorescence_guides import write_chinese_figure_guides
from plaque_analysis.plotting.fluorescence_layout import figure_bundle_base
from plaque_analysis.plotting.fluorescence_publication import render_publication_figure
from plaque_analysis.plotting.raw_channels import render_raw_channel_figures
from plaque_analysis.plotting.fluorescence_style import configure_plotting
from plaque_analysis.plotting.fluorescence import render_figures
from plaque_analysis.analysis.fluorescence_quality import saturation_qc
from plaque_analysis.io.fluorescence_utils import json_clean, package_source_hashes, sha256_file, write_json
from plaque_analysis.common.output import validate_output_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile


def redraw_existing_figures(args):
    """在既有结果中只重绘图形；不调用分割、强度分层或特征计算函数。"""
    if not args.overwrite:
        raise ValueError("只重绘将替换同名图片，请同时使用 --overwrite；分析文件保持不变。")
    if args.no_figures or args.check_inputs:
        raise ValueError("--redraw-figures 不能与 --no-figures 或 --check-inputs 同用。")
    output = validate_output_path(args.out_dir)
    acquisition = json.loads((output/"acquisition_manifest.json").read_text(encoding="utf-8"))
    validate_output_path(output, (acquisition.get("input_file"),))
    parameters = dict(acquisition["parameters"])
    parameters["texture_offsets"] = tuple(tuple(p) for p in parameters["texture_offsets"])
    parameters["texture_extra_steps"] = tuple(parameters["texture_extra_steps"])
    if parameters["texture_bin_edges"] is not None:
        parameters["texture_bin_edges"] = tuple(parameters["texture_bin_edges"])
    spacing = tuple(acquisition["spacing_yx_um"])
    config = MeasurementConfig(**parameters).resolved(spacing)
    matrix = lambda name: np.asarray(tifffile.imread(output/"matrices"/(name+".tif")))
    raw = [matrix(tag+"_raw_detector") for tag in ("blue_Aβ", "red_LAMP1")]
    signed = [matrix(tag+"_background_subtracted_signed") for tag in ("blue_Aβ", "red_LAMP1")]
    valid = [matrix(tag+"_valid_observed_mask").astype(bool) for tag in ("blue_Aβ", "red_LAMP1")]
    labels, mask_l = matrix("abeta_instance_labels"), matrix("red_LAMP1_foreground_mask").astype(bool)
    quality = pd.read_csv(output/"qc.csv")
    paths = []
    for record in quality.to_dict("records"):
        if str(record["status"]).startswith("excluded"):
            continue
        oid, plaque_id = int(record["object_id"]), record["PlaqueID"]
        mask_a = labels == oid
        window = matrix(f"object_{oid:03d}_analysis_window").astype(bool)
        geometry = {"distance": matrix(f"object_{oid:03d}_signed_boundary_distance_um"), "expected": window,
                    "object_truncated": bool(record["object_truncated"])}
        ranks = [matrix(f"object_{oid:03d}_{tag}_intensity_rank") for tag in ("abeta", "lamp1")]
        strata = [pd.read_csv(output/f"object_{oid:03d}_{tag}_intensity_strata.csv").to_dict("records")
                  for tag in ("abeta", "lamp1")]
        sat = []
        for ci, mask in enumerate((mask_a, mask_l)):
            prior = acquisition["saturation"][ci]
            sat.append(saturation_qc(raw[ci], mask, prior.get("effective_bits")))
        angular = pd.read_csv(output/"angular_profiles.csv")
        angular = angular[angular.PlaqueID == plaque_id].to_dict("records")
        if args.redraw_manuscript_only:
            plt = configure_plotting()
            data = load_saved_visual_data(output, plaque_id, window, mask_l)
            data["strata"] = strata
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in plaque_id)
            plaque_dir = output/"figures"/safe_id
            base = figure_bundle_base(plaque_dir, "manuscript_figure")
            for is_a4 in (False, True):
                render_publication_figure(plt, data, raw, signed, (mask_a, mask_l), valid, geometry,
                    sat, spacing, config, ranks, strata, base.parent/"A4校样" if is_a4 else base,
                    args.dpi, a4=is_a4)
            write_chinese_figure_guides(data, raw, sat, spacing, config, plaque_dir, args.dpi, only_manuscript=True)
            paths.append(str(base)+".png")
        else:
            paths.append(render_figures(raw, signed, (mask_a, mask_l), valid, geometry, [], angular, [],
                {}, {}, ranks, strata, sat, spacing, config, output, plaque_id, args.dpi, preserve_source_data=True))
    if not args.redraw_manuscript_only and not args.skip_raw_channel_figures and acquisition.get("raw_figure_channels"):
        # 原始图使用独立的原生观测范围，不能把组织分析掩膜冒充探测器覆盖范围。
        old_args = acquisition["parser_arguments"]
        source_path = Path(acquisition["input_file"])
        if not source_path.is_file() or sha256_file(source_path) != acquisition["input_sha256"]:
            raise ValueError("原始 CZI 不可用或散列已变化；原始通道图重绘前必须核对来源。")
        display_raw, display_valid, native_spacing, native_acquisition = load_czi_input_channels(source_path,
            (("Aβ/CR", old_args["abeta_channel"], None), ("LAMP1", old_args["lamp1_channel"], None),
             ("tau", old_args.get("tau_channel", DEFAULT_TAU_CHANNEL), None)),
            scene_index=old_args["scene_index"], time_index=old_args["time_index"], z_index=old_args["z_index"],
            spacing_override=spacing)
        render_raw_channel_figures(display_raw, display_valid, native_spacing, native_acquisition["channels"],
            native_acquisition, output, args.dpi,
            effective_bits=tuple(old_args.get(k+"_effective_bits") for k in ("abeta", "lamp1", "tau")),
            preserve_manifest=True)
    record = {"version": VERSION, "script_sha256": sha256_file(__file__), "package_source_sha256": package_source_hashes(),
        "mode": "redraw_manuscript_only" if args.redraw_manuscript_only else "redraw_cached_results_only",
        "source_analysis_manifest": "acquisition_manifest.json",
        "resegmentation": False, "reclustering": False, "feature_recomputation": False,
        "font": FIGURE_FONT, "publication_size_mm": [183, 170], "proof_size_mm": [210, 297]}
    write_json(output/"figure_redraw_manifest.json", record)
    print(json.dumps(json_clean({"status": "completed",
          "font": FIGURE_FONT, "figures": paths}), ensure_ascii=False, indent=2))
    return record

