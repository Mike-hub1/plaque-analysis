"""Portable command line; help and configuration templates use only the standard library."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from plaque_analysis.settings.modeling import Config, load_config
from plaque_analysis.common.paths import CONFIG_ROOT, OUTPUT_ROOT, Data
from plaque_analysis.common.output import validate_output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Grouped prediction and imaging-reference classification")
    parser.add_argument("--task", choices=("prediction", "atlas", "stability", "all", "plot"), default="all")
    parser.add_argument("--data", type=Path, default=Path(Data) / "plaques.csv",
                        help="Your plaque-level CSV; replace the template 'your project/dataset/plaques.csv'")
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "modeling.json")
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_ROOT / "modeling",
                        help="Result directory; supplied relative paths use the current directory")
    parser.add_argument("--write-config", type=Path, help="Write default configuration and exit")
    parser.add_argument("--plots", action="store_true", help="Generate panels from the computed tables")
    parser.add_argument("--no-pathology", action="store_true", help="Fit reference classes without the validation residual calculation")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing result files in the selected output directory")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.write_config:
            destination = validate_output_path(args.write_config, (args.data.parent, args.config))
            if destination.exists():
                raise ValueError("Configuration destination already exists")
            destination.write_text(json.dumps(asdict(Config()), indent=2) + "\n", encoding="utf-8")
            return 0
        if args.task != "plot" and "your project" in args.data.parts:
            raise ValueError("Configure --data with your plaque CSV; replace the 'your project/dataset' placeholder")
        config = load_config(args.config)
        output = validate_output_path(args.output, (args.data.parent, args.config))
        if args.task == "plot":
            from plaque_analysis.plotting.modeling import plot_results
            plot_results(output, config)
            return 0
        from plaque_analysis.io.modeling import read_table, write_run_manifest, construction_mask
        require_target = args.task in ("prediction", "all") or (args.task == "atlas" and not args.no_pathology)
        frame = read_table(args.data, config, require_target=require_target)
        if args.task in ("atlas", "all"):
            construction_mask(frame, config)
            if not args.no_pathology and len(config.construction_animals) < config.inner_folds:
                raise ValueError("The atlas baseline needs at least inner_folds construction animals")
        if args.task in ("prediction", "all") and frame[config.animal_column].nunique() - 1 < config.inner_folds:
            raise ValueError("Insufficient outer-training animals for inner_folds")
        if args.task in ("stability", "all"):
            from plaque_analysis.models.splits import stratified_animal_sets
            stratified_animal_sets(frame, config)
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise ValueError("Output directory is not empty; choose another directory or use --overwrite")
        output.mkdir(parents=True, exist_ok=True)
        write_run_manifest(output, args.data, config, args.task)
        if args.task in ("prediction", "all"):
            from plaque_analysis.models.prediction import run_prediction
            run_prediction(frame, config, output / "prediction")
        if args.task in ("atlas", "all"):
            from plaque_analysis.models.atlas import run_atlas
            run_atlas(frame, config, output / "atlas", pathology=not args.no_pathology)
        if args.task in ("stability", "all"):
            from plaque_analysis.models.stability import run_stability
            run_stability(frame, config, output / "stability")
        if args.plots:
            from plaque_analysis.plotting.modeling import plot_results
            plot_results(output, config)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Completed: {args.output}")
    return 0
