"""Command-line entry point; optional scientific imports are delayed until run."""
import argparse
from pathlib import Path
from plaque_analysis.common.paths import CONFIG_ROOT, Data
from plaque_analysis.io.polarization import load_config, resolve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct optical structure from 11 polarization TIFFs and a native PA DOA matrix.")
    parser.add_argument("--config", type=Path,
                        default=CONFIG_ROOT / "polarization.json",
                        help="JSON configuration; relative data paths are resolved beside this file")
    parser.add_argument("--data-dir", type=Path, default=Path(Data),
                        help="Dataset root used for placeholder input paths; defaults to Data in common/paths.py")
    parser.add_argument("--output", type=Path, help="Override the output directory (relative to the current directory)")
    parser.add_argument("--no-figures", action="store_true", help="Export numerical results only")
    parser.add_argument("--features-only", action="store_true", help="Export the complete feature table and atlases only; no artwork required")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, data_dir=args.data_dir)
        keys = ["raw_image_dir", "pa_csv"] + (["mask_npy"] if config.get("mask_npy") else [])
        if any(not config.get(key) or "your project" in Path(config[key]).parts for key in keys):
            raise ValueError(
                "Set Data in common/paths.py, pass --data-dir, or replace the placeholder input paths "
                "in --config. Other relative input paths are resolved beside the configuration file.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    import matplotlib
    matplotlib.use("Agg")
    from plaque_analysis.workflows.polarization import run_pipeline
    try:
        out = args.output.resolve() if args.output else resolve(config, "output_dir")
        run_pipeline(config, out, no_figures=args.no_figures, features_only=args.features_only)
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
