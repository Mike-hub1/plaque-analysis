"""Portable CLI; help requires only the standard library."""
from pathlib import Path
import argparse
import json
import subprocess
from plaque_analysis.common.paths import ARTWORK_ROOT, CODE_ROOT, CONFIG_ROOT, OUTPUT_ROOT, Data
from plaque_analysis.common.output import validate_output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze registered spatial transcriptomics and export its figures")
    parser.add_argument("--data-dir", type=Path, default=Path(Data))
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "transcriptomics.json",
                        help="Method parameters; measured calibration is read from the dataset")
    parser.add_argument("--artwork", type=Path, default=ARTWORK_ROOT / "Figure6.svg")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "transcriptomics")
    parser.add_argument("--export-final", action="store_true", help="Copy final SVG unchanged and export exact a-f SVG groups")
    parser.add_argument("--replot", action="store_true", help="Draw a-f from the computed manuscript-method results")
    parser.add_argument("--render-final", action="store_true", help="Also render final SVGs to PNG/PDF; needs Rscript and rsvg")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI for regenerations and optional final rendering")
    args = parser.parse_args(argv)
    if args.data_dir == Path("your project/dataset"):
        parser.error("Set Data to your dataset directory or pass --data-dir PATH; 'your project/dataset' is a placeholder")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.render_final and not args.export_final:
        parser.error("--render-final requires --export-final")
    try:
        output = validate_output_path(args.output, (args.data_dir, args.artwork, args.config))
        from plaque_analysis.workflows.transcriptomics import run
        report = run(args.data_dir, args.artwork, output, export=args.export_final,
                     config_file=args.config, replot=args.replot, dpi=args.dpi,
                     layout_file=CONFIG_ROOT / "transcriptomics_layout.json")
        if args.render_final:
            subprocess.run(["Rscript", str(CODE_ROOT / "R/render_artwork.R"),
                            str(output / "final"), str(args.dpi)], check=True)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "panels": list(report["panels"]), "output": str(args.output)}, ensure_ascii=False))
    return 0
