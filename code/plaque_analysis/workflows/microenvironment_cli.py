"""Portable command line; help requires only the standard library."""

import argparse
from pathlib import Path
from plaque_analysis.common.paths import ARTWORK_ROOT, CONFIG_ROOT, OUTPUT_ROOT, Data
from plaque_analysis.common.output import validate_output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plaque microenvironment and human imaging analysis")
    parser.add_argument("--data-dir", type=Path, default=Path(Data))
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "microenvironment.json")
    parser.add_argument("--artwork-dir", type=Path, default=ARTWORK_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "microenvironment")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--export-final", action="store_true", help="Also copy the existing final SVG without modification")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panels", default="abcdefgh", help="Unique letters a–h; all eight also create a reference composite")
    args = parser.parse_args(argv)
    if args.data_dir == Path("your project/dataset"):
        parser.error("Set Data to your dataset directory or pass --data-dir PATH; 'your project/dataset' is a placeholder")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if args.dpi <= 0 or not args.panels or len(set(args.panels)) != len(args.panels) or set(args.panels) - set("abcdefgh"):
        parser.error("DPI must be positive; panels must be unique letters a–h")
    try:
        output = validate_output_path(args.output, (args.data_dir, args.artwork_dir, args.config))
        from plaque_analysis.workflows.microenvironment import run
        report = run(args.data_dir, output, args.artwork_dir, no_figures=args.no_figures,
                     dpi=args.dpi, panels=args.panels, config_file=args.config, export_final=args.export_final)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Prepared {len(report['table_rows'])} source tables. Outputs: {args.output}")
    return 0
