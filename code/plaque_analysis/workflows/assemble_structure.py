#!/usr/bin/env python3
"""Compose a reference A4 layout; the manuscript SVG is in figures/Figure2.svg."""
import argparse
import json
from pathlib import Path
from plaque_analysis.plotting.layout import compose
from plaque_analysis.common.paths import OUTPUT_ROOT
from plaque_analysis.common.output import validate_output_path

OUTPUT_DIR = OUTPUT_ROOT / "structure"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--panel-dir", type=Path, default=OUTPUT_DIR / "panels",
                        help="Rendered Figure 2 SVG panels")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "Figure2_reconstructed_A4.svg",
                        help="Reconstructed A4 SVG")
    args = parser.parse_args(argv)
    args.panel_dir = args.panel_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    try:
        validate_output_path(args.output.parent)
        panel_files = {(args.panel_dir / f"fig2{letter}.svg").resolve()
                       for letter in "abcdefghi"}
        if args.output in panel_files:
            raise ValueError("The assembled SVG must not overwrite an input panel")
    except ValueError as exc:
        parser.error(str(exc))
    missing = [letter for letter in "abcdefghi"
               if not (args.panel_dir / f"fig2{letter}.svg").is_file()]
    if missing:
        parser.error("Missing SVG panels: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    height, placements, panels = compose(str(args.panel_dir), str(args.output))
    audit = {"page_width_pt": 595.28, "page_height_pt": height,
             "content_height_pt": panels["_content_h"],
             "vertical_fill_factor": panels["_fv"], "panels": {}}
    for letter, placement in placements.items():
        panel = panels[letter]
        audit["panels"][letter] = dict(placement, font_scale=panel["k"],
                                      effective_font_sizes_pt=panel["eff"])
    args.output.with_suffix(".layout.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} (210 × 297 mm)")


if __name__ == "__main__":
    main()
