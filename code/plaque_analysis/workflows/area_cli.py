"""Command line interface; --help needs only the Python standard library."""

import argparse
from pathlib import Path
from plaque_analysis.settings.area import AreaConfig
from plaque_analysis.common.paths import Data


def build_parser():
    parser = argparse.ArgumentParser(description="Red/white colour-image LAMP1 area measurements")
    parser.add_argument("--image", default=str(Path(Data) / "image.png"),
                        help="8-bit colour image; set Data in common/paths.py or pass --image PATH")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--red-reference")
    parser.add_argument("--white-reference")
    defaults = AreaConfig()
    for name in ("small_region_threshold", "top_n", "min_white_region_size", "dilation_size", "inner_erosion", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("merge_distance_threshold", "pixel_size_x", "pixel_size_y", "search_range_multiplier"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--interactive", action="store_true", help="Open OpenCV trackbars (requires a desktop session)")
    return parser


def main(argv=None):
    parser = build_parser()
    options = vars(parser.parse_args(argv))
    image, output = options.pop("image"), options.pop("output")
    red, white = options.pop("red_reference"), options.pop("white_reference")
    if "your project" in Path(image).parts:
        parser.error("Set Data = 'your project/dataset' in common/paths.py to your dataset, or pass --image with your image path.")
    try:
        config = AreaConfig(**options)
        from plaque_analysis.workflows.area import run
        report = run(image, output, red_reference=red, white_reference=white, config=config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Measured {report['selected_regions']} selected regions. Outputs: {output}")
    return 0
