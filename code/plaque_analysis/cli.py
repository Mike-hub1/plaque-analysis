"""One command entry for the integrated analysis workflows."""

import argparse
import importlib
import os
import shutil
import subprocess
import sys

from plaque_analysis.common.paths import CODE_ROOT, Data, OUTPUT_ROOT


WORKFLOWS = {
    "structure": (None, "Structural statistics and model comparisons (Figure 2; R)"),
    "fluorescence": ("fluorescence_cli", "Confocal image features and LAMP1 measurements (Figure 3)"),
    "polarization": ("polarization_cli", "Optical reconstruction and photoacoustic features (Figure 4)"),
    "modeling": ("modeling_cli", "Grouped prediction, imaging atlas and stability (Figure 5)"),
    "transcriptomics": ("transcriptomics_cli", "Registration, fixed classification and RNA analysis (Figure 6)"),
    "microenvironment": ("microenvironment_cli", "Plaque neighborhoods and human imaging (Figure 7)"),
    "area": ("area_cli", "LAMP1 area measurements from color images"),
    "assemble-structure": ("assemble_structure", "Compose the structural analysis SVG panels"),
}


def _run_structure(arguments):
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Rscript is required for structure; install the R environment listed in requirement.txt")
    arguments = list(arguments)
    if not any(arg == "--data-dir" or arg.startswith("--data-dir=") for arg in arguments):
        arguments.extend(["--data-dir", os.environ.get("PLAQUE_DATA_DIR", Data)])
    if not any(arg == "--python" or arg.startswith("--python=") for arg in arguments):
        arguments.extend(["--python", sys.executable])
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in arguments):
        arguments.extend(["--output-dir", str(OUTPUT_ROOT / "structure")])
    result = subprocess.run([executable, str(CODE_ROOT / "R/run_structure.R"), *arguments])
    return result.returncode


def main(argv=None):
    workflow_help = "\n".join(f"  {name:20s} {description}"
                              for name, (_, description) in WORKFLOWS.items())
    parser = argparse.ArgumentParser(
        description="Integrated plaque imaging, modeling and spatial analysis.",
        epilog=f"Workflows:\n{workflow_help}\n\n"
               "Use %(prog)s WORKFLOW --help for input and output options.\n"
               "Set Data in plaque_analysis/common/paths.py or pass the workflow input path.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("workflow", nargs="?", choices=WORKFLOWS,
                        metavar="WORKFLOW", help="Analysis workflow to run")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.workflow is None:
        parser.print_help()
        return 0
    try:
        module_name = WORKFLOWS[args.workflow][0]
        if module_name is None:
            return _run_structure(args.arguments)
        module = importlib.import_module(f"plaque_analysis.workflows.{module_name}")
        return module.main(args.arguments)
    except (OSError, ImportError, RuntimeError) as exc:
        if isinstance(exc, ModuleNotFoundError) and (
                (exc.name or "").startswith("plaque_analysis._native") or
                exc.name in ("plaque_analysis.imaging.reconstruction", "plaque_analysis.imaging.orientation")):
            parser.error("This workflow requires the private plaque-analysis-core package. "
                         "Request the matching core package and signed license from the project owner.")
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
