"""Protect source, artwork and input paths when selecting an output directory."""

from pathlib import Path

from plaque_analysis.common.paths import ARTWORK_ROOT, CODE_ROOT


def validate_output_path(output, protected=()):
    """Resolve an output path and reject overlap with protected inputs or sources.

    Existing-output and overwrite policies remain with the calling workflow.
    Directory paths protect their descendants as well as their ancestors.
    """
    output = Path(output).expanduser().resolve()
    for root in (CODE_ROOT, ARTWORK_ROOT):
        root = root.resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"Output directory overlaps protected source or artwork: {root}")
    for value in protected:
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if path.is_relative_to(output) or (path.is_dir() and output.is_relative_to(path)):
            raise ValueError(f"Output directory overlaps an input path: {path}")
    return output
