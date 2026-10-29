"""Keep the existing final artwork separate from newly calculated panels."""

from pathlib import Path
import shutil

MASTER_NAME = "Figure7.svg"


def copy_final(artwork_dir, output_dir):
    source = Path(artwork_dir) / MASTER_NAME
    destination = Path(output_dir) / "Figure7_final.svg"
    if not source.is_file():
        raise FileNotFoundError(f"Final artwork is absent: {source}")
    shutil.copyfile(source, destination)
    return destination
