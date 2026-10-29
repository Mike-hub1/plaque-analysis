"""Shared dataset placeholder and locations within the source distribution."""

from pathlib import Path


Data = "your project/dataset"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_INSTALLED_RESOURCES = PACKAGE_ROOT / "_resources"
_IS_INSTALLED = (_INSTALLED_RESOURCES / "code").is_dir()
CODE_ROOT = _INSTALLED_RESOURCES / "code" if _IS_INSTALLED else PACKAGE_ROOT.parent
REPOSITORY_ROOT = CODE_ROOT.parent
CONFIG_ROOT = CODE_ROOT / "config"
ARTWORK_ROOT = REPOSITORY_ROOT / "figures"
OUTPUT_ROOT = (Path.cwd() if _IS_INSTALLED else REPOSITORY_ROOT) / "outputs"
