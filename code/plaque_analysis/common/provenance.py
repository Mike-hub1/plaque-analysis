"""Portable source provenance for the integrated analysis package."""
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
from plaque_analysis.common.paths import PACKAGE_ROOT
from plaque_analysis.common.files import sha256_file


def source_hashes():
    """Hash Python and native modules relative to the trial package root."""
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path)
            for path in sorted(PACKAGE_ROOT.rglob("*"))
            if path.is_file() and (path.suffix == ".py" or
                any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES))}


package_source_hashes = source_hashes
code_hashes = source_hashes


def code_sha256():
    digest = hashlib.sha256()
    for name, value in code_hashes().items():
        digest.update((name + "\0" + value + "\n").encode("utf-8"))
    return digest.hexdigest()
