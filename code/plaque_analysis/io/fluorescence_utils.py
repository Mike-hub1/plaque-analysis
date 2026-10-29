"""Fluorescence matrix export and shared result serialization."""
from __future__ import annotations

import numpy as np
import tifffile
from plaque_analysis.common.files import sha256_file, array_sha256, json_clean, write_json
from plaque_analysis.common.provenance import package_source_hashes


def write_matrix(path, array, spacing):
    tifffile.imwrite(path, np.asarray(array), metadata={"axes": "YX", "spacing_yx_um": list(spacing)})



