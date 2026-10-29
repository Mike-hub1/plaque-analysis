"""Typed plaque tables, explicit feature roles and portable provenance."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from plaque_analysis.common.files import sha256_file, write_json
from plaque_analysis.common.provenance import source_hashes


def read_table(path, config, *, require_target=True):
    frame = pd.read_csv(path, dtype={config.id_column: str, config.animal_column: str})
    validate_table(frame, config, require_target=require_target)
    return frame


def validate_table(frame, config, *, require_target=True):
    config.validate()
    required = [config.id_column, config.animal_column, *config.baseline, *config.imaging_features()]
    if require_target:
        required.append(config.target_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty or frame[config.id_column].isna().any() or frame[config.id_column].duplicated().any():
        raise ValueError("plaque_id must be nonmissing and unique; input must not be empty")
    if frame[config.animal_column].isna().any():
        raise ValueError("animal_id must be nonmissing")
    numeric = [*config.baseline, *config.imaging_features()]
    if require_target:
        numeric.append(config.target_column)
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Declared numerical columns must be finite; resolve measurement QC before analysis")
    if require_target and (frame[config.target_column] < 0).any():
        raise ValueError("LAMP1 areas must be nonnegative µm² values before log1p")
    area_column = next(name for name in config.baseline if name != config.age_column)
    if (frame[area_column] < 0).any() or (frame[config.age_column] <= 0).any():
        raise ValueError("Aβ area must be nonnegative and age must be positive")
    if frame.groupby(config.animal_column)[config.age_column].nunique().max() != 1:
        raise ValueError("Each animal must have one declared age")
    if frame[config.animal_column].nunique() < 3:
        raise ValueError("At least three independent animal identifiers are required")


def targets(frame, config):
    return np.log1p(frame[config.target_column].to_numpy(dtype=float))


def construction_mask(frame, config):
    if not config.construction_animals:
        raise ValueError("List construction_animals explicitly in the configuration for atlas fitting")
    observed = set(frame[config.animal_column].astype(str))
    if not set(config.construction_animals) <= observed:
        raise ValueError("construction_animals contains identifiers absent from the input")
    mask = frame[config.animal_column].astype(str).isin(config.construction_animals).to_numpy()
    if mask.all() or not mask.any() or len(config.construction_animals) < 2:
        raise ValueError("Atlas construction and validation must each contain separate animals")
    return mask


def write_run_manifest(output, source, config, task):
    import importlib.metadata
    write_json(Path(output) / "run_manifest.json", {
        "task": task, "input_file": Path(source).name,
        "input_sha256": sha256_file(source),
        "configuration": asdict(config),
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "scikit-learn", "torch")},
        "source_sha256": source_hashes(),
        "target_scale": "log1p(LAMP1 area / µm²)",
    })
