"""Read ROI measurements and summarize human optical, PA and LAMP1 images."""

from pathlib import Path
import numpy as np
import pandas as pd
from plaque_analysis.imaging.human import optical_parameters, mask_touches_field, lamp1_measurements


def _boolean(value, name):
    text = str(value).strip().lower()
    if text not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be True/False or 1/0")
    return text in ("true", "1")


def _mask(array, shape, name):
    value = np.asarray(array)
    if value.shape != shape or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must be a Boolean/0–1 mask with shape {shape}")
    return value.astype(bool)


def _distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, [.25, .5, .75]) if len(values) else [np.nan]*3
    return {"q25": float(quantiles[0]), "median": float(quantiles[1]),
            "q75": float(quantiles[2]), "n": len(values)}


def _load_roi(root, row, settings):
    relative = Path(str(row.file))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("ROI file paths must be relative to the supplied dataset directory")
    with np.load(path, allow_pickle=False) as archive:
        required = {"rgb", "optical_stack", "angles_deg", "plaque_mask", "lamp1", "valid_mask", "doa", "doa_mask"}
        if required - set(archive.files):
            raise ValueError(f"ROI {row.roi_id} is missing arrays: {sorted(required-set(archive.files))}")
        data = {key: np.asarray(archive[key]) for key in required}
        if "doa_pixel_size_um" in archive.files:
            data["doa_pixel_size_um"] = np.asarray(archive["doa_pixel_size_um"], float)
    spacing = np.asarray([row.pixel_size_x_um, row.pixel_size_y_um], dtype=float)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Each ROI requires positive measured x/y pixel sizes")
    shape = data["plaque_mask"].shape
    if len(shape) != 2 or not min(shape):
        raise ValueError("plaque_mask must be a nonempty 2D image")
    mask = _mask(data["plaque_mask"], shape, "plaque_mask")
    valid = _mask(data["valid_mask"], shape, "valid_mask")
    if not mask.any() or data["lamp1"].shape != shape or data["optical_stack"].shape[1:] != shape:
        raise ValueError("Plaque, LAMP1 and optical images must share the calibrated ROI pixel grid")
    if data["rgb"].shape != (*shape, 3) or not np.isfinite(data["rgb"]).all():
        raise ValueError("rgb must contain a finite RGB image on the ROI grid")
    doa = np.asarray(data["doa"], float)
    if doa.ndim != 2 or not min(doa.shape):
        raise ValueError("doa must be an independent 2D PA measurement grid")
    doa_mask = _mask(data["doa_mask"], doa.shape, "doa_mask")
    optical = optical_parameters(data["optical_stack"], data["angles_deg"],
        phi_sign=settings.get("phi_sign", 1), phi_offset_deg=settings.get("phi_offset_deg", 0),
        max_se_deg=settings.get("max_angle_se_deg", 10))
    optical["phi_valid"] &= mask & valid
    optical["m2"] = np.where(mask & valid, optical["m2"], np.nan)
    optical["phi"] = np.where(optical["phi_valid"], optical["phi"], np.nan)
    data.update(optical, plaque_mask=mask, valid_mask=valid, doa=np.where(doa_mask, doa, np.nan),
                spacing_xy=spacing, roi_id=row.roi_id, section_id=row.section_id)
    # PA coordinates are independent. A physical PA scale bar requires its own
    # calibration; no optical-to-PA pixel correspondence is inferred here.
    pa_spacing = [row.get("doa_pixel_size_x_um", np.nan), row.get("doa_pixel_size_y_um", np.nan)]
    pa_spacing = np.asarray(pa_spacing, float)
    if not np.isfinite(pa_spacing).any() and "doa_pixel_size_um" in data:
        pa_spacing = data["doa_pixel_size_um"]
    if pa_spacing.shape != (2,) or not np.isfinite(pa_spacing).all() or (pa_spacing <= 0).any():
        raise ValueError("Supply independent DOA [x, y] pixel sizes in doa_pixel_size_um or both ROI table DOA pixel-size columns")
    data["doa_spacing_xy"] = pa_spacing
    return data


def human_rows(frame, gap=1.5):
    """Keep the supplied ROI order within each dynamically discovered section."""
    pieces, offset = [], 0.
    for _, rows in frame.groupby("section_id", sort=False):
        rows = rows.copy()
        rows["row_y"] = np.arange(len(rows), dtype=float) + offset
        pieces.append(rows)
        offset += len(rows) + gap
    return pd.concat(pieces, ignore_index=True)


def prepare_human(data_dir, settings=None):
    """Read Data/human/rois.csv and return three tables plus representative maps.

    Required columns: roi_id, section_id, file, pixel_size_x_um,
    pixel_size_y_um, representative. Each file is relative to Data. Input LAMP1
    intensities must already be background corrected; RGB is for display only.
    Independent PA pixel sizes are supplied as NPZ doa_pixel_size_um=[x,y]
    or ROI-table doa_pixel_size_x_um / doa_pixel_size_y_um columns.
    """
    root, settings = Path(data_dir).resolve(), settings or {}
    rois = pd.read_csv(root / "human" / "rois.csv", dtype={"roi_id": str, "section_id": str, "file": str})
    required = {"roi_id", "section_id", "file", "pixel_size_x_um", "pixel_size_y_um", "representative"}
    if required - set(rois) or rois.empty:
        raise ValueError(f"Provide nonempty human/rois.csv with columns {sorted(required)}")
    if rois[list(required)].isna().any().any() or rois.roi_id.duplicated().any():
        raise ValueError("ROI metadata must be complete and roi_id must be unique")
    if any(rois[column].str.strip().eq("").any() for column in ("roi_id", "section_id", "file")):
        raise ValueError("ROI, section and file identifiers must not be empty")
    rois["representative"] = rois.representative.map(lambda value: _boolean(value, "representative"))
    if rois.representative.sum() != 1:
        raise ValueError("Mark exactly one ROI as representative for panel f")
    structures, intensities, representative = [], [], None
    for _, row in rois.iterrows():
        data = _load_roi(root, row, settings)
        phi = data["phi"][data["phi_valid"]]
        order = float(np.clip(abs(np.exp(2j*np.deg2rad(phi)).mean()), 0, 1)) if len(phi) else np.nan
        record = {"roi_id": row.roi_id, "section_id": row.section_id, "S_global": order,
                  "direction_pixels": len(phi), "segmentation_boundary_limited": mask_touches_field(data["plaque_mask"])}
        for metric in ("m2", "doa"):
            record.update({f"{metric}_{name}": value for name, value in _distribution(data[metric]).items()})
        structures.append(record)
        for values in lamp1_measurements(data["lamp1"], data["plaque_mask"], data["valid_mask"], data["spacing_xy"]):
            intensities.append({"roi_id": row.roi_id, "section_id": row.section_id, **values})
        if row.representative:
            representative = {key: data[key] for key in ("rgb", "m2", "phi", "doa", "spacing_xy", "doa_spacing_xy", "roi_id", "section_id")}
    human = human_rows(pd.DataFrame(structures))
    lamp1 = pd.DataFrame(intensities).merge(human[["roi_id", "row_y"]], on="roi_id", validate="many_to_one", sort=False)
    pairs = lamp1.groupby("roi_id", sort=False).available.all()
    lamp1["paired"] = lamp1.roi_id.map(pairs)
    support = human[["roi_id", "section_id"]].assign(paired=lambda d: d.roi_id.map(pairs))
    support = support.groupby("section_id", sort=False).agg(n_rois=("roi_id", "count"), n_pairs=("paired", "sum")).reset_index()
    return {"panel_g_human_structure": human, "panel_h_lamp1": lamp1,
            "panel_h_pair_support": support}, representative
