"""Final SVG artwork and the preserved numerical PA field calibration."""
import base64
from io import BytesIO
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from plaque_analysis.io.polarization import resolve, sha256


def decode_svg_image(root, image_id):
    """Decode one embedded image in its SVG display orientation, without resampling."""
    elements = [el for el in root.iter() if el.get("id") == image_id]
    if len(elements) != 1 or elements[0].tag.rsplit("}", 1)[-1] != "image":
        raise ValueError(f"Expected one embedded SVG image with ID {image_id}")
    element = elements[0]
    href = element.get("{http://www.w3.org/1999/xlink}href", element.get("href", ""))
    if not href.startswith("data:image/") or ";base64," not in href:
        raise ValueError("The final SVG must contain its raster pixels as embedded base64 data")
    encoded = base64.b64decode(href.split(",", 1)[1])
    with Image.open(BytesIO(encoded)) as image:
        pixels = np.array(image.convert("RGBA"))
    match = re.fullmatch(r"translate\([-.\d]+[ ,]+[-.\d]+\)\s*scale\(([-.\d]+)[ ,]+([-.\d]+)\)",
                         element.get("transform", ""))
    if match is None:
        raise ValueError(f"Unsupported image transform for {image_id}")
    sx, sy = map(float, match.groups())
    if sx == 0 or sy == 0:
        raise ValueError("SVG image scales must be nonzero")
    if sx < 0:
        pixels = pixels[:, ::-1]
    if sy < 0:
        pixels = pixels[::-1]
    shape = (float(element.get("height")) * abs(sy), float(element.get("width")) * abs(sx))
    return pixels.copy(), shape


def load_final_artwork(config, out):
    """Read c and its colour ramp from the complete final SVG; copy that SVG unchanged."""
    final_svg = resolve(config, "final_svg")
    if not final_svg.is_file():
        raise FileNotFoundError(f"Final SVG artwork is missing: {final_svg}")
    destination = Path(out) / "Figure4_final.svg"
    if destination.exists() and sha256(destination) != sha256(final_svg):
        raise FileExistsError("The output contains a different Figure4_final.svg; choose a new output directory.")
    root = ET.parse(final_svg).getroot()
    panel, display_shape = decode_svg_image(root, config["panel_c_svg_id"])
    colourbar, _ = decode_svg_image(root, config["panel_c_colourbar_svg_id"])
    # The SVG shows high values at the top; ListedColormap needs low to high.
    centre = colourbar[::-1, colourbar.shape[1] // 2]
    opaque = centre[:, 3] == 255
    if opaque.sum() < 2:
        raise ValueError("The embedded SVG colour ramp has insufficient opaque pixels")
    ramp = centre[opaque, :3].astype(float) / 255
    if not destination.exists():
        shutil.copy2(final_svg, destination)
    return {"panels": {"c": panel}, "colour_ramps": {"c": ramp},
            "display_shape_px": {"c": display_shape}, "source_final_svg_sha256": sha256(final_svg)}


def apply_pa_field_calibration(config, artwork=None):
    """Preserve the confirmed PA calibration independently of SVG raster size.

    The 608 x 634 reference field and grid bounds are the existing analysis
    calibration, not inferred from the embedded SVG image dimensions.
    """
    if config.get("pa_calibration_mode") != "author_matched_optical_field":
        raise ValueError("Figure 4 requires its confirmed PA field calibration.")
    if not config.get("pa_field_override_confirmed"):
        raise ValueError("PA field calibration is not confirmed in the configuration.")
    h, w = config.get("pa_reference_field_shape_px", [608, 634])
    x0, _, x1, _ = config["roi_xyxy"]
    width_um = (x1-x0)*config["pixel_size_um"]
    source_pixel_um = width_um/w
    display_dx = (565.5/27)*source_pixel_um
    display_dy = (453/13)*source_pixel_um
    if config["pa_display_rot90"] % 2:
        config["pa_native_row_pitch_um"], config["pa_native_column_pitch_um"] = display_dx, display_dy
    else:
        config["pa_native_row_pitch_um"], config["pa_native_column_pitch_um"] = display_dy, display_dx
    config["approved_main_field_width_um"] = width_um
    display_h, display_w = (artwork.get("display_shape_px", {}).get("c", artwork["panels"]["c"].shape[:2])
                            if artwork else (h, w))
    config["approved_main_field_height_um"] = width_um*display_h/display_w
    config["PA_calibration_record"] = {
        "mode": "Author-confirmed matching of c field width to the optical field",
        "previous_reported_scan_row_column_um": [2.0, 2.0],
        "previous_pitch_used_for_current_scale_bars": False,
        "calibration_reference_height_width_px": [h, w],
        "reference_pixel_um": source_pixel_um,
        "effective_display_column_row_um": [display_dx, display_dy],
        "effective_native_row_column_um": [config["pa_native_row_pitch_um"], config["pa_native_column_pitch_um"]],
        "independent_scanner_pitch_remeasurement": False,
        "source_PA_values_changed": False,
        "cross_modality_pixel_registration_inferred": False,
        "interpretation": "Original confirmed effective calibration retained; the SVG display dimensions affect display aspect only.",
    }
