"""Orchestrate segmentation, area measurement and exports without global paths."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from plaque_analysis.imaging.area_boundaries import auto_canny, draw_overlay, get_red_white_boundary, save_svg, subpixel_polylines
from plaque_analysis.settings.area import AreaConfig
from plaque_analysis.io.area import export_measurements
from plaque_analysis.plotting.area import create_three_split_images, select_and_annotate_top_regions, visualize_region_relationship, visualize_segmentation_comparison
from plaque_analysis.analysis.area_reference import adjust_segmentation_with_reference, evaluate_segmentation_quality, load_reference_masks
from plaque_analysis.imaging.area_regions import calculate_red_areas_for_all_regions, merge_small_regions_to_nearest_large
from plaque_analysis.imaging.area_segmentation import hybrid_red_white_segmentation, verify_red_outside_white_inside
from plaque_analysis.common.output import validate_output_path


def read_colour_image(path):
    """Read an 8-bit colour image; OpenCV decodes alpha images as BGRA."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError("This colour-threshold workflow requires an 8-bit image; use fluorescence for native channels")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected grayscale, BGR or BGRA image")
    return image


def segment_image(image, red_reference=None, white_reference=None, *, config=AreaConfig()):
    """Keep the original reference-guided segmentation and label correction."""
    red, white = hybrid_red_white_segmentation(image, red_reference, white_reference, config=config)
    red, white = adjust_segmentation_with_reference(image, red, white, red_reference, white_reference)
    if red_reference is not None and white_reference is not None:
        red_red = np.sum((red > 0) & (red_reference > 0)) / (np.sum(red_reference > 0) + 1e-6)
        red_white = np.sum((red > 0) & (white_reference > 0)) / (np.sum(white_reference > 0) + 1e-6)
        white_white = np.sum((white > 0) & (white_reference > 0)) / (np.sum(white_reference > 0) + 1e-6)
        white_red = np.sum((white > 0) & (red_reference > 0)) / (np.sum(red_reference > 0) + 1e-6)
        if red_white > red_red and white_red > white_white:
            red, white = white, red
    else:
        red, white = verify_red_outside_white_inside(red, white)
    # Report the final labels, including any positional correction.
    metrics = evaluate_segmentation_quality(red, white, red_reference, white_reference)
    return red, white, metrics


def export_boundaries(image, red, white, output):
    """Export the original Canny/gradient/subpixel boundary products."""
    band = get_red_white_boundary(red, white)
    if not np.any(band):
        band = cv2.bitwise_or(cv2.Canny(red, 50, 150), cv2.Canny(white, 50, 150))
        _write_image(output / "08_boundary_band_fallback.png", band)
    else:
        _write_image(output / "09_boundary_band.png", band)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = max(1.0, min(3.0, image.shape[1] / 256.0))
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.bitwise_and(auto_canny(blur, sigma=0.33), band)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    lines = subpixel_polylines(edges, magnitude, gx, gy)
    _write_image(output / "10_edges_binary.png", edges)
    _write_image(output / "11_grad_mag.png", np.uint8(np.clip(magnitude / (magnitude.max() + 1e-6) * 255, 0, 255)))
    _write_image(output / "12_overlay_subpixel.png", draw_overlay(image, lines, color=(0, 255, 0)))
    save_svg(lines, str(output / "13_subpixel_edges.svg"), stroke_color="lime", stroke_width=1.0,
             image_shape=image.shape)
    return {"boundary_pixels": int(np.sum(band > 0)), "canny_pixels": int(np.sum(edges > 0)),
            "subpixel_polylines": len(lines), "subpixel_points": sum(len(line) for line in lines)}


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def export_region_figures(image, red, white, red_reference, white_reference, labels, results, output, config):
    _write_image(output / "01_mask_red.png", red)
    _write_image(output / "02_mask_white.png", white)
    _write_image(output / "03_region_relationship.png", visualize_region_relationship(image, red, white))
    _write_image(output / "04_segmentation_comparison.png",
                 visualize_segmentation_comparison(image, red, white, red_reference, white_reference))
    _write_image(output / "05_merged_white_mask.png", (labels > 0).astype(np.uint8) * 255)
    annotated, _ = select_and_annotate_top_regions(image, labels, top_n=config.top_n, config=config)
    _write_image(output / "06_annotated_top_white_regions.png", annotated)
    white_view, rings, boundaries, rows = create_three_split_images(image, labels, results, top_n=config.top_n)
    for name, view in (("07a_white_regions_only.png", white_view), ("07b_search_rings.png", rings),
                       ("07c_red_areas_boundaries.png", boundaries)):
        _write_image(output / name, view)
    _write_image(output / "14_segmentation_binary.png", (red > 0).astype(np.uint8) * 255)
    return rows


def run(image_path, output_dir, *, red_reference=None, white_reference=None, config=AreaConfig()):
    """Measure one image and write auditable products. Returns its JSON report.

    Calls seed OpenCV's process RNG. NumPy's prior random state is restored.
    Run separate processes for concurrent measurements with different seeds.
    """
    output = validate_output_path(output_dir, (image_path, red_reference, white_reference))
    for path in (red_reference, white_reference):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Reference image not found: {path}")
    image = read_colour_image(image_path)
    for path in (red_reference, white_reference):
        if path is not None:
            # IMREAD_COLOR silently downcasts a 16-bit reference to 8-bit.
            # Validate the decoded source before the legacy mask loader uses
            # colour thresholds; valid 8-bit images follow its unchanged path.
            try:
                read_colour_image(path)
            except ValueError as exc:
                raise ValueError(f"Invalid reference image {path}: {exc}") from exc
    output.mkdir(parents=True, exist_ok=True)
    red_ref, white_ref = load_reference_masks(
        str(red_reference) if red_reference else None,
        str(white_reference) if white_reference else None, image.shape[:2])
    state = np.random.get_state()
    try:
        np.random.seed(config.seed)
        cv2.setRNGSeed(config.seed)
        red, white, metrics = segment_image(image, red_ref, white_ref, config=config)
        labels, merge_info = merge_small_regions_to_nearest_large(
            white, config.small_region_threshold, config.merge_distance_threshold)
    finally:
        np.random.set_state(state)
    custom = None
    if config.interactive and np.any(labels):
        from plaque_analysis.imaging.area_interactive import interactive_search_ring_verification
        sizes = [(int(i), int(np.sum(labels == i))) for i in np.unique(labels) if i]
        sizes.sort(key=lambda row: row[1], reverse=True)
        custom = interactive_search_ring_verification(image, labels, red, sizes, config.top_n, config=config)
    results, _ = calculate_red_areas_for_all_regions(labels, red, custom, config=config)
    rows = export_region_figures(image, red, white, red_ref, white_ref, labels, results, output, config)
    boundary_summary = export_boundaries(image, red, white, output)
    export_measurements(rows, output / "15_area_measurements.csv", config)
    report = {
        "input": Path(image_path).name,
        "input_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "parameters": asdict(config), "shape": list(image.shape[:2]),
        "red_pixels": int(np.sum(red > 0)), "white_pixels": int(np.sum(white > 0)),
        "quality": metrics, "merge": merge_info, "selected_regions": len(rows),
        "red_area_all_regions_pixels": int(sum(r["red_area"] for r in results)),
        "red_area_selected_regions_pixels": int(sum(r["red_area"] for r in rows)),
        "boundary": boundary_summary, "custom_search_parameters": custom or {},
    }
    (output / "16_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
