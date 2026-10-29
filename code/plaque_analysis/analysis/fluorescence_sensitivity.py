"""Figure 3: sensitivity."""
from __future__ import annotations

from plaque_analysis.analysis.fluorescence import analyse_arrays
from plaque_analysis.imaging.fluorescence_segmentation import segment_signal_foreground
from dataclasses import asdict, dataclass, replace
from scipy import ndimage, stats
from skimage.measure import label as sk_label
import numpy as np


def sensitivity_analysis(raw, signed, seg_images, valid, labels, spacing, config,
                         segmentation_kwargs, effective_bits, metadata_bits, manual_masks):
    """One-factor perturbations on the same image, not biological repeats or ICC."""
    scenarios = [("A_threshold", 0.9), ("A_threshold", 1.1),
                 ("L_threshold", 0.9), ("L_threshold", 1.1),
                 ("window_radius", 0.8), ("window_radius", 1.2),
                 ("ring_width", 1.5), ("L_registration_x", -1), ("L_registration_x", 1)]
    rows = []
    base_ids = [int(v) for v in np.unique(labels) if v > 0]
    for parameter, factor in scenarios:
        if parameter in ("A_threshold", "L_threshold") and (isinstance(manual_masks[0 if parameter == "A_threshold" else 1], np.ndarray) or manual_masks[0 if parameter == "A_threshold" else 1] is True):
            continue
        masks = [labels > 0, None]
        masks[1] = (manual_masks[1].copy() if isinstance(manual_masks[1], np.ndarray)
                    else segment_signal_foreground(seg_images[1], valid=valid[1],
                                                   **segmentation_kwargs[1])[0])
        perturbed = labels.copy()
        if parameter == "A_threshold":
            masks[0], _ = segment_signal_foreground(seg_images[0], valid=valid[0], threshold_factor=factor,
                                                    **segmentation_kwargs[0])
            perturbed = sk_label(masks[0], connectivity=2)
        if parameter == "L_threshold":
            masks[1], _ = segment_signal_foreground(seg_images[1], valid=valid[1], threshold_factor=factor,
                                                    **segmentation_kwargs[1])
        lc, lv = signed[1], valid[1]
        if parameter == "L_registration_x":
            shift = (0, factor)
            masks[1] = ndimage.shift(masks[1].astype(np.uint8), shift, order=0, mode="constant", cval=0) > 0
            lv = ndimage.shift(valid[1].astype(np.uint8), shift, order=0, mode="constant", cval=0) > 0
            lc = ndimage.shift(signed[1], shift, order=0, mode="constant", cval=0)
        cfg = replace(config, **({"window_radius_um": config.window_radius_um*factor} if parameter == "window_radius"
                                else {"ring_width_um": config.ring_width_um*factor} if parameter == "ring_width" else {}))
        perturbed_result = analyse_arrays(raw, [signed[0], lc], [perturbed, masks[1]],
            [valid[0], lv], spacing, cfg, effective_bits=effective_bits, metadata_bits=metadata_bits,
            identities={"PlaqueID": "sensitivity"})
        matched_objects = {obj["id"]: obj for obj in perturbed_result["objects"]}
        for base_id in base_ids:
            overlap = perturbed[labels == base_id]
            overlap = overlap[overlap > 0]
            if not len(overlap):
                rows.append({"base_object_id": base_id, "parameter": parameter, "factor": factor,
                             "status": "object_lost"})
                continue
            values, counts = np.unique(overlap, return_counts=True)
            target_id = int(values[np.argmax(counts)])
            mask = perturbed == target_id
            if target_id not in matched_objects:
                rows.append({"base_object_id": base_id, "parameter": parameter, "factor": factor,
                             "status": "matched_object_below_minimum_size"})
                continue
            measured = matched_objects[target_id]
            rows.append({"base_object_id": base_id, "parameter": parameter, "factor": factor,
                         "status": "exploratory_single_image_perturbation",
                         "object_match_iou": float(np.count_nonzero(mask & (labels == base_id)) /
                                                   np.count_nonzero(mask | (labels == base_id))),
                         "matched_fragment_count": int(len(values)),
                         "base_labels_in_matched_object": int(len(np.unique(labels[mask & (labels > 0)]))),
                         **{"A_"+k: v for k, v in measured["features"].items()},
                         **{"L_"+k: v for k, v in measured["outcomes"].items()},
                         **{"AL_"+k: v for k, v in measured["relations"].items()}})
    return rows

