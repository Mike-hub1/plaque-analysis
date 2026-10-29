"""Figure 2 SVG panels; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.geometry import analyse, nums
from plaque_analysis.plotting.layout.editing import prefix_ids, standardise_axis_titles, strip_titles, unclip_canvas
from plaque_analysis.plotting.layout.legend import apply_legend, legend_transform, relayout_e_legend
from plaque_analysis.plotting.layout.typography import apply_text, choose_k, content_bbox, scaled_layout
from plaque_analysis.plotting.layout.constants import LETTER_CAP, LETTER_CLEAR, LETTER_W, MIN_SCALE_KEEP, SLOTS


def fit(boxes, bbox, slot):
    x0, y0, x1, y1 = bbox
    cw, ch = x1 - x0, y1 - y0
    slot_w = slot["x1"] - slot["x0"]
    lx0, lx1 = slot["lx"] - 1.0, slot["lx"] + LETTER_W
    need = LETTER_CAP + LETTER_CLEAR
    s = min(slot_w / cw, slot["hmax"] / ch)
    delta = 0.0
    for _ in range(30):
        d = 0.0
        for b in boxes:
            bx0 = slot["x0"] + (b[0] - x0) * s
            bx1 = slot["x0"] + (b[2] - x0) * s
            if bx1 >= lx0 and bx0 <= lx1:
                d = max(d, need - (b[1] - y0) * s)
        delta = d
        s_new = min(slot_w / cw, (slot["hmax"] - delta) / ch)
        if abs(s_new - s) < 1e-7:
            break
        s = s_new
    return s, delta


def prepare_panel(path, L, slot=None):
    root = ET.parse(path).getroot()
    vb = nums(root.get("viewBox"))
    removed = strip_titles(L, root, vb)
    renamed = standardise_axis_titles(L, root)
    unclip_canvas(root, vb)
    info = analyse(root, vb, L)
    if L == "e" and relayout_e_legend(root, info):
        info = analyse(root, vb, L)
    info["lt"] = None
    slot = slot or SLOTS[L]
    k_ax = k_ot = 1.0
    new = scaled_layout(info, 1.0, 1.0)
    bbox, boxes = content_bbox(info, new)
    s, delta = fit(boxes, bbox, slot)
    s0, kcap = s, float("inf")
    for _ in range(12):
        s = s0
        for _ in range(8):
            info["lt"] = legend_transform(info, s)
            k_ax, k_ot = choose_k(info, s, kcap)
            new = scaled_layout(info, k_ax, k_ot)
            bbox, boxes = content_bbox(info, new)
            s_new, delta = fit(boxes, bbox, slot)
            if abs(s_new - s) < 1e-4:
                s = s_new
                break
            s = s_new
        if s >= MIN_SCALE_KEEP * s0 - 1e-6:
            break
        kcap = max(1.0, 0.92 * max(k_ax, k_ot))
    apply_text(info, new)
    apply_legend(root, info, info["lt"])
    prefix_ids(root, "fig2" + L)
    lk = info["lt"]["k"] if info["lt"] else 1.0
    eff = sorted({round(t["fs"] * (lk if t["group"] == "legend" else n["k"]) * s, 2)
                  for t, n in zip(info["texts"], new)})
    return dict(root=root, bbox=bbox, boxes=boxes, s=s, delta=delta, plot=info["plot"],
                k=(k_ax, k_ot, lk), removed=removed, renamed=renamed, eff=eff)


_CACHE = {}


def prepare_cached(path, L, slot):
    key = (path, L, tuple(round(slot[k], 3) for k in ("lx", "x0", "x1", "hmax")))
    if key not in _CACHE:
        _CACHE[key] = prepare_panel(path, L, slot)
    return _CACHE[key]
