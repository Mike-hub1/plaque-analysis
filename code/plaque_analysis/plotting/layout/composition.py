"""Figure 2 SVG composition; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.editing import keep_shapes_round
from plaque_analysis.plotting.layout.panels import prepare_cached, _CACHE
from plaque_analysis.plotting.layout.constants import BOTTOM_MARGIN, COL_GAP, LETTER_CAP, LETTER_INSET, PAGE_H, PAGE_W, PANEL_GAP_X, RIGHT, ROW_GAP, SLOTS, SVG_NS, TOP_R, XLINK_NS, make_slots


def compose(panel_dir, out_path, split=225.0, row_gap=ROW_GAP, write=True, fill=True):
    _CACHE.clear()  # Each render owns its mutable SVG trees.
    SLOTS = make_slots(split)
    SPLIT_R = split + COL_GAP
    ROW_GAP_ = row_gap
    P = {}
    place = {}

    def put(L, row_top, slot=None):
        slot = slot or SLOTS[L]
        p = P[L] = prepare_cached(os.path.join(panel_dir, f"fig2{L}.svg"), L, slot)
        s, delta = p["s"], p["delta"]
        x0, y0, x1, y1 = p["bbox"]
        top = row_top + delta
        place[L] = dict(s=s, tx=slot["x0"] - x0 * s, ty=top - y0 * s, top=top,
                        bottom=top + (y1 - y0) * s, right=slot["x0"] + (x1 - x0) * s,
                        row=row_top, lx=slot["lx"], x0=slot["x0"],
                        axis=top + (p["plot"][3] - y0) * s)

    def align_axes(row):
        """Shift panels of one row down so that their x-axes share one line."""
        target = max(place[L]["axis"] for L in row)
        for L in row:
            d = target - place[L]["axis"]
            for k in ("ty", "top", "bottom", "axis"):
                place[L][k] += d
        return max(place[L]["bottom"] for L in row)

    def row3(r3, h):
        """e in the left column; f | g share the right column at the same
        height h.  Returns g's right edge (bisected onto the page edge)."""
        put("e", r3, dict(SLOTS["e"], hmax=h))
        put("f", r3, dict(SLOTS["f"], hmax=h, x1=SPLIT_R + 1e4))
        x0 = place["f"]["right"] + PANEL_GAP_X
        put("g", r3, dict(lx=x0 + LETTER_INSET, x0=x0, x1=x0 + 1e4, hmax=h))
        return place["g"]["right"]

    put("a", TOP_R); put("b", TOP_R)
    r2 = align_axes("ab") + ROW_GAP_
    put("d", r2); put("c", r2)
    r3 = align_axes("dc") + ROW_GAP_
    lo, hi = 60.0, 320.0           # bisection: f | g fill the right column
    for _ in range(18):
        mid = (lo + hi) / 2
        if row3(r3, mid) > RIGHT:
            hi = mid
        else:
            lo = mid
    row3(r3, lo)
    r4 = align_axes("efg") + ROW_GAP_
    put("h", r4); put("i", r4)
    content_h = align_axes("hi")
    page_h = PAGE_H
    if not write:
        return content_h
    if content_h + BOTTOM_MARGIN > PAGE_H + 1e-6:
        raise SystemExit(f"content {content_h:.1f} pt exceeds the A4 height")
    # fill the A4 height: stretch the content vertically about the top margin;
    # text and circular markers are counter-scaled so they stay undistorted
    fv = (PAGE_H - BOTTOM_MARGIN - TOP_R) / (content_h - TOP_R) if fill else 1.0
    Y = lambda y: TOP_R + (y - TOP_R) * fv
    if fv != 1.0:
        for L in "abcdefghi":
            keep_shapes_round(P[L]["root"], fv)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" version="1.1" '
        f'width="210mm" height="297mm" viewBox="0 0 {PAGE_W:.2f} {page_h:.2f}">\n',
        '<style>.panel-letter { font-family: Arial, Helvetica, sans-serif; '
        'font-size: 17px; font-weight: 700; fill: #111111; }</style>\n',
        f'<rect id="page-background" x="0" y="0" width="{PAGE_W:.2f}" '
        f'height="{page_h:.2f}" fill="#FFFFFF"/>\n',
    ]
    for L in "abcdefghi":
        q = place[L]
        bx = P[L]["bbox"]
        m = 1.5 / q["s"]
        inner = "".join(ET.tostring(ch, encoding="unicode") for ch in P[L]["root"])
        inner = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", inner)
        inner = re.sub(r'\sdata-(?:lg|bbox)="[^"]*"', "", inner)   # layout-only helpers
        parts.append(
            f'<g id="fig2{L}">\n'
            f'  <g transform="translate({q["tx"]:.3f} {Y(q["ty"]):.3f}) scale({q["s"]:.5f} {q["s"] * fv:.5f})">\n'
            f'  <clipPath id="fig2{L}_extent" clipPathUnits="userSpaceOnUse">'
            f'<rect x="{bx[0] - m:.2f}" y="{bx[1] - m:.2f}" width="{bx[2] - bx[0] + 2 * m:.2f}" '
            f'height="{bx[3] - bx[1] + 2 * m:.2f}"/></clipPath>\n'
            f'  <g clip-path="url(#fig2{L}_extent)">\n'
            f'{inner}\n  </g>\n  </g>\n'
            f'  <text class="panel-letter" x="{q["lx"]:.2f}" y="{Y(q["row"]) + LETTER_CAP:.2f}">{L}</text>\n'
            f'</g>\n')
    parts.append("</svg>\n")
    svg = "".join(parts)
    ET.fromstring(svg.encode("utf-8"))  # well-formedness
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    P["_content_h"] = content_h
    P["_fv"] = fv
    return page_h, place, P
