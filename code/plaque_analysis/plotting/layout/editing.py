"""Figure 2 SVG editing; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.geometry import arial_width, font_size, nums, style_of, text_geom
from plaque_analysis.plotting.layout.constants import H_LABEL, NS, RENAMES, SUBTITLE_FILLS


def is_title(letter, el):
    st = style_of(el)
    _, y, ang, _ = text_geom(el)
    if ang:
        return False
    bold = "bold" in st.get("font-weight", "")
    if bold and y < 26 and font_size(el) >= 7.5:
        return True
    if (letter != "i" and not bold and y < 30
            and st.get("fill", "").lower() in SUBTITLE_FILLS):
        return True
    return False


def strip_titles(letter, root, vb):
    removed = []
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == NS + "text" and is_title(letter, child):
                removed.append("".join(child.itertext()))
                parent.remove(child)
            elif child.tag == NS + "rect" and not _in_clip(parent):
                # full-canvas white backgrounds would cover neighbouring panels
                st = style_of(child)
                w, h = child.get("width", ""), child.get("height", "")
                full = w.endswith("%") or (
                    nums(w) and nums(h) and abs(nums(w)[0] - vb[2]) < 0.5
                    and abs(nums(h)[0] - vb[3]) < 0.5)
                if full and st.get("fill", "").lower() in ("#ffffff", "white"):
                    parent.remove(child)
    return removed


def _in_clip(el):
    return el.tag == NS + "clipPath"


def standardise_axis_titles(letter, root):
    parent_of = {c: p for p in root.iter() for c in p}
    rot = [t for t in root.iter(NS + "text") if text_geom(t)[2] == -90]
    # a two-line y title ("LAMP1-related pathology" / "area (µm²)") -> one line
    if letter == "a":
        first = [t for t in rot if "".join(t.itertext()) == "LAMP1-related pathology"]
        second = [t for t in rot if "".join(t.itertext()) == "area (\u00b5m\u00b2)"]
        if first and second:
            x1, y1, _, _ = text_geom(first[0]); x2, _, _, _ = text_geom(second[0])
            first[0].text = "LAMP1-related pathology area (\u00b5m\u00b2)"
            first[0].set("transform", f"translate({(x1 + x2) / 2:.2f},{y1:.2f}) rotate(-90)")
            parent_of[second[0]].remove(second[0])
    changed = []
    if letter == "h":
        changed += split_h_labels(root)
    for t in root.iter(NS + "text"):
        ang = int(round(text_geom(t)[2]))
        table = RENAMES.get((letter, ang))
        old = "".join(t.itertext())
        if table and old in table:
            for ch in list(t):
                t.remove(ch)
            t.text = table[old]
            t.set("textLength", f"{arial_width(t.text, font_size(t)):.2f}px")
            changed.append((old, t.text))
    return changed


def split_h_labels(root):
    """Template style for h row labels: "Exclude HPF (n=298)" becomes the
    right-aligned lines "-HPF" / "n=298" (a leading "-" = region excluded)."""
    parent_of = {c: p for p in root.iter() for c in p}
    changed = []
    for t in list(root.iter(NS + "text")):
        old = "".join(t.itertext())
        m = H_LABEL.match(old)
        if not m or t.get("text-anchor", style_of(t).get("text-anchor")) != "end":
            continue
        name = re.sub(r"^Exclude ", "-", m.group(1))
        lines = ["All regions /", name[len("All regions / "):]] \
            if name.startswith("All regions / ") else [name]
        lines.append(f"n={m.group(2)}")
        x, y, _, _ = text_geom(t)
        fs = font_size(t)
        par = parent_of[t]
        idx = list(par).index(t)
        par.remove(t)
        for i, line in enumerate(lines):
            el = ET.Element(t.tag, dict(t.attrib))
            el.set("data-lg", f"{x:.2f}_{y:.2f}")
            el.text = line
            el.set("y", f"{y + (i - (len(lines) - 1) / 2) * 1.15 * fs:.2f}")
            el.set("textLength", f"{arial_width(line, fs):.2f}px")
            el.set("lengthAdjust", "spacingAndGlyphs")
            par.insert(idx + i, el)
        changed.append((old, " / ".join(lines)))
    return changed


def unclip_canvas(root, vb):
    """svglite clips everything to the panel canvas; pushed-out axis text may
    now lie just outside it, so widen the canvas-sized clip rectangles."""
    for cp in root.iter(NS + "clipPath"):
        r = cp.find(NS + "rect")
        if r is None:
            continue
        x, y = float(r.get("x", 0)), float(r.get("y", 0))
        w, h = float(r.get("width", 0)), float(r.get("height", 0))
        if x <= 0.5 and y <= 0.5 and x + w >= vb[2] - 0.5 and y + h >= vb[3] - 0.5:
            r.set("x", f"{-vb[2]:.2f}"); r.set("y", f"{-vb[3]:.2f}")
            r.set("width", f"{3 * vb[2]:.2f}"); r.set("height", f"{3 * vb[3]:.2f}")


def prefix_ids(root, prefix):
    mapping = {}
    for el in root.iter():
        i = el.get("id")
        if i:
            mapping[i] = prefix + "__" + re.sub(r"[^A-Za-z0-9_.-]", "_", i)
            el.set("id", mapping[i])
    if not mapping:
        return
    pat = re.compile(r"#(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
                     + r")(?![A-Za-z0-9_.-])")
    for el in root.iter():
        for k, v in list(el.attrib.items()):
            if "#" in v:
                nv = pat.sub(lambda m: "#" + mapping[m.group(1)], v)
                if nv != v:
                    el.set(k, nv)


def keep_shapes_round(root, fv):
    """Counter-scale text and circles by 1/fv about their own anchor."""
    inv = 1.0 / fv
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag == "text":
            tr = el.get("transform", "")
            m = re.match(r"\s*translate\(\s*([-\d.eE]+)[ ,]+([-\d.eE]+)\s*\)(.*)", tr)
            if m:
                el.set("transform", f"translate({m.group(1)},{m.group(2)}) scale(1,{inv:.6f}){m.group(3)}")
            else:
                x, y = float(el.get("x", 0)), float(el.get("y", 0))
                el.set("transform", f"translate({x:.3f},{y:.3f}) scale(1,{inv:.6f}) "
                                    f"translate({-x:.3f},{-y:.3f})" + (" " + tr if tr else ""))
        elif tag in ("circle", "ellipse"):
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            tr = el.get("transform", "")
            el.set("transform", f"translate({cx:.3f},{cy:.3f}) scale(1,{inv:.6f}) "
                                f"translate({-cx:.3f},{-cy:.3f})" + (" " + tr if tr else ""))
