"""Figure 2 SVG geometry; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.constants import LEGEND_LABELS, NS, _W


def nums(s):
    return [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s or "")]


def style_of(el):
    out = {}
    for part in el.get("style", "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    for k in ("fill", "stroke", "font-size", "font-weight", "text-anchor"):
        if el.get(k) is not None and k not in out:
            out[k] = el.get(k)
    return out


def font_size(el):
    v = nums(style_of(el).get("font-size", "0"))
    return v[0] if v else 0.0


def text_geom(el):
    """anchor point, angle (deg), whether position lives in a transform."""
    tr = el.get("transform", "")
    m = re.search(r"translate\(\s*([-\d.eE]+)[ ,]+([-\d.eE]+)\s*\)", tr)
    r = re.search(r"rotate\(\s*([-\d.eE]+)", tr)
    ang = float(r.group(1)) if r else 0.0
    if m:
        return float(m.group(1)), float(m.group(2)), ang, True
    return float(el.get("x", 0)), float(el.get("y", 0)), ang, False


def box_of_text(x, y, ang, fs, width, anchor):
    u0 = {"middle": -width / 2, "end": -width}.get(anchor, 0.0)
    u1 = u0 + width
    v0, v1 = -0.76 * fs, 0.22 * fs
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    xs, ys = [], []
    for u in (u0, u1):
        for v in (v0, v1):
            xs.append(x + ca * u - sa * v)
            ys.append(y + sa * u + ca * v)
    return (min(xs), min(ys), max(xs), max(ys))


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def arial_width(s, fs):
    return sum(_W.get(ch, 556 if not ch.isdigit() else 556) for ch in s) * fs / 1000.0


def visible(el):
    st = style_of(el)
    fill, stroke = st.get("fill", "").lower(), st.get("stroke", "").lower()
    if el.tag == NS + "rect":
        if fill in ("#ffffff", "white", "none", "") and stroke in ("#ffffff", "white", "none", ""):
            return False
    return True


def graphic_bbox(el):
    tag = el.tag.replace(NS, "")
    g = lambda k: float(el.get(k))
    try:
        if tag == "circle":
            return (g("cx") - g("r"), g("cy") - g("r"), g("cx") + g("r"), g("cy") + g("r"))
        if tag == "ellipse":
            return (g("cx") - g("rx"), g("cy") - g("ry"), g("cx") + g("rx"), g("cy") + g("ry"))
        if tag == "line":
            return (min(g("x1"), g("x2")), min(g("y1"), g("y2")),
                    max(g("x1"), g("x2")), max(g("y1"), g("y2")))
        if tag in ("polyline", "polygon", "path"):
            v = nums(el.get("points") if tag != "path" else el.get("d"))
            xs, ys = v[0::2], v[1::2]
            return (min(xs), min(ys), max(xs), max(ys)) if xs else None
        if tag in ("rect", "image"):
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            m = re.search(r"translate\(\s*([-\d.eE]+)[ ,]+([-\d.eE]+)", el.get("transform", ""))
            if m:
                x += float(m.group(1)); y += float(m.group(2))
            return (x, y, x + g("width"), y + g("height"))
    except (TypeError, ValueError):
        return None
    return None


def analyse(root, vb, letter=None):
    clips = {}
    for cp in root.iter(NS + "clipPath"):
        r = cp.find(NS + "rect")
        if r is not None:
            x, y = float(r.get("x", 0)), float(r.get("y", 0))
            clips[cp.get("id")] = (x, y, x + float(r.get("width")), y + float(r.get("height")))
    graphics, texts, graphic_els, hidden = [], [], [], []

    def walk(el, clip):
        cp = el.get("clip-path")
        if cp:
            m = re.search(r"#([^)]+)\)", cp)
            if m and m.group(1) in clips:
                c = clips[m.group(1)]
                clip = c if clip is None else (max(c[0], clip[0]), max(c[1], clip[1]),
                                               min(c[2], clip[2]), min(c[3], clip[3]))
        tag = el.tag.replace(NS, "")
        if tag in ("defs", "clipPath", "style", "metadata"):
            return
        if el.get("data-bbox"):              # pre-measured group (rotated colorbar)
            graphics.append(tuple(nums(el.get("data-bbox"))))
            graphic_els.append(el)
            return
        if tag == "text":
            s = "".join(el.itertext())
            if s.strip():
                x, y, ang, in_tr = text_geom(el)
                fs = font_size(el)
                tl = nums(el.get("textLength"))
                width = tl[0] if tl else arial_width(s, fs)
                anchor = style_of(el).get("text-anchor", el.get("text-anchor", "start"))
                texts.append(dict(el=el, s=s, x=x, y=y, ang=ang, in_tr=in_tr, fs=fs,
                                  w=width, anchor=anchor, lg=el.get("data-lg"),
                                  box=box_of_text(x, y, ang, fs, width, anchor)))
            return
        if tag == "rect" and not visible(el):
            b = graphic_bbox(el)
            if b:
                hidden.append((b, el))
        if tag in ("circle", "ellipse", "line", "polyline", "polygon", "path", "rect", "image") \
                and visible(el):
            b = graphic_bbox(el)
            if b:
                if clip:
                    b = (max(b[0], clip[0]), max(b[1], clip[1]), min(b[2], clip[2]), min(b[3], clip[3]))
                if b[2] >= b[0] and b[3] >= b[1]:
                    graphics.append(b)
                    graphic_els.append(el)
        for ch in el:
            walk(ch, clip)

    walk(root, None)
    inner = [c for c in clips.values()
             if not (c[0] <= 0.5 and c[1] <= 0.5 and c[2] >= vb[2] - 0.5 and c[3] >= vb[3] - 0.5)
             and (c[2] - c[0]) > 20 and (c[3] - c[1]) > 20]
    if inner:
        plot = (min(c[0] for c in inner), min(c[1] for c in inner),
                max(c[2] for c in inner), max(c[3] for c in inner))
    else:
        # no panel clip: take the axes, i.e. the lowest long horizontal line
        # and the leftmost long vertical line
        segs = [(b, el) for b, el in zip(graphics, graphic_els)
                if el.tag in (NS + "line", NS + "polyline")]
        hs = [b for b, el in segs if b[3] - b[1] < 0.2 and b[2] - b[0] > 0.4 * vb[2]]
        vs = [b for b, el in segs if b[2] - b[0] < 0.2 and b[3] - b[1] > 0.4 * vb[3]]
        if hs and vs:
            xa = max(hs, key=lambda b: b[3])
            ya = min(vs, key=lambda b: b[0])
            plot = (ya[0], ya[1], xa[2], xa[3])
        else:
            plot = (min(b[0] for b in graphics), min(b[1] for b in graphics),
                    max(b[2] for b in graphics), max(b[3] for b in graphics))
    if plot[3] - plot[1] > 0.9 * vb[3]:
        # e.g. the rank-concordance heatmap: clip covers the full height, so
        # take the extent of its square tiles as the plotting region
        tiles = [b for b, el in zip(graphics, graphic_els)
                 if el.tag == NS + "rect" and 6 < b[2] - b[0] < 30
                 and abs((b[2] - b[0]) - (b[3] - b[1])) < 0.5]
        if len(tiles) >= 25:
            plot = (min(b[0] for b in tiles), min(b[1] for b in tiles),
                    max(b[2] for b in tiles), max(b[3] for b in tiles))
    for t in texts:
        b = t["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if letter == "e" and b[0] > plot[2] + 2:
            side = "right"
        elif b[3] <= plot[1] - 2:
            side = "above"
        elif b[2] <= plot[0] + 1:
            side = "left"
        elif b[1] >= plot[3] - 1:
            side = "below"
        elif b[3] <= plot[1] + 1:
            side = "above"
        elif b[0] >= plot[2] - 1:
            side = "right"
        else:
            side = "inside"
        t["side"] = side
        t["group"] = {"left": "axis", "below": "axis", "above": "legend"}.get(side, "other")
        if letter == "e" and side == "right":
            t["group"] = "legend"
    # everything strictly above the plot (legends, the i statistic line) is
    # scaled as one block so that keys and labels stay together
    legend = [i for i, b in enumerate(graphics) if b[3] < plot[1] - 0.3]
    if letter == "e":   # e's legend sits to the right of the heatmap
        legend += [i for i, b in enumerate(graphics)
                   if b[0] > plot[2] + 2 and (b[2] - b[0]) < 40 and i not in legend]
    # the h legend can sit inside the plot's clip region: recognise it by label
    forced = [t for t in texts if t["group"] != "legend" and LEGEND_LABELS.match(t["s"])]
    if forced:
        for t in forced:
            t["group"] = "legend"
        fy0 = min(t["box"][1] for t in forced) - 3
        fy1 = max(t["box"][3] for t in forced) + 3
        fx0 = min(t["box"][0] for t in forced) - 20
        fx1 = max(t["box"][2] for t in forced) + 5
        for i, b in enumerate(graphics):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if (b[2] - b[0]) < 25 and (b[3] - b[1]) < 25 and fy0 <= cy <= fy1 \
                    and fx0 <= cx <= fx1 and i not in legend:
                legend.append(i)
    legend_keys = [b for i, b in enumerate(graphics)
                   if (b[2] - b[0]) < 25 and (b[3] - b[1]) < 25 and inter(b, plot) == 0
                   and i not in legend]
    reg = [graphics[i] for i in legend] + [t["box"] for t in texts if t["group"] == "legend"]
    legend_hidden = [el for b, el in hidden if b[3] < plot[1] - 0.3 or (
        reg and b[0] >= min(r[0] for r in reg) - 1 and b[2] <= max(r[2] for r in reg) + 1
        and b[1] >= min(r[1] for r in reg) - 1 and b[3] <= max(r[3] for r in reg) + 1)]
    legend_right = bool(reg) and min(r[0] for r in reg) > plot[2]
    return dict(texts=texts, graphics=graphics, graphic_els=graphic_els, plot=plot,
                keys=legend_keys, legend_g=legend, legend_hidden=legend_hidden,
                legend_right=legend_right, hidden=hidden)
