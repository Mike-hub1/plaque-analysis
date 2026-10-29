"""Figure 2 SVG legend; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.geometry import nums
from plaque_analysis.plotting.layout.constants import LEGEND_K_MAX, NS, OTHER_PT


def legend_region(info):
    boxes = [info["graphics"][i] for i in info["legend_g"]] + \
            [t["box"] for t in info["texts"] if t["group"] == "legend"]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def legend_transform(info, s):
    """Block scale (k, anchor, shift) for the region above the plot."""
    reg = legend_region(info)
    fs = [t["fs"] for t in info["texts"] if t["group"] == "legend"]
    if reg is None or not fs:
        return None
    k = min(OTHER_PT / (max(fs) * s), LEGEND_K_MAX)
    if info.get("legend_right"):
        return dict(k=k, ax=reg[0], ay=reg[1], dx=0.0)
    rest = [b for i, b in enumerate(info["graphics"]) if i not in info["legend_g"]] + \
           [t["box"] for t in info["texts"] if t["group"] != "legend"]
    lo = min([reg[0]] + [b[0] for b in rest])
    hi = max([reg[2]] + [b[2] for b in rest])
    k = min(k, (hi - lo) / (reg[2] - reg[0]))
    plot = info["plot"]
    cands = {"left": abs(reg[0] - plot[0]), "right": abs(reg[2] - plot[2]),
             "centre": abs((reg[0] + reg[2]) / 2 - (plot[0] + plot[2]) / 2)}
    side = min(cands, key=cands.get)
    ax = {"left": reg[0], "right": reg[2], "centre": (reg[0] + reg[2]) / 2}[side]
    ay = reg[3]
    nx0, nx1 = ax + (reg[0] - ax) * k, ax + (reg[2] - ax) * k
    dx = 0.0
    if nx0 < lo:
        dx = lo - nx0
    elif nx1 > hi:
        dx = hi - nx1
    return dict(k=k, ax=ax, ay=ay, dx=dx)


def lt_box(b, lt):
    if lt is None:
        return b
    f = lambda x, a: a + (x - a) * lt["k"]
    return (f(b[0], lt["ax"]) + lt["dx"], f(b[1], lt["ay"]),
            f(b[2], lt["ax"]) + lt["dx"], f(b[3], lt["ay"]))


def relayout_e_legend(root, info):
    """Move e's legends to the right of the heatmap as in the template:
    a vertical colorbar (+9 top, -9 bottom) under its title, then the plaque
    count key as a column of circles.  Works in the panel's own units."""
    P, G, B = info["plot"], info["graphic_els"], info["graphics"]
    leg = info["legend_g"]
    imgs = [i for i in leg if G[i].tag == NS + "image"]
    if not imgs:
        return False
    ib = B[imgs[0]]
    img = G[imgs[0]]
    iw, ih = ib[2] - ib[0], ib[3] - ib[1]
    ticks = [G[i] for i in leg if G[i].tag == NS + "polyline"
             and ib[0] - 1 <= B[i][0] and B[i][2] <= ib[2] + 1
             and ib[1] - 1 <= B[i][1] and B[i][3] <= ib[3] + 1]
    circles = [G[i] for i in leg if G[i].tag == NS + "circle"]
    lt = [t for t in info["texts"] if t["group"] == "legend"]
    cb_titles = sorted([t for t in lt if t["y"] < ib[1]], key=lambda t: t["y"])
    cb_labels = [t for t in lt if ib[3] < (t["box"][1] + t["box"][3]) / 2 < ib[3] + 14]
    size_title = [t for t in lt if t["s"].startswith("Plaque count")]
    cys = [float(c.get("cy")) for c in circles]
    size_labels = sorted([t for t in lt if t not in size_title and cys
                          and abs((t["box"][1] + t["box"][3]) / 2 - cys[0]) < 6],
                         key=lambda t: t["x"])
    circles = sorted(circles, key=lambda c: float(c.get("cx")))
    parent_of = {c: p for p in root.iter() for c in p}
    # three size keys, as in the template: smallest, largest and the existing
    # key whose radius lies closest to halfway between them (area ~ count)
    if len(circles) > 3 and len(size_labels) == len(circles):
        vals = [float(nums(t["s"])[0]) for t in size_labels]
        mid_r = ((math.sqrt(vals[0]) + math.sqrt(vals[-1])) / 2) ** 2
        m = min(range(1, len(vals) - 1), key=lambda k: abs(vals[k] - mid_r))
        keep = [0, m, len(vals) - 1]
        for k in range(len(vals)):
            if k not in keep:
                parent_of[circles[k]].remove(circles[k])
                parent_of[size_labels[k]["el"]].remove(size_labels[k]["el"])
        circles = [circles[k] for k in keep]
        size_labels = [size_labels[k] for k in keep]
    # the legend's white key backgrounds are no longer needed
    for b, el in info["hidden"]:
        if b[3] < P[1] - 0.3 and el in parent_of:
            parent_of[el].remove(el)
    rest_x1 = max(b[2] for i, b in enumerate(B) if i not in leg)
    L = max(P[2], rest_x1) + 6.0
    fs = cb_titles[0]["fs"] if cb_titles else 5.9
    gap = (cb_titles[1]["y"] - cb_titles[0]["y"]) if len(cb_titles) > 1 else 1.1 * fs

    def place_text(t, x, y, anchor="start"):
        el = t["el"]
        el.set("x", f"{x:.2f}"); el.set("y", f"{y:.2f}")
        el.set("text-anchor", anchor)
        st = el.get("style", "")
        if "text-anchor" in st:
            el.set("style", re.sub(r"text-anchor:\s*\w+;?", "", st))

    y = P[1] + 0.76 * fs
    for t in cb_titles:
        place_text(t, L, y)
        y += gap
    bar_top = y - gap + 0.22 * fs + 0.5 * fs + 3.0   # room for the "+9" label
    Lb = max(iw, 0.33 * (P[3] - P[1]))
    bottom = bar_top + Lb
    g = ET.Element(NS + "g", {
        "transform": (f"translate({L:.2f} {bottom:.2f}) rotate(-90) "
                      f"scale({Lb / iw:.5f} 1) translate({-ib[0]:.2f} {-ib[1]:.2f})"),
        "data-bbox": f"{L:.2f} {bar_top:.2f} {L + ih:.2f} {bottom:.2f}"})
    par = parent_of[img]
    par.insert(list(par).index(img), g)
    for el in [img] + ticks:
        parent_of[el].remove(el)
        g.append(el)
    for t in cb_labels:
        u = t["x"] - ib[0]
        place_text(t, L + ih + 1.8, bottom - u * Lb / iw + 0.3 * t["fs"])
    y = bottom + 9.0
    for t in size_title:
        place_text(t, L, y)
    if circles:
        rmax = max(float(c.get("r")) for c in circles)
        labels = size_labels + [None] * (len(circles) - len(size_labels))
        lw = max([t["w"] for t in size_labels] + [0.0])
        top = y + 0.22 * fs + 2.5
        avail = max(P[3] - top, 4 * rmax)     # keep the key within the heatmap height
        rowh = lambda c: 2 * max(float(c.get("r")), 2.2) + 2.2
        ncol = 1
        while ncol < len(circles):
            per = math.ceil(len(circles) / ncol)
            cols = [circles[i:i + per] for i in range(0, len(circles), per)]
            if max(sum(rowh(c) for c in col) for col in cols) <= avail:
                break
            ncol += 1
        per = math.ceil(len(circles) / ncol)
        colw = 2 * rmax + 3.0 + lw + 5.0
        for i, (c, t) in enumerate(zip(circles, labels)):
            col, row = divmod(i, per)
            if row == 0:
                cur = top
            r = float(c.get("r"))
            cx = L + col * colw + rmax
            cy = cur + max(r, 2.2)
            c.set("cx", f"{cx:.2f}"); c.set("cy", f"{cy:.2f}")
            if t is not None:
                place_text(t, cx + rmax + 3.0, cy + 0.3 * t["fs"])
            cur = cy + max(r, 2.2) + 2.2
    return True


def apply_legend(root, info, lt):
    if lt is None or abs(lt["k"] - 1) < 1e-6 and abs(lt["dx"]) < 1e-6:
        return
    parent_of = {c: p for p in root.iter() for c in p}
    els = [info["graphic_els"][i] for i in info["legend_g"]] + \
          [t["el"] for t in info["texts"] if t["group"] == "legend"] + info["legend_hidden"]
    tr = (f"translate({lt['ax'] + lt['dx']:.3f} {lt['ay']:.3f}) scale({lt['k']:.5f}) "
          f"translate({-lt['ax']:.3f} {-lt['ay']:.3f})")
    for el in els:
        par = parent_of[el]
        idx = list(par).index(el)
        g = ET.Element(NS + "g", {"transform": tr})
        par.remove(el)
        g.append(el)
        par.insert(idx, g)
