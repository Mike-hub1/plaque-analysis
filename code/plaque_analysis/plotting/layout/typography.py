"""Figure 2 SVG typography; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET
from plaque_analysis.plotting.layout.geometry import box_of_text, inter
from plaque_analysis.plotting.layout.legend import lt_box
from plaque_analysis.plotting.layout.constants import AXIS_PT, OTHER_PT


def scaled_layout(info, k_ax, k_ot):
    """New text boxes for group factors; axis-side text pushed outward."""
    plot = info["plot"]
    new = []
    centres = {}
    for t in info["texts"]:
        if t["lg"]:
            centres.setdefault(t["lg"], []).append(t["y"] - 0.27 * t["fs"])
    for t in info["texts"]:
        k = {"axis": k_ax, "other": k_ot}.get(t["group"], 1.0)
        d = 0.27 * t["fs"] * (k - 1)            # keep the glyph centre
        if t["lg"]:                             # multi-line label: spread lines by k
            c = sum(centres[t["lg"]]) / len(centres[t["lg"]])
            d += (t["y"] - 0.27 * t["fs"] - c) * (k - 1)
        a = math.radians(t["ang"])
        x, y = t["x"] - math.sin(a) * d, t["y"] + math.cos(a) * d
        box = box_of_text(x, y, t["ang"], t["fs"] * k, t["w"] * k, t["anchor"])
        if t["group"] == "legend":
            box = lt_box(box, info.get("lt"))
        new.append(dict(x=x, y=y, k=k, box=box))

    def shift(i, dx, dy):
        n = new[i]
        n["x"] += dx; n["y"] += dy
        b = n["box"]
        n["box"] = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    T = info["texts"]
    # below the plot: keep original clearances, working outward
    idx = sorted([i for i, t in enumerate(T) if t["side"] == "below"], key=lambda i: T[i]["box"][1])
    done = []
    for i in idx:
        o = T[i]["box"]
        need = new[i]["box"][1]
        need = max(need, plot[3] + max(0.0, o[1] - plot[3]))
        for j in done:
            oj = T[j]["box"]
            if min(o[2], oj[2]) - max(o[0], oj[0]) > 0 or \
                    min(new[i]["box"][2], new[j]["box"][2]) - max(new[i]["box"][0], new[j]["box"][0]) > 0:
                gap = o[1] - oj[3]
                if gap >= 0:
                    need = max(need, new[j]["box"][3] + gap)
        if need > new[i]["box"][1]:
            shift(i, 0, need - new[i]["box"][1])
        done.append(i)
    # left of the plot
    idx = sorted([i for i, t in enumerate(T) if t["side"] == "left"], key=lambda i: -T[i]["box"][2])
    done = []
    for i in idx:
        o = T[i]["box"]
        need = new[i]["box"][2]
        need = min(need, plot[0] - max(0.0, plot[0] - o[2]))
        for j in done:
            oj = T[j]["box"]
            if min(o[3], oj[3]) - max(o[1], oj[1]) > 0 or \
                    min(new[i]["box"][3], new[j]["box"][3]) - max(new[i]["box"][1], new[j]["box"][1]) > 0:
                gap = oj[0] - o[2]
                if gap >= 0:
                    need = min(need, new[j]["box"][0] - gap)
        if need < new[i]["box"][2]:
            shift(i, need - new[i]["box"][2], 0)
        done.append(i)
    return new


def violations(info, new, group):
    T = info["texts"]
    tol = 0.05
    for i, t in enumerate(T):
        if t["group"] != group:
            continue
        nb = new[i]["box"]
        for j, u in enumerate(T):
            if j == i:
                continue
            if t["lg"] and t["lg"] == u["lg"]:
                continue
            if inter(nb, new[j]["box"]) > tol and inter(t["box"], u["box"]) <= tol:
                return True
        if t["side"] != "inside":
            for kb in info["keys"]:
                if inter(nb, kb) > tol and inter(t["box"], kb) <= tol:
                    return True
        if t["side"] == "right":
            if inter(nb, info["plot"]) > tol and inter(t["box"], info["plot"]) <= tol:
                return True
    return False


def choose_k(info, s, kcap=float("inf")):
    def largest(group):
        v = [t["fs"] for t in info["texts"] if t["group"] == group]
        return max(v) if v else None

    targets = {}
    for group, pt in (("axis", AXIS_PT), ("other", OTHER_PT)):
        L = largest(group)
        targets[group] = min(kcap, pt / (L * s)) if L else 1.0

    def search(group, fixed):
        tgt = targets[group]
        if tgt <= 1.0:
            return tgt
        test = lambda k: violations(info, scaled_layout(
            info, k if group == "axis" else fixed, k if group == "other" else fixed), group)
        if not test(tgt):
            return tgt
        lo, hi = 1.0, tgt
        for _ in range(14):
            mid = (lo + hi) / 2
            if test(mid):
                hi = mid
            else:
                lo = mid
        return lo

    k_ax = search("axis", 1.0)
    k_ot = search("other", k_ax)
    return k_ax, k_ot


def content_bbox(info, new):
    lt = info.get("lt")
    boxes = [lt_box(b, lt) if i in info["legend_g"] else b
             for i, b in enumerate(info["graphics"])] + [n["box"] for n in new]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)), boxes


def apply_text(info, new):
    for t, n in zip(info["texts"], new):
        el, k = t["el"], n["k"]
        if abs(k - 1) < 1e-9:
            continue
        st = el.get("style")
        fs_new = t["fs"] * k
        if st and "font-size" in st:
            el.set("style", re.sub(r"font-size:\s*[\d.]+px", f"font-size: {fs_new:.2f}px", st))
        else:
            el.set("font-size", f"{fs_new:.2f}")
        if el.get("textLength"):
            el.set("textLength", f"{t['w'] * k:.2f}px")
        if t["in_tr"]:
            el.set("transform", re.sub(r"translate\(\s*[-\d.eE]+[ ,]+[-\d.eE]+\s*\)",
                                       f"translate({n['x']:.2f},{n['y']:.2f})",
                                       el.get("transform"), count=1))
        else:
            el.set("x", f"{n['x']:.2f}")
            el.set("y", f"{n['y']:.2f}")
