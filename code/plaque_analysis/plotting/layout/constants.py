"""Figure 2 SVG constants; extracted from the manuscript layout workflow."""
import math
import os
import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"


XLINK_NS = "http://www.w3.org/1999/xlink"


NS = "{%s}" % SVG_NS


PAGE_W = 595.28


LETTER_CAP = 12.17


ROW_GAP = 8.03


TOP_R = 1.03


BOTTOM_MARGIN = 4.37


LETTER_W = 11.0


LETTER_CLEAR = 2.0


AXIS_PT = 8.12


OTHER_PT = 7.0


COL_GAP, RIGHT = 7.0, 592.5


def make_slots(split):
    L = dict(lx=2.36, x0=0.59, x1=split, hmax=1e9)
    R = dict(lx=split + COL_GAP + 2.4, x0=split + COL_GAP, x1=RIGHT, hmax=1e9)
    return {k: dict(L if k in "adeh" else R) for k in "abcdefghi"}


SLOTS = make_slots(225.0)


PAGE_H = 841.89


SUBTITLE_FILLS = {"#555555", "#666666", "#626262"}


PANEL_GAP_X = 9.0


LETTER_INSET = 2.0


RIGHT_EDGE = 592.5


LEGEND_LABELS = re.compile(r"^(Brainstem status|BS retained|BS absent)$")


LEGEND_K_MAX = 1.8


MIN_SCALE_KEEP = 0.8


_W = {" ": 278, "(": 333, ")": 333, ",": 278, "-": 333, ".": 278, "/": 278,
      ":": 278, "=": 584, "+": 584, "\u2212": 584, "\u00b2": 333, "\u03b2": 575,
      "\u03bc": 577, "\u00b5": 577, "\u2265": 549, "\u0394": 668, "%": 889,
      "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
      "H": 722, "I": 278, "L": 556, "M": 833, "O": 778, "P": 667, "R": 722,
      "S": 667, "T": 611, "X": 667, "a": 556, "b": 556, "c": 500, "d": 556,
      "e": 556, "f": 278, "g": 556, "h": 556, "i": 222, "j": 222, "k": 500,
      "l": 222, "m": 833, "n": 556, "o": 556, "p": 556, "q": 556, "r": 333,
      "s": 500, "t": 278, "u": 556, "v": 500, "w": 722, "x": 500, "y": 500,
      "z": 500}


RENAMES = {
    ("a", -90): {"LAMP1-related pathology area (\u00b5m\u00b2)": "LAMP1-related area (\u03bcm\u00b2, log)"},
    ("a", 0): {"A\u03b2 plaque area (\u00b5m\u00b2)": "A\u03b2 plaque area (\u03bcm\u00b2, log)"},
    ("b", -90): {"LAMP1-related pathology area (\u00b5m\u00b2, log scale)": "LAMP1-related area (\u03bcm\u00b2, log)"},
    ("c", -90): {"Matched A\u03b2 plaque area (\u00b5m\u00b2)": "Matched A\u03b2 area (\u03bcm\u00b2, log)"},
    ("d", -90): {"Cross-validated prediction (\u00b5m\u00b2)": "Out-of-fold prediction (\u03bcm\u00b2, log)"},
    ("d", 0): {"Observed LAMP1-related area (\u00b5m\u00b2)": "Observed LAMP1-related area (\u03bcm\u00b2, log)"},
    ("e", -90): {"LAMP1-related pathology area (decile)": "LAMP1-related area (decile)"},
    ("f", -90): {"GAM residual (log1p scale)": "GAM residual (log1p)"},
}


H_LABEL = re.compile(r"^(.*) \(n=(\d+)\)$")


ET.register_namespace("", SVG_NS)


ET.register_namespace("xlink", XLINK_NS)
