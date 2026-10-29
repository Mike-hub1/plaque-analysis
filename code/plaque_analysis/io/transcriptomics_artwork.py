"""Read-only export of the author-confirmed final SVG and exact panel groups."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from lxml import etree

from plaque_analysis.io.svg_utils import group, sha256

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def parse_master(path):
    tree = etree.parse(str(path), etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True))
    for letter in "abcdef":
        group(tree, letter)
    if tree.xpath('//*[local-name()="image"]'):
        raise ValueError("The archived master must remain self-contained vector artwork")
    return tree


def _preserve_copy(source, destination):
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise FileExistsError(f"A different final artwork exists: {destination}; use a new output directory")
    else:
        shutil.copy2(source, destination)


def export_final(source_svg, output_dir, layout_file):
    """Copy the complete final SVG unchanged and crop its exact a-f groups."""
    source_svg, output_dir = Path(source_svg).resolve(), Path(output_dir).resolve()
    if output_dir == source_svg.parent or source_svg.parent in output_dir.parents:
        raise ValueError("Final exports must not be written into the source artwork directory")
    tree = parse_master(source_svg)
    before = sha256(source_svg)
    layout = json.loads(Path(layout_file).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _preserve_copy(source_svg, output_dir / "Figure6.svg")
    definitions = tree.getroot().find(f"{{{SVG_NS}}}defs")
    if definitions is None:
        raise ValueError("Final SVG is missing its shared definitions")
    report = {"master_sha256": before, "master_copied_unchanged": True, "panels": {}}
    for letter in "abcdef":
        box = layout["panels"][letter]
        x, y, width, height = (box[k]*72/25.4 for k in ("x", "y", "w", "h"))
        root = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "xlink": XLINK_NS},
                             width=f'{box["w"]}mm', height=f'{box["h"]}mm', version="1.1",
                             viewBox=f"{x:.4f} {y:.4f} {width:.4f} {height:.4f}")
        root.append(deepcopy(definitions))
        etree.SubElement(root, f"{{{SVG_NS}}}rect", x=f"{x:.4f}", y=f"{y:.4f}",
                         width=f"{width:.4f}", height=f"{height:.4f}", fill="#FFFFFF")
        original = group(tree, letter)
        root.append(deepcopy(original))
        target = output_dir / f"Figure6{letter}.svg"
        etree.ElementTree(root).write(str(target), encoding="utf-8", xml_declaration=True)
        report["panels"][letter] = {"file": target.name, "sha256": sha256(target)}
    return report
