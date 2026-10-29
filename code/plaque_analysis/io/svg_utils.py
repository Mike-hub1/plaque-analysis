"""Small SVG selection and file-digest helpers for artwork exports."""

import hashlib
from pathlib import Path
from lxml import etree


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(tree: etree._ElementTree, panel: str) -> etree._Element:
    found = tree.xpath(f'//*[local-name()="g" and @id="panel_{panel}"]')
    if len(found) != 1:
        raise ValueError(f"Expected exactly one SVG group for panel {panel}")
    return found[0]
