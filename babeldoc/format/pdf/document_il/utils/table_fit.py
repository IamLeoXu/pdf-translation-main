"""Bounds for isolated text paragraphs inside ruled PDF table cells."""
from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from babeldoc.format.pdf.document_il import Box

logger = logging.getLogger(__name__)


def _contains(outer, inner, tolerance=0.7):
    return (outer.x - tolerance <= inner.x and inner.x2 <= outer.x2 + tolerance
            and outer.y - tolerance <= inner.y and inner.y2 <= outer.y2 + tolerance)


def isolated_table_cell_boxes(input_file, page):
    """Return cells with exactly one paragraph; leave multi-paragraph cells alone."""
    if not input_file or not Path(input_file).is_file():
        return {}
    paragraphs = [p for p in page.pdf_paragraph if p.box is not None and not getattr(p, "vertical", False)]
    result = {}
    try:
        with pymupdf.open(input_file) as document:
            source = document[int(page.page_number or 0)]
            # Do not infer bounds for rotated pages from a different coordinate frame.
            if source.rotation:
                return {}
            inverse = ~source.transformation_matrix
            for table in source.find_tables().tables:
                if table.row_count < 2 or table.col_count < 2:
                    continue
                for cell in table.cells:
                    if cell is None:
                        continue
                    rect = pymupdf.Rect(cell) * inverse
                    if rect.width < 12 or rect.height < 5:
                        continue
                    box = Box(rect.x0, rect.y0, rect.x1, rect.y1)
                    occupants = [p for p in paragraphs if _contains(box, p.box)]
                    if len(occupants) != 1:
                        continue
                    paragraph = occupants[0]
                    previous = result.get(id(paragraph))
                    area = rect.width * rect.height
                    if previous is None or area < (previous.x2 - previous.x) * (previous.y2 - previous.y):
                        result[id(paragraph)] = box
    except Exception:
        # Table detection is optional; keep the established non-table layout path.
        logger.debug("Could not infer source table cell bounds", exc_info=True)
    return result


def cell_layout_box(cell, units, scale, source_box=None):
    """Reserve actual font ascenders/descenders around the baseline-based layout."""
    extra_ascent = 0.0
    descent = 0.0
    for unit in units:
        font = getattr(unit, "font", None)
        size = float(getattr(unit, "font_size", None) or 0) * scale
        if font is not None and size:
            extra_ascent = max(extra_ascent, max(0.0, float(getattr(font, "ascender", 1.0)) - 1.0) * size)
            descent = max(descent, max(0.0, -float(getattr(font, "descender", 0.0))) * size)
    padding = 0.6
    left = max(cell.x + padding, source_box.x) if source_box is not None else cell.x + padding
    return Box(left, cell.y + padding + descent,
               cell.x2 - padding, cell.y2 - padding - extra_ascent)
