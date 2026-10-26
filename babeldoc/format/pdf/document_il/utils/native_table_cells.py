"""Recover native-text table cells without an OCR or layout-model request.

PyMuPDF uses top-left page coordinates; returned IL ``Box`` objects use PDF
bottom-left coordinates. Text occupancy ignores whitespace and uses glyph
centres, so padded PDF text runs do not move a cell's actual drawn border.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import pymupdf

from babeldoc.format.pdf.document_il import Box


@dataclass
class _TextGroup:
    rect: pymupdf.Rect
    chars: list


def _union(rectangles):
    rectangles = list(rectangles)
    return pymupdf.Rect(min(r.x0 for r in rectangles), min(r.y0 for r in rectangles),
                        max(r.x1 for r in rectangles), max(r.y1 for r in rectangles))


def _contains_point(rect, x, y, tolerance=0.35):
    return (rect.x0 - tolerance <= x <= rect.x1 + tolerance
            and rect.y0 - tolerance <= y <= rect.y1 + tolerance)


def _contains_rect(outer, inner, tolerance=0.7):
    return (outer.x0 - tolerance <= inner.x0 and inner.x1 <= outer.x1 + tolerance
            and outer.y0 - tolerance <= inner.y0 and inner.y1 <= outer.y1 + tolerance)


def _native_text_groups(page):
    groups = []
    for block in page.get_text("rawdict").get("blocks", ()):
        for line in block.get("lines", ()):
            # The native renderer handles vertical / angled text separately.
            direction = line.get("dir", (1, 0))
            if abs(direction[0] - 1) > 0.02 or abs(direction[1]) > 0.02:
                continue
            chars = [char for span in line.get("spans", ())
                     for char in span.get("chars", ()) if char.get("c", "").strip()]
            if not chars:
                continue
            # A PDF line can contain both columns. A real gutter is much wider
            # than normal word spacing; split it before inferring cell bounds.
            gap_limit = max(6.0, median(pymupdf.Rect(c["bbox"]).height for c in chars) * 0.85)
            segment = []
            previous = None
            for char in chars:
                rect = pymupdf.Rect(char["bbox"])
                if previous is not None and rect.x0 - previous.x1 > gap_limit:
                    groups.append(_TextGroup(_union(pymupdf.Rect(c["bbox"]) for c in segment), segment))
                    segment = []
                segment.append(char)
                previous = rect
            if segment:
                groups.append(_TextGroup(_union(pymupdf.Rect(c["bbox"]) for c in segment), segment))
    return groups


def _glyph_centres(group):
    for char in group.chars:
        rect = pymupdf.Rect(char["bbox"])
        yield (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2


def _occupied(rect, groups):
    return any(_contains_point(rect, x, y) for group in groups for x, y in _glyph_centres(group))


def _closed_cell_path(path):
    items = path.get("items", ())
    if len(items) == 1 and items[0][0] == "re":
        return True
    if not 4 <= len(items) <= 16 or any(item[0] not in ("l", "c") for item in items):
        return False
    # A closed circle or arbitrary logo outline is not a cell. Rounded cells
    # still have two horizontal sides, even when closePath is false.
    horizontals = [item for item in items if item[0] == "l" and abs(item[1].y - item[2].y) < 0.7]
    if len(horizontals) < 2:
        return False
    endpoints = [(item[1], item[-1]) for item in items]
    for (_, end), (start, _) in zip(endpoints, endpoints[1:] + endpoints[:1]):
        if abs(end.x - start.x) > 0.8 or abs(end.y - start.y) > 0.8:
            return False
    return True


def _graphic_candidates(page, groups):
    candidates = []
    bounds = page.rect
    for drawing in page.get_drawings():
        if not drawing.get("color") or not _closed_cell_path(drawing):
            continue
        rect = pymupdf.Rect(drawing["rect"])
        if (rect.width < 12 or rect.height < 5 or rect.height > bounds.height * 0.5
                or rect.width > bounds.width * 0.98
                or rect.get_area() > bounds.get_area() * 0.45):
            continue
        if _occupied(rect, groups):
            candidates.append(rect)
    return _deduplicate(candidates)


def _deduplicate(rectangles):
    result = []
    for rect in sorted(rectangles, key=lambda r: r.get_area()):
        if any(max(abs(a - b) for a, b in zip(rect, other)) < 1.0 for other in result):
            continue
        result.append(rect)
    return result


def _repeated_graphic_cells(candidates):
    """Accept repeated columns/rows, then their small header/merged cells."""
    columns = []
    for rect in sorted(candidates, key=lambda r: (r.x0, r.width)):
        match = next((column for column in columns
                      if abs(rect.x0 - median(r.x0 for r in column)) <= max(5, rect.width * 0.04)
                      and abs(rect.width - median(r.width for r in column)) <= max(5, rect.width * 0.08)), None)
        if match is None:
            columns.append([rect])
        else:
            match.append(rect)
    columns = [column for column in columns if len(column) >= 3]
    accepted = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            lrect, rrect = _union(left), _union(right)
            if lrect.x1 > rrect.x0 + 1.0:
                continue
            heights = [r.height for r in left + right]
            height = median(heights)
            if rrect.x0 - lrect.x1 > max(height * 4, min(lrect.width, rrect.width) * 0.75):
                continue
            aligned = [(a, b) for a in left for b in right
                       if abs((a.y0 + a.y1 - b.y0 - b.y1) / 2) <= max(2, height * 0.35)]
            if len(aligned) < 3:
                continue
            # Repeated cards distributed across a page are not a dense table.
            row_centres = sorted(set(round((a.y0 + a.y1) / 2, 1) for a, _ in aligned))
            if median(b - a for a, b in zip(row_centres, row_centres[1:])) > height * 4:
                continue
            core = [r for pair in aligned for r in pair]
            extent = _union(core)
            for rect in candidates:
                if (extent.x0 - 5 <= rect.x0 and rect.x1 <= extent.x1 + 5
                        and extent.y0 - height * 1.8 <= rect.y0
                        and rect.y1 <= extent.y1 + height * 0.5
                        and rect.height <= height * 2.2):
                    accepted.append(rect)
    return _deduplicate(accepted)


def _ruled_cells(page, groups):
    # Avoid a slow line-grid search on vector-heavy brochures with rounded
    # borders: those have horizontal sides but no vertical straight rulings.
    vertical, horizontal = 0, 0
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            if item[0] == "l":
                a, b = item[1:]
                vertical += abs(a.x - b.x) < 0.7 and abs(a.y - b.y) > 8
                horizontal += abs(a.y - b.y) < 0.7 and abs(a.x - b.x) > 12
            elif item[0] == "re":
                vertical += 2
                horizontal += 2
    if vertical < 2 or horizontal < 3:
        return []
    result = []
    try:
        for table in page.find_tables().tables:
            if table.row_count < 2 or table.col_count < 2:
                continue
            for cell in table.cells:
                if cell is None:
                    continue
                rect = pymupdf.Rect(cell)
                if rect.width >= 12 and rect.height >= 5 and _occupied(rect, groups):
                    result.append(rect)
    except (AttributeError, RuntimeError, ValueError):
        # The geometric discovery is a fallback, not a reason to fail a job.
        pass
    return result


def _borderless_cells(region, groups, known):
    remaining = [group for group in groups
                 if _contains_rect(region, group.rect, 2)
                 and not any(sum(_contains_point(cell, x, y) for x, y in _glyph_centres(group))
                             >= max(1, len(group.chars) * 0.6) for cell in known)]
    if not remaining:
        return []
    rows = []
    for group in sorted(remaining, key=lambda g: ((g.rect.y0 + g.rect.y1) / 2, g.rect.x0)):
        centre = (group.rect.y0 + group.rect.y1) / 2
        row = next((row for row in rows
                    if abs(centre - median((g.rect.y0 + g.rect.y1) / 2 for g in row))
                    <= max(2, group.rect.height * 0.4)), None)
        if row is None:
            rows.append([group])
        else:
            row.append(group)
    for row in rows:
        row.sort(key=lambda g: g.rect.x0)
    # Infer full cells only for a consistent row/column arrangement. Otherwise
    # retain each native line's own box; never swallow the entire model region.
    counts = [len(row) for row in rows]
    if len(rows) < 2 or len(set(counts)) != 1 or counts[0] < 2:
        return [pymupdf.Rect(group.rect) for group in remaining]
    count = counts[0]
    boundaries = [region.x0]
    for column in range(count - 1):
        right = max(row[column].rect.x1 for row in rows)
        left = min(row[column + 1].rect.x0 for row in rows)
        if left - right < 3:
            return [pymupdf.Rect(group.rect) for group in remaining]
        boundaries.append((right + left) / 2)
    boundaries.append(region.x1)
    centres = [median((g.rect.y0 + g.rect.y1) / 2 for g in row) for row in rows]
    row_bounds = [region.y0] + [(a + b) / 2 for a, b in zip(centres, centres[1:])] + [region.y1]
    return [pymupdf.Rect(boundaries[c], row_bounds[r], boundaries[c + 1], row_bounds[r + 1])
            for r in range(len(rows)) for c in range(count)]


def discover_native_table_cells(source_page, table_regions=()) -> list[Box]:
    """Return complete, text-bearing native table cell bounds in IL coordinates.

    ``table_regions`` contains optional model table ``Box`` objects or IL
    ``(x, y, x2, y2)`` tuples. Without them, discovery requires a ruled grid or
    at least two repeated graphic columns with three matching rows. Only
    existing horizontal native PDF text participates; image-only content is
    untouched. Cropbox translation is handled by the page matrix. Rotated
    pages are deliberately skipped to avoid mixing parser coordinate frames.
    Font family and size are neither required nor changed by this helper.
    """
    if source_page.rotation:
        return []
    groups = _native_text_groups(source_page)
    if not groups:
        return []
    candidates = _graphic_candidates(source_page, groups)
    result = _repeated_graphic_cells(candidates) + _ruled_cells(source_page, groups)
    for box in table_regions or ():
        values = (box.x, box.y, box.x2, box.y2) if hasattr(box, "x2") else tuple(box)
        region = pymupdf.Rect(values) * source_page.transformation_matrix
        if region.is_empty or region.is_infinite:
            continue
        enclosed = [cell for cell in candidates if _contains_rect(region, cell, 2)]
        # Discard an outer frame when there are actual smaller cells inside it.
        enclosed = [cell for cell in enclosed if not any(
            other.get_area() < cell.get_area() * 0.8 and _contains_rect(cell, other)
            for other in enclosed)]
        result.extend(enclosed)
        result.extend(_borderless_cells(region, groups, result))
    inverse = ~source_page.transformation_matrix
    result = _deduplicate(result)
    result.sort(key=lambda rect: (round(rect.y0, 1), rect.x0))
    return [Box(rect.x0, rect.y0, rect.x1, rect.y1)
            for rect in (rect * inverse for rect in result)]


def native_table_cell_text_boxes(source_page, cells) -> dict[int, list[Box]]:
    """Associate whole native text groups with one discovered cell each.

    Some existing PDFs draw a label a few points outside its border. Keep the
    layout/fit boundary unchanged and use these source-text boxes to associate
    the overflowing suffix with the same cell as the rest of its line. Keys
    are indices in ``cells``. Whitespace is excluded; every text group has at
    most one owner, chosen by the most nonblank glyph centres inside a cell.
    """
    if source_page.rotation:
        return {}
    matrix = source_page.transformation_matrix
    inverse = ~matrix
    rectangles = [pymupdf.Rect(cell.x, cell.y, cell.x2, cell.y2) * matrix for cell in cells]
    assigned = {}
    for group in _native_text_groups(source_page):
        centres = list(_glyph_centres(group))
        counts = [sum(_contains_point(rect, x, y) for x, y in centres) for rect in rectangles]
        if not counts or max(counts) < max(1, len(centres) * 0.5):
            continue
        maximum = max(counts)
        choices = [index for index, count in enumerate(counts) if count == maximum]
        # For a tie prefer the smaller containing cell, never its outer frame.
        owner = min(choices, key=lambda index: rectangles[index].get_area())
        rect = group.rect * inverse
        assigned.setdefault(owner, []).append(Box(rect.x0, rect.y0, rect.x1, rect.y1))
    return assigned


def _page_native_characters(il_page):
    """Yield existing native IL glyphs once, including pre-existing lines."""
    seen = set()
    characters = list(getattr(il_page, "pdf_character", ()))
    for paragraph in getattr(il_page, "pdf_paragraph", ()):
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_character is not None:
                characters.append(composition.pdf_character)
            for name in ("pdf_line", "pdf_same_style_characters", "pdf_formula"):
                container = getattr(composition, name, None)
                if container is not None:
                    characters.extend(container.pdf_character)
    for character in characters:
        if id(character) not in seen:
            seen.add(id(character))
            yield character


def repair_native_table_cid_characters(source_page, il_page, cells=None, text_boxes=None) -> int:
    """Restore table-only CID text from unambiguous existing native PDF glyphs.

    Call before ParagraphFinder groups and filters the IL characters. Only
    ``char_unicode == '(cid:digits)'`` is changed. A source glyph must have an
    almost identical baseline origin and advance width, and its font box must
    contain the IL visual glyph. The existing glyph ID, font, geometry and
    drawing instructions are untouched. Unknown, synthetic or conflicting
    source characters are left alone rather than inferred from font IDs.
    Whitespace from a positively identified table line is included so a CID
    space cannot split a table label into extra paragraphs.
    """
    import re

    if source_page.rotation:
        return 0
    if cells is None:
        cells = discover_native_table_cells(source_page)
    if not cells:
        return 0
    if text_boxes is None:
        text_boxes = native_table_cell_text_boxes(source_page, cells)
    owners = [pymupdf.Rect(box.x, box.y, box.x2, box.y2)
              for boxes in text_boxes.values() for box in boxes]
    if not owners:
        return 0
    inverse = ~source_page.transformation_matrix
    source_glyphs = []
    for block in source_page.get_text("rawdict").get("blocks", ()):
        for line in block.get("lines", ()):
            direction = line.get("dir", (1, 0))
            if abs(direction[0] - 1) > 0.02 or abs(direction[1]) > 0.02:
                continue
            glyphs = [(char, span.get("size", 0)) for span in line.get("spans", ())
                      for char in span.get("chars", ())]
            selected = []
            visible = []
            for char, size in glyphs:
                rect = pymupdf.Rect(char["bbox"]) * inverse
                x, y = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
                is_table = any(_contains_point(owner, x, y) for owner in owners)
                selected.append(is_table)
                if char.get("c", "").strip():
                    visible.append(is_table)
            line_is_table = visible and sum(visible) >= max(1, len(visible) * 0.8)
            for (char, size), is_table in zip(glyphs, selected):
                text = char.get("c", "")
                if (len(text) != 1 or text == "\ufffd" or char.get("synthetic")
                        or not (text.isprintable() or text.isspace())
                        or not (is_table or (line_is_table and text.isspace()))
                        or "origin" not in char):
                    continue
                rect = pymupdf.Rect(char["bbox"]) * inverse
                origin = pymupdf.Point(char["origin"]) * inverse
                source_glyphs.append((" " if text.isspace() else text, rect, origin, size))
    repaired = 0
    for character in _page_native_characters(il_page):
        if re.fullmatch(r"\(cid:\d+\)", character.char_unicode or "") is None or character.box is None:
            continue
        box = character.box
        visual = getattr(character, "visual_bbox", None)
        visual = getattr(visual, "box", None) or box
        visual_rect = pymupdf.Rect(visual.x, visual.y, visual.x2, visual.y2)
        font_size = float(getattr(getattr(character, "pdf_style", None), "font_size", 0) or 0)
        matches = []
        for text, rect, origin, size in source_glyphs:
            tolerance = max(0.15, min(font_size or size or 8, 24) * 0.025)
            if abs(box.x - origin.x) > tolerance or abs(box.y - origin.y) > tolerance:
                continue
            width = box.x2 - box.x
            if abs(width - rect.width) > max(0.25, max(width, rect.width) * 0.08):
                continue
            if font_size and size and abs(font_size - size) > max(0.2, size * 0.04):
                continue
            intersection = rect & visual_rect
            if visual_rect.get_area() > 0 and intersection.get_area() < visual_rect.get_area() * 0.8:
                continue
            matches.append(text)
        # Overprinted / conflicting glyphs at one position are not evidence of
        # which Unicode belongs to this CID. Duplicate identical text is safe.
        if len(set(matches)) == 1:
            character.char_unicode = matches[0]
            repaired += 1
    return repaired
