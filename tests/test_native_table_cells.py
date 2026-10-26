"""Native cell discovery must preserve drawn bounds and complete source lines."""
from __future__ import annotations

import pymupdf
import pytest

from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il.utils.native_table_cells import (
    discover_native_table_cells,
    native_table_cell_text_boxes,
)


def _draw_cell(page, rect, text=None, rounded=True):
    page.draw_rect(rect, color=(1, 0, 0), fill=(1, 1, 0), radius=0.5 if rounded else None)
    if text:
        page.insert_text((rect.x0 + 4, rect.y0 + 13), text, fontsize=9)


def _grid(page, x=25, y=50, width=110, height=20, rows=3):
    cells = []
    for row in range(rows):
        for column in range(2):
            rect = pymupdf.Rect(x + column * (width + 3), y + row * (height + 5),
                                x + column * (width + 3) + width, y + row * (height + 5) + height)
            _draw_cell(page, rect, f"Cell {row}-{column}")
            cells.append(rect)
    return cells


def _top_left(page, cells):
    return [pymupdf.Rect(cell.x, cell.y, cell.x2, cell.y2) * page.transformation_matrix for cell in cells]


def _same_rect(a, b, tolerance=0.05):
    return max(abs(x - y) for x, y in zip(a, b)) < tolerance


def _nonblank_character_centres(page):
    return [((char["bbox"][0] + char["bbox"][2]) / 2,
             (char["bbox"][1] + char["bbox"][3]) / 2)
            for block in page.get_text("rawdict")["blocks"] for line in block.get("lines", ())
            for span in line["spans"] for char in span["chars"] if char["c"].strip()]


def test_rounded_cells_keep_entire_bounds_and_include_small_header():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=400)
        expected = _grid(page)
        header = pymupdf.Rect(25, 25, 90, 45)
        _draw_cell(page, header, "Specs")
        expected.append(header)
        # The whole page's frame and an empty photo frame are not cells.
        page.draw_rect(pymupdf.Rect(5, 5, 395, 395), color=(1, 0, 0))
        page.draw_rect(pymupdf.Rect(260, 60, 380, 180), color=(1, 0, 0), radius=0.2)
        page.insert_text((25, 210), "Ordinary body paragraph", fontsize=10)
        before = document.tobytes(no_new_id=True)
        cells = discover_native_table_cells(page)
        actual = _top_left(page, cells)
        assert len(cells) == 7
        assert all(any(_same_rect(rect, item) for item in actual) for rect in expected)
        assert document.tobytes(no_new_id=True) == before
        # Every source glyph inside this table hits exactly one complete cell.
        for x, y in _nonblank_character_centres(page):
            if y < 130:
                assert sum(rect.contains(pymupdf.Point(x, y)) for rect in actual) == 1


def test_straight_ruled_grid_without_individual_closed_paths():
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=220)
        for x in (20, 130, 240):
            page.draw_line((x, 30), (x, 90))
        for y in (30, 60, 90):
            page.draw_line((20, y), (240, y))
        for row in range(2):
            for column in range(2):
                page.insert_text((26 + column * 110, 47 + row * 30), f"R{row} C{column}", fontsize=10)
        cells = _top_left(page, discover_native_table_cells(page))
        assert len(cells) == 4
        assert all(_same_rect(cell, pymupdf.Rect(20 + c * 110, 30 + r * 30,
                                               130 + c * 110, 60 + r * 30))
                   for cell, (r, c) in zip(cells, ((0, 0), (0, 1), (1, 0), (1, 1))))


def test_isolated_callouts_single_column_and_pictures_are_not_tables():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=500)
        for row in range(3):
            _draw_cell(page, pymupdf.Rect(20, 30 + row * 25, 150, 50 + row * 25), f"Label {row}")
        _draw_cell(page, pymupdf.Rect(230, 300, 360, 320), "Caption")
        _draw_cell(page, pymupdf.Rect(220, 30, 360, 200))
        assert discover_native_table_cells(page) == []


def test_distant_repeated_cards_are_not_a_dense_table():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=600)
        for row in range(3):
            for column in range(2):
                _draw_cell(page, pymupdf.Rect(20 + column * 140, 30 + row * 150,
                                              150 + column * 140, 50 + row * 150), "Card")
        assert discover_native_table_cells(page) == []


@pytest.mark.parametrize("as_tuple", [False, True])
def test_model_table_region_recovers_borderless_rows_and_columns(as_tuple):
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=240)
        for row in range(3):
            page.insert_text((35, 50 + row * 25), f"Left {row}", fontsize=9)
            page.insert_text((165, 50 + row * 25), f"Right {row}", fontsize=9)
        page.insert_text((25, 180), "Body outside the region", fontsize=9)
        region = pymupdf.Rect(20, 30, 260, 115) * ~page.transformation_matrix
        region_box = tuple(region) if as_tuple else Box(*region)
        assert discover_native_table_cells(page) == []
        cells = _top_left(page, discover_native_table_cells(page, [region_box]))
        assert len(cells) == 6
        assert min(rect.x0 for rect in cells) == pytest.approx(20)
        assert max(rect.x1 for rect in cells) == pytest.approx(260)
        assert min(rect.y0 for rect in cells) == pytest.approx(30)
        assert max(rect.y1 for rect in cells) == pytest.approx(115)
        for x, y in _nonblank_character_centres(page):
            if y < 115:
                assert sum(rect.contains(pymupdf.Point(x, y)) for rect in cells) == 1


def test_model_region_accepts_short_one_row_drawn_table():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        expected = _grid(page, rows=1)
        region = pymupdf.Rect(20, 45, 255, 75) * ~page.transformation_matrix
        cells = _top_left(page, discover_native_table_cells(page, [tuple(region)]))
        assert len(cells) == 2
        assert all(any(_same_rect(rect, item) for item in cells) for rect in expected)


def test_complete_line_assignment_includes_overflow_and_excludes_padding():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        for row in range(3):
            _draw_cell(page, pymupdf.Rect(25, 50 + row * 25, 100, 70 + row * 25))
            _draw_cell(page, pymupdf.Rect(103, 50 + row * 25, 210, 70 + row * 25))
            page.insert_text((145, 63 + row * 25), f"Value {row}", fontsize=9)
            # Most glyphs are inside the left cell, the final glyphs overflow.
            page.insert_text((29, 63 + row * 25), "Label_ABCDEFGHI()", fontsize=9)
        cells = discover_native_table_cells(page)
        assigned = native_table_cell_text_boxes(page, cells)
        assert len(cells) == len(assigned) == 6
        assert sum(len(boxes) for boxes in assigned.values()) == 6
        source_boxes = [(index, rect) for index, boxes in assigned.items()
                        for rect in _top_left(page, boxes)]
        for x, y in _nonblank_character_centres(page):
            assert sum(rect.contains(pymupdf.Point(x, y)) for _, rect in source_boxes) == 1
        assert any(rect.x1 > _top_left(page, cells)[index].x1 + 2
                   for index, rect in source_boxes)
        # Exactly the drawn cell is retained for fitting, irrespective of overflow.
        assert _top_left(page, cells)[0].x1 == pytest.approx(100)


def test_leading_spaces_do_not_create_extra_cells():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        expected = _grid(page)
        for row in range(3):
            page.insert_text((3, 63 + row * 25), "          Padded", fontsize=9)
        cells = discover_native_table_cells(page)
        assert len(cells) == len(expected)
        assignment = native_table_cell_text_boxes(page, cells)
        # Trimmed source boxes start at the first visible glyph, not outside cell.
        assert all(box.x >= 25 for boxes in assignment.values() for box in boxes)


def test_cropbox_round_trip_and_rotation_guard():
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=500)
        _grid(page, x=125, y=150)
        page.set_cropbox(pymupdf.Rect(100, 100, 500, 400))
        expected = [drawing["rect"] for drawing in page.get_drawings()]
        actual = _top_left(page, discover_native_table_cells(page))
        assert len(actual) == 6
        assert all(any(_same_rect(rect, item) for item in actual) for rect in expected)
        page.set_rotation(90)
        assert discover_native_table_cells(page) == []
        assert native_table_cell_text_boxes(page, []) == {}


def test_empty_image_only_page_is_untouched():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        for row in range(3):
            for column in range(2):
                _draw_cell(page, pymupdf.Rect(20 + column * 140, 30 + row * 25,
                                              150 + column * 140, 50 + row * 25))
        assert discover_native_table_cells(page) == []



def _cid_il_page(source, keep_first=False):
    from babeldoc.format.pdf.document_il import Page, PdfCharacter, PdfStyle, VisualBbox
    characters = []
    originals = []
    inverse = ~source.transformation_matrix
    for block in source.get_text("rawdict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                for char in span["chars"]:
                    origin = pymupdf.Point(char["origin"]) * inverse
                    rect = pymupdf.Rect(char["bbox"]) * inverse
                    index = len(characters)
                    original = PdfCharacter(
                        char_unicode="preserved" if keep_first and index == 0 else f"(cid:{index + 1})",
                        pdf_character_id=700 + index,
                        pdf_style=PdfStyle(font_id=f"Identity-H-{span['font']}", font_size=span["size"]),
                        box=Box(origin.x, origin.y, origin.x + rect.width, origin.y + span["size"]),
                        visual_bbox=VisualBbox(Box(*rect)), advance=0.5, xobj_id=-1)
                    characters.append(original)
                    originals.append(char["c"])
    return Page(pdf_character=characters), originals


def test_cid_table_repair_recovers_source_unicode_and_spaces_only():
    from babeldoc.format.pdf.document_il import PdfParagraph, PdfParagraphComposition
    from babeldoc.format.pdf.document_il.utils.native_table_cells import repair_native_table_cid_characters
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        for row in range(3):
            for column in range(2):
                rect = pymupdf.Rect(25 + column * 113, 50 + row * 25,
                                     135 + column * 113, 70 + row * 25)
                _draw_cell(page, rect)
                # Whitespace before the label extends outside its drawn cell.
                page.insert_text((20 + column * 113, rect.y0 + 13),
                                 "    速度 油箱" if column == 0 else "    55 kW", fontsize=9, fontname="china-s")
        page.insert_text((25, 190), "Body outside table", fontsize=9)
        il_page, originals = _cid_il_page(page, keep_first=True)
        alias = il_page.pdf_character[1]
        il_page.pdf_paragraph = [PdfParagraph(pdf_paragraph_composition=[PdfParagraphComposition(pdf_character=alias)])]
        before = [(char.pdf_character_id, char.pdf_style, char.box, char.visual_bbox, char.advance)
                  for char in il_page.pdf_character]
        cells = discover_native_table_cells(page)
        count = repair_native_table_cid_characters(page, il_page, cells)
        assert count > 20
        assert il_page.pdf_character[0].char_unicode == "preserved"
        for index, (character, original) in enumerate(zip(il_page.pdf_character, originals)):
            if character.box.y > 150 and index:
                assert character.char_unicode == original
            elif character.box.y < 150:
                assert character.char_unicode.startswith("(cid:")
            glyph_id, style, box, visual, advance = before[index]
            assert character.pdf_character_id == glyph_id
            assert character.pdf_style is style and character.box is box and character.visual_bbox is visual
            assert character.advance == advance
        assert count == sum(not char.char_unicode.startswith("(cid:") for char in il_page.pdf_character) - 1
        assert repair_native_table_cid_characters(page, il_page, cells) == 0


def test_cid_repair_rejects_conflicting_overprinted_native_glyphs():
    from babeldoc.format.pdf.document_il.utils.native_table_cells import repair_native_table_cid_characters
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        _grid(page)
        # Helvetica A and B have the same advance and baseline: geometry alone
        # cannot establish which native character an opaque CID represents.
        page.insert_text((40, 63), "A", fontsize=9)
        page.insert_text((40, 63), "B", fontsize=9)
        il_page, originals = _cid_il_page(page)
        ambiguous = [(char, char.char_unicode) for char, original in zip(il_page.pdf_character, originals)
                     if original in ("A", "B") and abs(char.box.x - 40) < 0.01]
        assert len(ambiguous) == 2
        assert repair_native_table_cid_characters(page, il_page) > 0
        assert all(character.char_unicode == previous for character, previous in ambiguous)
