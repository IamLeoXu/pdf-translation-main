"""Real PDF cell detection and real-font layout tests for clipped table labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from babeldoc.format.pdf.document_il import Box, Cropbox, Page, PdfParagraph, PdfStyle
from babeldoc.format.pdf.document_il.utils.table_fit import isolated_table_cell_boxes


def load_typesetting(monkeypatch):
    def stub(name, **members):
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    stub("regex", compile=re.compile)
    stub("rtree", index=SimpleNamespace())
    stub("babeldoc.const", WATERMARK_VERSION="offline")
    stub("babeldoc.format.pdf.document_il.utils.fontmap", FontMapper=lambda config: None)
    stub("babeldoc.format.pdf.document_il.utils.formular_helper", update_formula_data=lambda value: None)
    stub("babeldoc.format.pdf.document_il.utils.layout_helper", box_to_tuple=lambda b: (b.x, b.y, b.x2, b.y2))
    stub("babeldoc.format.pdf.translation_config", TranslationConfig=object, WatermarkOutputMode=SimpleNamespace(Watermarked=1))
    spec = importlib.util.spec_from_file_location("cell_typesetting_under_test", ROOT / "babeldoc/format/pdf/document_il/midend/typesetting.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine_module(monkeypatch):
    return load_typesetting(monkeypatch)


def make_source(tmp_path, width=144):
    path = tmp_path / "table.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        for x in (30, 30 + width, 360):
            page.draw_line((x, 70), (x, 114))
        for y in (70, 92, 114):
            page.draw_line((30, y), (360, y))
        page.insert_text((34, 84), "Prufpunkt", fontsize=9)
        page.insert_text((34, 106), "Probenkennung", fontsize=9)
        page.insert_text((34 + width, 84), "200 mm", fontsize=9)
        page.insert_text((34 + width, 106), "DE-IT-03", fontsize=9)
        document.save(path)
    return path


def source_paragraph(box=None):
    return PdfParagraph(box=box or Box(34, 214, 75, 224), pdf_style=PdfStyle(font_id="base", font_size=9),
                        unicode="Prufpunkt", vertical=False, first_line_indent=False, debug_id="cell-1", xobj_id=-1)


def make_engine(module, path, paragraphs):
    font = pymupdf.Font(fontfile=str(ROOT / "fonts/NotoSerif-Regular.ttf"))
    page = Page(page_number=0, cropbox=Cropbox(Box(0, 0, 400, 300)), pdf_paragraph=paragraphs)
    engine = module.Typesetting.__new__(module.Typesetting)
    engine.translation_config = SimpleNamespace(input_file=str(path))
    engine.font_mapper = SimpleNamespace(base_font=font)
    engine.is_cjk = False
    return engine, page, font


def translated_units(module, font, paragraph, text):
    return [module.TypesettingUnit(unicode=char, font=font, font_size=9,
            style=paragraph.pdf_style, xobj_id=-1) for char in text]


@pytest.mark.parametrize("text", ["Punto di collaudo", "Diametro della tubazione",
                                 "Lunghezza della sezione", "Identificativo del campione"])
def test_cell_uses_available_width_and_retains_original_font(engine_module, tmp_path, text):
    path = make_source(tmp_path)
    before = path.read_bytes()
    paragraph = source_paragraph()
    engine, page, font = make_engine(engine_module, path, [paragraph])
    units = translated_units(engine_module, font, paragraph, text)
    scale, rendered = engine._find_optimal_scale_and_layout(paragraph, page, units, 0.65, apply_layout=True)
    assert scale == 1.0
    assert "".join(unit.unicode for unit in rendered) == text
    assert len({round(unit.y, 3) for unit in rendered}) == 1
    assert rendered[0].x == 34
    assert path.read_bytes() == before


def test_long_cell_text_shrinks_and_stays_within_real_glyph_bounds(engine_module, tmp_path):
    path = make_source(tmp_path, width=60)
    paragraph = source_paragraph()
    engine, page, font = make_engine(engine_module, path, [paragraph])
    cell = engine._prepare_table_cell_boxes(page)[id(paragraph)]
    text = "Identificativo del campione di prova"
    scale, rendered = engine._find_optimal_scale_and_layout(
        paragraph, page, translated_units(engine_module, font, paragraph, text), 1.0, apply_layout=True,
    )
    assert 0 < scale < 1
    assert "".join(unit.unicode for unit in rendered).replace(" ", "") == text.replace(" ", "")
    for unit in rendered:
        assert unit.x >= cell.x
        assert unit.box.x2 <= cell.x2 + 0.01
        assert unit.y + font.descender * unit.font_size >= cell.y
        assert unit.y + font.ascender * unit.font_size <= cell.y2


def test_non_table_paragraph_keeps_existing_scale(engine_module, tmp_path):
    path = make_source(tmp_path)
    paragraph = source_paragraph(Box(30, 250, 360, 270))
    engine, page, font = make_engine(engine_module, path, [paragraph])
    scale, _ = engine._find_optimal_scale_and_layout(
        paragraph, page, translated_units(engine_module, font, paragraph, "Testo fuori dalla tabella"),
        0.65, apply_layout=True,
    )
    assert scale == 0.65


def test_cells_with_multiple_paragraphs_do_not_get_overlapping_expansion(engine_module, tmp_path):
    path = make_source(tmp_path)
    first = source_paragraph(Box(34, 214, 75, 224))
    second = source_paragraph(Box(80, 214, 120, 224))
    _, page, _ = make_engine(engine_module, path, [first, second])
    assert isolated_table_cell_boxes(path, page) == {}


def test_missing_source_leaves_existing_layout_path(engine_module, tmp_path):
    _, page, _ = make_engine(engine_module, tmp_path / "absent.pdf", [source_paragraph()])
    assert isolated_table_cell_boxes(tmp_path / "absent.pdf", page) == {}
