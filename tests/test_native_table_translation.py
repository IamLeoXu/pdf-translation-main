"""Exercise native-table integration with real IL objects and offline models."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import re
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pymupdf
import pytest

from test_table_cell_fit import engine_module
from test_table_cell_fit import make_engine
from test_table_cell_fit import make_source
from test_table_cell_fit import source_paragraph
from test_table_cell_fit import translated_units
from test_v3_language_guard import v3_modules

from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PageLayout
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import VisualBbox
from babeldoc.format.pdf.document_il.utils.table_fit import isolated_table_cell_boxes

ROOT = Path(__file__).resolve().parents[1]


def _stub(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parser_module(monkeypatch):
    _stub(monkeypatch, "cv2")
    _stub(monkeypatch, "numpy", ndarray=object, bool_=bool, isscalar=lambda value: isinstance(value, (int, float)))
    _stub(monkeypatch, "babeldoc.format.pdf.document_il.utils.extract_char")
    _stub(monkeypatch, "babeldoc.format.pdf.translation_config", TranslationConfig=object)
    return _load(monkeypatch, "native_table_layout_under_test",
                 "babeldoc/format/pdf/document_il/midend/layout_parser.py")


def _rounded_source(document):
    source = document.new_page(width=400, height=300)
    for row in range(3):
        for column in range(2):
            rect = pymupdf.Rect(25 + column * 113, 50 + row * 25,
                                 135 + column * 113, 70 + row * 25)
            source.draw_rect(rect, radius=0.5, color=(1, 0, 0))
            source.insert_text((rect.x0 + 4, rect.y0 + 13), f"Label {row}-{column}", fontsize=9)
    return source


@pytest.mark.parametrize("enabled", [False, True])
def test_layout_parser_only_adds_real_native_cells_when_enabled(parser_module, monkeypatch, enabled):
    page = Page(page_number=0)
    progress = SimpleNamespace(advance=lambda *args: None)
    model = SimpleNamespace(handle_document=lambda *args: [(page, SimpleNamespace(boxes=[]))])
    parser = parser_module.LayoutParser(SimpleNamespace(
        doc_layout_model=model, enable_table_translation=enabled,
        progress_monitor=SimpleNamespace(stage_start=lambda *args: nullcontext(progress))))
    monkeypatch.setattr(parser, "generate_fallback_line_layout_for_page", lambda *args: None)
    with pymupdf.open() as source:
        _rounded_source(source)
        parser.process(Document(page=[page]), source)
    assert len(page.page_layout) == (6 if enabled else 0)
    for layout in page.page_layout:
        assert isinstance(layout, PageLayout)
        assert layout.class_name == "table_text"
        assert layout._native_text_boxes
        # Transient ownership survives copies but does not enter the IL schema.
        assert deepcopy(layout)._native_text_boxes == layout._native_text_boxes
        assert "_native_text_boxes" not in asdict(layout)


def test_suffix_uses_original_cell_owner_and_body_keeps_normal_layout(parser_module, monkeypatch):
    _stub(monkeypatch, "regex", compile=re.compile, UNICODE=re.UNICODE)
    helper = _load(monkeypatch, "native_table_layout_helper_under_test",
                   "babeldoc/format/pdf/document_il/utils/layout_helper.py")
    left = parser_module.NativeTableTextLayout(id=1, box=Box(20, 50, 100, 70), class_name="table_text")
    left._native_text_boxes = [Box(24, 53, 109, 65)]
    right = parser_module.NativeTableTextLayout(id=2, box=Box(103, 50, 210, 70), class_name="table_text")
    right._native_text_boxes = [Box(145, 53, 190, 65)]
    body = PageLayout(id=3, box=Box(20, 100, 220, 120), class_name="text")
    layouts = {1: left, 2: right, 3: body}
    index = SimpleNamespace(intersection=lambda *args: layouts.keys())
    suffix = PdfCharacter(char_unicode=")", box=Box(105, 54, 109, 64),
                          visual_bbox=VisualBbox(Box(105, 54, 109, 64)))
    original = PdfCharacter(char_unicode="a", box=Box(30, 104, 35, 114),
                            visual_bbox=VisualBbox(Box(30, 104, 35, 114)))
    assert helper.get_character_layout(suffix, index, layouts).id == 1
    assert helper.get_character_layout(original, index, layouts).id == 3
    assert sum(any(box.x <= 107 <= box.x2 and box.y <= 59 <= box.y2
                   for box in getattr(layout, "_native_text_boxes", ()))
               for layout in layouts.values()) == 1


@pytest.mark.parametrize("enabled,label,accepted", [
    (True, "table_text", True), (False, "table_text", False),
    (True, "text", False), (True, "paragraph", False),
])
def test_short_labels_pass_both_translation_filters_without_changing_body(v3_modules, enabled, label, accepted):
    config = SimpleNamespace(enable_table_translation=enabled, min_text_length=5,
                             disable_rich_text_translate=True, glossary_hard=False)
    paragraph = PdfParagraph(unicode="速度", layout_label=label, debug_id="short-label",
                             vertical=False, xobj_id=-1)
    batch = v3_modules.batch.ILTranslatorLLMOnly.__new__(v3_modules.batch.ILTranslatorLLMOnly)
    batch.translation_config = config
    assert batch._should_translate_paragraph(paragraph) is accepted
    translator = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    translator.translation_config = config
    translator.support_llm_translate = False
    translator._cached_glossaries = []
    text = SimpleNamespace(unicode=paragraph.unicode, placeholders=[], glossary_term_tokens={})
    translator.get_translate_input = lambda *args: text
    result, _ = translator.pre_translate_paragraph(
        paragraph, v3_modules.il.ParagraphTranslateTracker(), {}, {})
    assert (result == "速度") is accepted


def test_native_cell_fits_full_width_and_centers_at_original_font(engine_module, parser_module, tmp_path):
    path = make_source(tmp_path)
    paragraph = source_paragraph(Box(90, 214, 114, 224))
    paragraph.layout_id, paragraph.layout_label = 17, "table_text"
    engine, page, font = make_engine(engine_module, path, [paragraph])
    cell = Box(30, 208, 174, 230)
    layout = parser_module.NativeTableTextLayout(id=17, box=cell, class_name="table_text")
    layout._native_text_boxes = [deepcopy(paragraph.box)]
    page.page_layout = [layout]
    scale, rendered = engine._find_optimal_scale_and_layout(
        paragraph, page, translated_units(engine_module, font, paragraph, "Punto di collaudo"),
        0.65, apply_layout=True)
    assert scale == 1
    assert "".join(unit.unicode for unit in rendered) == "Punto di collaudo"
    assert (min(unit.x for unit in rendered) + max(unit.box.x2 for unit in rendered)) / 2 == pytest.approx(102)
    for unit in rendered:
        assert cell.x <= unit.x <= unit.box.x2 <= cell.x2
        assert unit.y + font.descender * unit.font_size >= cell.y
        assert unit.y + font.ascender * unit.font_size <= cell.y2


def test_multiple_native_paragraphs_in_one_cell_are_not_expanded(parser_module, tmp_path):
    path = make_source(tmp_path)
    first, second = source_paragraph(), source_paragraph(Box(80, 214, 120, 224))
    first.layout_id = second.layout_id = 17
    layout = parser_module.NativeTableTextLayout(id=17, box=Box(30, 208, 174, 230), class_name="table_text")
    layout._native_text_boxes = [first.box, second.box]
    page = Page(page_number=0, page_layout=[layout], pdf_paragraph=[first, second])
    assert isolated_table_cell_boxes(path, page) == {}


def test_existing_ruled_cell_keeps_original_left_alignment(engine_module, tmp_path):
    path = make_source(tmp_path)
    paragraph = source_paragraph()
    engine, page, font = make_engine(engine_module, path, [paragraph])
    scale, rendered = engine._find_optimal_scale_and_layout(
        paragraph, page, translated_units(engine_module, font, paragraph, "Punto di collaudo"),
        0.65, apply_layout=True)
    assert scale == 1
    assert rendered[0].x == 34


@pytest.mark.parametrize("enabled", [False, True])
def test_v3_worker_passes_table_flag_into_actual_translation_call(monkeypatch, tmp_path, enabled):
    calls = []
    def translate(progress, config):
        calls.append(config)
        (Path(config.output_dir) / "input.mono.pdf").write_bytes(b"%PDF-offline-test")
    _stub(monkeypatch, "babeldoc.format.pdf.high_level", do_translate=translate,
          get_translation_stage=lambda config: ())
    _stub(monkeypatch, "babeldoc.format.pdf.translation_config",
          TranslationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
          WatermarkOutputMode=SimpleNamespace(NoWatermark=0, Watermarked=1))
    _stub(monkeypatch, "babeldoc.glossary", Glossary=object)
    _stub(monkeypatch, "babeldoc.progress_monitor", ProgressMonitor=lambda *args, **kwargs: object())
    _stub(monkeypatch, "babeldoc.translator.translator", OpenAITranslator=object)
    _stub(monkeypatch, "babeldoc.docvision.doclayout", DocLayoutModel=SimpleNamespace(load_onnx=lambda: object()))
    _stub(monkeypatch, "config", settings=SimpleNamespace(allow_retranslate=False))
    worker = _load(monkeypatch, "native_table_worker_under_test", "v3_worker.py")
    monkeypatch.setattr(worker, "_build_translator", lambda params: object())
    monkeypatch.setattr(worker, "validate_tracking_outputs", lambda *args: None)
    record = worker.V3Record(input_pdf=tmp_path / "source.pdf", work_dir=tmp_path,
        params=worker.TranslateParams(lang_in="de", lang_out="it", enable_table_translation=enabled))
    worker.run_translate(record)
    assert len(calls) == 1
    assert calls[0].enable_table_translation is enabled
    assert (calls[0].lang_in, calls[0].lang_out) == ("de", "it")
    assert record.progress == 100
    assert record.result_files == ["input.mono.pdf"]




def test_centered_native_cell_preserves_passthrough_numeric_glyph(engine_module, parser_module, tmp_path):
    path = make_source(tmp_path)
    paragraph = source_paragraph(Box(90, 214, 114, 224))
    paragraph.layout_id, paragraph.layout_label = 17, "table_text"
    engine, page, font = make_engine(engine_module, path, [paragraph])
    cell = Box(30, 208, 174, 230)
    layout = parser_module.NativeTableTextLayout(id=17, box=cell, class_name="table_text")
    layout._native_text_boxes = [deepcopy(paragraph.box)]
    page.page_layout = [layout]
    original = PdfCharacter(char_unicode="5", box=Box(100, 214, 105, 223),
        visual_bbox=VisualBbox(Box(100, 214, 105, 223)), pdf_style=paragraph.pdf_style,
        pdf_character_id=5, vertical=False, xobj_id=-1)
    units = [engine_module.TypesettingUnit(char=original)] + translated_units(
        engine_module, font, paragraph, " kW")
    scale, rendered = engine._find_optimal_scale_and_layout(
        paragraph, page, units, 0.65, apply_layout=True)
    assert scale == 1
    assert rendered is not None
    characters = [composition.pdf_character for composition in paragraph.pdf_paragraph_composition]
    assert "".join(char.char_unicode for char in characters) == "5 kW"
    assert (min(char.box.x for char in characters) + max(char.box.x2 for char in characters)) / 2 == pytest.approx(102)
    assert all(cell.x <= char.box.x < char.box.x2 <= cell.x2 for char in characters)


@pytest.mark.parametrize("text,minimum", [("括:", 1), ("ID:", 1), ("名：", 1),
    ("型:", 1), ("نوع:", 1), ("5:", 5), ("x", 5), ("速度", 5)])
def test_short_explicit_labels_are_not_silently_skipped(v3_modules, text, minimum):
    config = SimpleNamespace(enable_table_translation=False, min_text_length=5)
    paragraph = PdfParagraph(unicode=text, layout_label="plain text")
    assert v3_modules.il.minimum_translation_length(paragraph, config) == minimum
