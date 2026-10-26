from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babeldoc.format.pdf.new_parser.active_font_backend import ActiveFontFactory
from babeldoc.format.pdf.new_parser.active_font_resource_runtime import create_active_font_resource_runtime
from babeldoc.format.pdf.new_parser.pymupdf_prepared_page_access import load_prepared_pdf_pages
from babeldoc.format.pdf.new_parser.runtime.font_data_runtime import FontMetricsDB
from babeldoc.format.pdf.new_parser.runtime.font_encoding_runtime import EncodingDB, name2unicode
from babeldoc.format.pdf.new_parser.runtime.ps_primitives_runtime import LIT


@pytest.mark.parametrize("name,expected", [
    ("agrave", "à"), ("eacute", "é"), ("germandbls", "ß"), ("Adieresis", "Ä"),
    ("uni00E900E0", "éà"), ("u1F600", "😀"), ("f_f_i", "ffi"),
])
def test_native_glyph_names_resolve(name, expected):
    assert name2unicode(name) == expected


def test_winansi_retains_italian_and_german_accents():
    text = "àèéìòùäöüß"
    encoding = EncodingDB.get_encoding("WinAnsiEncoding")
    assert "".join(encoding[value] for value in text.encode("cp1252")) == text


def test_font_encoding_differences_do_not_modify_base_table():
    custom = EncodingDB.get_encoding("WinAnsiEncoding", [65, LIT("agrave"), LIT("eacute")])
    assert (custom[65], custom[66]) == ("à", "é")
    assert EncodingDB.get_encoding("WinAnsiEncoding")[65] == "A"


def test_core_font_metrics_and_aliases_are_available():
    _, widths = FontMetricsDB.get_metrics("Helvetica")
    assert widths["W"] > widths["i"] > 0
    assert FontMetricsDB.get_metrics("Arial") == FontMetricsDB.get_metrics("Helvetica")
    _, fixed = FontMetricsDB.get_metrics("Courier")
    assert fixed["W"] == fixed["i"] > 0


def test_active_font_factory_uses_real_encoding_and_metrics():
    font = ActiveFontFactory().create_font(None, {
        "Subtype": "Type1", "BaseFont": "Helvetica", "Encoding": "WinAnsiEncoding",
    })
    assert font.unicode_text(0xE8, "") == "è"
    assert font.char_width(ord("W")) > font.char_width(ord("i")) > 0


def test_pdf_page_font_resources_can_be_prepared_and_decoded(tmp_path):
    text = "Qualità, Straße, più"
    pdf_path = tmp_path / "latin-font-resources.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((30, 50), text, fontname="helv", fontsize=12)
        document.save(pdf_path)
    runtime = create_active_font_resource_runtime()
    with load_prepared_pdf_pages(pdf_path) as pages:
        assert len(pages) == 1
        bundle = runtime.build_page_resource_bundle(pages[0].resource_tree)
        font = next(iter(bundle.get_direct_font_map(()).values()))
        decoded = "".join(font.unicode_text(value, "") for value in text.encode("cp1252"))
        assert decoded == text
        assert font.char_width(ord("W")) > font.char_width(ord("i")) > 0
