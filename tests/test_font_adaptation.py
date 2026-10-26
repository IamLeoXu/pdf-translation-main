from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from foundation.config import fonts
from foundation.shared.font_policy import current_font_profile, font_context, resolve_font_profile
from services.rendering.layout.payload.capacity import text_demand_units
from services.rendering.layout.payload import blocks as payload_blocks
from services.rendering.layout.text_measurement import text_width_em
from services.rendering.output.typst import source_pages
from services.rendering.output.typst.overlay_source_cache import overlay_source_fingerprint
from services.rendering.source.prewarm_fingerprint import build_payload_structure_hash
from services.rendering.source.dev_overlay.text_draw import apply_translated_items_to_page


@pytest.fixture(autouse=True)
def automatic_fonts(monkeypatch):
    monkeypatch.setenv("RETAIN_PDF_TYPST_FONT_FAMILY", "auto")
    monkeypatch.delenv("RETAIN_PDF_FONT_PATH", raising=False)
    monkeypatch.delenv("RETAIN_PDF_LATIN_FONT_FAMILY", raising=False)


@pytest.mark.parametrize("language,family", [
    ("en_US", "Noto Serif"), ("it_IT", "Noto Serif"), ("de_DE", "Noto Serif"),
    ("tr_TR", "Noto Serif"), ("zh_CN", "Source Han Serif SC"),
    ("zh_TW", "Noto Serif CJK TC"), ("ja_JP", "Noto Serif CJK JP"),
    ("ar_SA", "Noto Naskh Arabic"),
])
def test_target_language_selects_font(language, family):
    assert resolve_font_profile(language).families[0] == family


@pytest.mark.parametrize("family", ["NotoSerif", "NotoSans"])
@pytest.mark.parametrize("style", ["Regular", "Bold", "Italic", "BoldItalic"])
def test_bundled_styles_cover_italian_german_turkish(family, style):
    font = fitz.Font(fontfile=str(fonts.BACKEND_FONTS_DIR / f"{family}-{style}.ttf"))
    assert all(font.has_glyph(ord(char), fallback=False) for char in "àèéìòùÀÈÉÌÒÙäöüÄÖÜßẞçğışİÇĞÖŞÜ’")


def test_widths_use_selected_font_and_do_not_reuse_other_family_cache():
    text = "L’operatore può verificare qualità, sicurezza e Größe."
    serif = fitz.Font(fontfile=str(fonts.BACKEND_FONTS_DIR / "NotoSerif-Regular.ttf"))
    sans = fitz.Font(fontfile=str(fonts.BACKEND_FONTS_DIR / "NotoSans-Regular.ttf"))
    a = text_width_em(text, target_lang="it", font_family="Noto Serif")
    b = text_width_em(text, target_lang="it", font_family="Noto Sans")
    assert a == pytest.approx(serif.text_length(text, fontsize=1))
    assert b == pytest.approx(sans.text_length(text, fontsize=1))
    assert a != pytest.approx(b)
    assert text_width_em(text, target_lang="it", font_family="Noto Serif") == a


def test_capacity_cache_and_context_follow_custom_family():
    text = "Qualità e Sicherheitsmaßnahmen " * 4
    previous = current_font_profile()
    with font_context([{"target_lang": "it"}], font_family="Noto Serif"):
        serif = text_demand_units(text, [])
        with font_context([{"target_lang": "it"}], font_family="Noto Sans"):
            sans = text_demand_units(text, [])
        assert text_demand_units(text, []) == serif
    assert current_font_profile() == previous
    assert serif != pytest.approx(sans)


def test_font_context_isolated_between_page_workers():
    def measure(family):
        with font_context([{"target_lang": "it"}], font_family=family):
            return current_font_profile().families[0], text_width_em("Qualità e sicurezza")
    with ThreadPoolExecutor(max_workers=2) as executor:
        serif, sans = list(executor.map(measure, ["Noto Serif", "Noto Sans"]))
    assert serif[0] == "Noto Serif" and sans[0] == "Noto Sans"
    assert serif[1] != pytest.approx(sans[1])


def test_approved_chinese_name_uses_explicit_glyph_fallback():
    latin = fitz.Font(fontfile=str(fonts.BACKEND_FONTS_DIR / "NotoSerif-Regular.ttf"))
    han = fitz.Font(fontfile=str(fonts.BACKEND_FONTS_DIR / "SourceHanSerifSC-Regular.otf"))
    expected = latin.text_length("Nome: ", fontsize=1) + han.text_length("上海", fontsize=1)
    assert text_width_em("Nome: 上海", target_lang="it") == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["overlay", "background"])
def test_typst_source_and_layout_share_explicit_family(monkeypatch, tmp_path, mode):
    seen = []
    def build_payloads(*args, **kwargs):
        seen.append(current_font_profile().families[0])
        return [], 0.0
    monkeypatch.setattr(payload_blocks, "build_block_payloads", build_payloads)
    # A new explicit render option takes priority over a stale cached option.
    items = [{"target_lang": "it_IT", "_render_font_family": "Source Han Serif SC"}]
    if mode == "overlay":
        lines = source_pages.build_book_overlay_source_lines(
            [(200, 300, items)], font_family="Noto Sans", block_builder=lambda *_: "")
    else:
        lines = source_pages.build_book_background_source_lines(
            tmp_path / "source.pdf", [(0, 200, 300, items)], work_dir=tmp_path,
            font_family="Noto Sans", block_builder=lambda *_: "")
    source = "\n".join(lines)
    assert seen == ["Noto Sans"]
    assert 'font: ("Noto Sans",' in source and 'lang: "it"' in source
    assert 'font: "auto"' not in source


def test_automatic_latin_profile_ignores_stale_cjk_path(monkeypatch):
    monkeypatch.setenv("RETAIN_PDF_FONT_PATH", str(fonts.BACKEND_FONTS_DIR / "SourceHanSerifSC-Regular.otf"))
    profile = resolve_font_profile("it")
    assert Path(profile.paths[0]).name == "NotoSerif-Regular.ttf"


def test_container_font_directory_works_without_source_fonts(monkeypatch, tmp_path):
    bundled = fonts.BACKEND_FONTS_DIR
    monkeypatch.setattr(fonts, "BACKEND_FONTS_DIR", tmp_path)
    monkeypatch.setenv("RETAIN_PDF_TYPST_FONT_DIRS", str(bundled))
    assert Path(resolve_font_profile("it").paths[0]) == bundled / "NotoSerif-Regular.ttf"


def test_font_or_language_change_invalidates_render_caches():
    item = {"item_id": "a", "block_kind": "text", "block_type": "text", "bbox": [0, 0, 100, 50],
            "target_lang": "it", "protected_translated_text": "Qualità e sicurezza"}
    serif = overlay_source_fingerprint([(200, 300, [item])], font_family="Noto Serif")
    sans = overlay_source_fingerprint([(200, 300, [item])], font_family="Noto Sans")
    assert serif != sans
    initial = build_payload_structure_hash({0: [item]})
    item["_render_font_family"] = "Noto Sans"
    assert build_payload_structure_hash({0: [item]}) != initial
    item["target_lang"] = "de"
    assert overlay_source_fingerprint([(200, 300, [item])], font_family="Noto Serif") != serif


def test_direct_overlay_embeds_target_font_and_retains_accents():
    text = "Qualità, perché, più; Größe, Straße."
    with fitz.open() as document:
        page = document.new_page(width=420, height=180)
        item = {"item_id": "a", "block_type": "text", "block_kind": "text", "should_translate": True,
                "target_lang": "it", "bbox": [20, 20, 400, 120], "translated_text": text,
                "protected_translated_text": text, "source_text": "Prüfen Sie die Sicherheit."}
        apply_translated_items_to_page(page, [item], fonts.DEFAULT_FONT_PATH)
        assert text in page.get_text()
        assert any("NotoSerif" in entry[3].replace(" ", "") for entry in page.get_fonts())
        data = document.tobytes()
    with fitz.open(stream=data, filetype="pdf") as reopened:
        assert text in reopened[0].get_text()


def test_font_sources_match_pinned_checksums_and_include_license():
    manifest = json.loads((fonts.BACKEND_FONTS_DIR / "Noto-fonts.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert hashlib.sha256((fonts.BACKEND_FONTS_DIR / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert "SIL OPEN FONT LICENSE" in (fonts.BACKEND_FONTS_DIR / "Noto-LICENSE.txt").read_text(encoding="utf-8")
