from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from foundation.shared.language import bind_language_context
from foundation.shared.language import language_name
from foundation.shared.language import normalize_language_code
from services.translation.llm.result_validator import validate_batch_result
from services.translation.llm.shared.control_context import build_translation_control_context
from services.translation.llm.shared.orchestration.formula_segment_path import try_formula_segment_path
from services.translation.llm.shared.orchestration.segment_prompts import build_formula_segment_messages
from services.translation.llm.validation.errors import WrongLanguageError
from services.translation.llm.validation.quality import review_translation_item
from services.translation.services.context.session_context import build_translation_context
from services.translation.services.postprocess.garbled_reconstruction import GarbledReconstructionRuntime
from services.translation.services.postprocess.garbled_reconstruction import _apply_reconstruction
from services.translation.services.postprocess.garbled_reconstruction import _repair_item_translation
from services.rendering.layout.page_fit import should_fit_wrapped_markdown
from services.rendering.layout.payload.text_common import layout_density_ratio
from services.rendering.layout.payload.text_common import translation_density_ratio
from services.rendering.layout.text_analysis import analyze_text
from services.rendering.layout.text_measurement import text_width_em
from services.rendering.output.typst.source_pages import typst_language_settings


GERMAN = (
    "Vor dem Einschalten der Maschine muss der Bediener alle Schutzvorrichtungen "
    "prüfen und die in diesem Handbuch beschriebenen Sicherheitsmaßnahmen vollständig beachten."
)
ITALIAN = (
    "Prima di accendere la macchina, l'operatore deve controllare tutti i dispositivi "
    "di protezione e rispettare le misure di sicurezza descritte in questo manuale."
)
CHINESE = "启动机器前，操作人员必须检查所有防护装置，并完全遵守本手册中描述的安全措施。"


def item(**updates):
    return {
        "item_id": "de-it-regression", "block_type": "text", "raw_block_type": "text",
        "block_kind": "text", "semantic_role": "body_text", "should_translate": True,
        "source_text": GERMAN, "source_lang": "de", "target_lang": "it",
        "bbox": [0.0, 0.0, 100.0, 20.0], **updates,
    }


@pytest.mark.parametrize("locale,name", [
    ("en_US", "English (US)"), ("it_IT", "Italiano"), ("de_DE", "Deutsch"),
    ("zh_CN", "简体中文"), ("zh_TW", "繁體中文"), ("ja_JP", "日本語"),
    ("tr_TR", "Türkçe"), ("ar_SA", "العربية"),
])
def test_locale_names(locale, name):
    assert language_name(locale) == name
    assert "_" not in normalize_language_code(locale)


def test_explicit_target_does_not_inherit_chinese_language_name(monkeypatch):
    monkeypatch.setenv("RETAIN_TARGET_LANG", "zh-CN")
    monkeypatch.setenv("RETAIN_TARGET_LANGUAGE_NAME", "简体中文")
    context = build_translation_context(target_lang="it_IT")
    assert context.target_lang == "it-IT"
    assert context.target_language_name == "Italiano"
    assert build_translation_control_context(target_lang="it").target_language_name == "Italiano"


def test_low_level_fallback_context_uses_worker_language(monkeypatch):
    monkeypatch.setenv("RETAIN_TARGET_LANG", "it")
    context = build_translation_control_context()
    assert context.target_lang == "it"
    assert context.target_language_name == "Italiano"


@pytest.mark.parametrize("style", ["tagged", "json"])
@pytest.mark.parametrize("locale,name", [("it_IT", "Italiano"), ("de_DE", "Deutsch"), ("zh_TW", "繁體中文")])
def test_formula_prompts_use_item_target(style, locale, name):
    messages = build_formula_segment_messages(
        item(target_lang=locale), [("text", "1")],
        [{"segment_id": "1", "source_text": GERMAN}], response_style=style,
    )
    assert name in messages[0]["content"]
    assert "Simplified Chinese" not in messages[0]["content"]


def test_formula_route_binds_context_before_calling_translator():
    context = build_translation_context(source_lang="de", target_lang="it")
    captured = []

    def translate(bound_item, **kwargs):
        captured.append(bound_item)
        return {bound_item["item_id"]: {"decision": "translate", "translated_text": ITALIAN}}

    result = try_formula_segment_path(
        item(target_lang="zh"), api_key="unused", model="offline", base_url="unused",
        request_label="", context=context, diagnostics=None, allow_transport_tail_defer=False,
        is_transport_error_fn=lambda exc: False, formula_segment_translator_fn=translate,
    )
    assert captured[0]["target_lang"] == "it"
    assert captured[0]["target_language_name"] == "Italiano"
    assert result[item()["item_id"]]["translated_text"] == ITALIAN


def test_garbled_repair_prompt_uses_explicit_runtime_language():
    captured = []

    def request(messages, **kwargs):
        captured.append(messages)
        return json.dumps({"translated_text": ITALIAN})

    runtime = GarbledReconstructionRuntime(
        api_key="unused", model="offline", base_url="unused", provider_reason="test",
        request_chat_content_fn=request, target_lang="it", target_language_name="Italiano",
    )
    assert _repair_item_translation(item(), runtime=runtime) == ITALIAN
    assert "Italiano" in captured[0][0]["content"]
    assert "Chinese" not in captured[0][0]["content"]
    runtime_without_name = GarbledReconstructionRuntime(
        api_key="unused", model="offline", base_url="unused", provider_reason="test",
        request_chat_content_fn=request, target_lang="it",
    )
    _repair_item_translation(item(target_lang="zh", target_language_name="简体中文"), runtime=runtime_without_name)
    assert "Italiano" in captured[-1][0]["content"]


@pytest.mark.parametrize("math_mode", ["placeholder", "direct_typst"])
def test_italian_passes_quality_and_chinese_is_rejected(math_mode):
    source = item(math_mode=math_mode)
    correct = {"decision": "translate", "translated_text": ITALIAN}
    wrong = {"decision": "translate", "translated_text": CHINESE}
    assert not review_translation_item(source, correct).issues
    assert any(issue.kind == "wrong_target_language" for issue in review_translation_item(source, wrong).issues)
    with pytest.raises(WrongLanguageError):
        validate_batch_result([source], {source["item_id"]: wrong})


def test_chinese_task_still_accepts_chinese():
    source = item(target_lang="zh-CN")
    assert not review_translation_item(source, {"decision": "translate", "translated_text": CHINESE}).has_errors


def test_chinese_names_already_in_source_are_preserved():
    source = item(source_text="Die Universität 上海交通大学 betreibt dieses Labor.")
    translated = "Il laboratorio appartiene all'università 上海交通大学."
    assert not review_translation_item(source, {"translated_text": translated}).has_errors


def test_explicit_glossary_literal_is_allowed():
    context = build_translation_control_context(
        source_lang="de", target_lang="it", target_language_name="Italiano",
        glossary_entries=[{"source": "Name", "target": "上海交通大学"}],
    )
    source = bind_language_context(item(source_text="Name"), context)
    assert not review_translation_item(source, {"translated_text": "上海交通大学"}).has_errors


def test_reconstruction_does_not_commit_wrong_language():
    source = item(final_status="failed")
    assert _apply_reconstruction([source], CHINESE) is False
    assert source["final_status"] == "failed"
    assert "translated_text" not in source
    assert source["translation_diagnostics"]["garbled_reconstruction_rejected"] is True


def test_long_italian_triggers_layout_fit():
    assert should_fit_wrapped_markdown(item(), ITALIAN * 10, font_size_pt=10, leading_em=0.6)
    assert not should_fit_wrapped_markdown(item(bbox=[0, 0, 500, 100]), "Breve testo.", font_size_pt=10, leading_em=0.6)


def test_density_counts_latin_and_grows_with_text():
    short = layout_density_ratio([0, 0, 100, 40], ITALIAN, font_size_pt=10, line_step_pt=16)
    long = layout_density_ratio([0, 0, 100, 40], ITALIAN * 5, font_size_pt=10, line_step_pt=16)
    assert short > 0
    assert long > short
    assert translation_density_ratio(item(), ITALIAN) > 0


def test_accented_words_and_curly_apostrophes_are_not_split():
    analysis = analyze_text("Sicherheitsmaßnahmen l’operatore perché")
    assert analysis.word_count == 3
    assert text_width_em("WWW") > text_width_em("iii") > 0


def test_typst_uses_target_hyphenation_language():
    settings = typst_language_settings([{"target_lang": "it_IT"}])
    assert 'lang: "it"' in settings
    assert "hyphenate: true" in settings
    assert "hyphenate: false" in typst_language_settings([{"target_lang": "zh_CN"}])


def test_unscoped_chinese_glossary_is_excluded_from_italian():
    from retain import _parse_glossary_csv

    rows = "source,target,src_lng,tgt_lng\nMaschine,机器,,\nGerät,dispositivo,de,it\nName,上海交通大学,de,it\n"
    entries = _parse_glossary_csv(rows.encode("utf-8"), "de", "it")
    assert [entry["source"] for entry in entries] == ["Gerät", "Name"]
    assert _parse_glossary_csv("source,target\nMaschine,机器\n".encode("utf-8"), "de", "zh")


@pytest.mark.parametrize("math_mode", ["placeholder", "direct_typst"])
def test_real_retry_flow_recovers_from_chinese_response(monkeypatch, math_mode):
    from services.translation.llm.providers.deepseek import translation_client
    from services.translation.llm.shared.orchestration.single_item_flow import translate_single_item_plain_text_with_retries

    captured = []

    def request(messages, **kwargs):
        captured.append(messages)
        return CHINESE if len(captured) == 1 else ITALIAN

    monkeypatch.setattr(translation_client, "request_chat_content", request)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    context = build_translation_context(source_lang="de", target_lang="it")
    result = translate_single_item_plain_text_with_retries(
        item(math_mode=math_mode), context=context, api_key="unused", model="deepseek-test", base_url="unused",
    )
    assert result[item()["item_id"]]["translated_text"] == ITALIAN
    assert len(captured) >= 2
    assert all("Italiano" in messages[0]["content"] for messages in captured)


@pytest.mark.parametrize("math_mode", ["placeholder", "direct_typst"])
def test_repeated_wrong_language_is_not_accepted(monkeypatch, math_mode):
    from services.translation.llm.providers.deepseek import translation_client
    from services.translation.llm.shared.orchestration.single_item_flow import translate_single_item_plain_text_with_retries

    monkeypatch.setattr(translation_client, "request_chat_content", lambda *args, **kwargs: CHINESE)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    context = build_translation_context(source_lang="de", target_lang="it")
    result = translate_single_item_plain_text_with_retries(
        item(math_mode=math_mode), context=context, api_key="unused", model="deepseek-test", base_url="unused",
    )
    payload = result[item()["item_id"]]
    assert CHINESE not in payload.get("translated_text", "")
    assert payload.get("final_status") != "translated"


def test_cached_chinese_is_retried_for_italian():
    from services.translation.llm.shared.orchestration.batched_plain_cache import split_and_validate_cached_batch

    source = item()
    cached = {source["item_id"]: {"decision": "translate", "translated_text": CHINESE}}
    accepted, pending = split_and_validate_cached_batch(
        [source], model="offline", base_url="unused", context=build_translation_context(target_lang="it"),
        diagnostics=None, request_label="", split_cached_batch_fn=lambda *args, **kwargs: (cached, []),
    )
    assert not accepted
    assert pending == [source]


def test_rejected_garbled_repair_is_not_counted_as_success():
    from services.translation.services.postprocess.garbled_reconstruction import reconstruct_garbled_page_payloads

    runtime = GarbledReconstructionRuntime(
        api_key="unused", model="offline", base_url="unused", provider_reason="test",
        request_chat_content_fn=lambda *args, **kwargs: json.dumps({"translated_text": CHINESE}),
        target_lang="it", target_language_name="Italiano",
    )
    source = item(final_status="failed", page_idx=0)
    result = reconstruct_garbled_page_payloads(
        {0: [source]}, api_key="unused", model="offline", base_url="unused", workers=1, runtime=runtime,
    )
    assert result["garbled_attempted"] == 1
    assert result["garbled_reconstructed"] == 0
    assert result["dirty_pages"] == []


def test_typst_syntax_repair_cannot_introduce_chinese():
    from services.rendering.output.typst.repair import _repair_item_with_llm_for_typst

    source = item(render_protected_text=ITALIAN)
    repaired = _repair_item_with_llm_for_typst(
        source, request_label="offline", api_key="unused", model="offline", base_url="unused",
        request_chat_content_fn=lambda *args, **kwargs: f"<<<PROTECTED_TEXT>>>\n{CHINESE}\n<<<END_PROTECTED_TEXT>>>",
    )
    assert repaired is source
    assert repaired["render_protected_text"] == ITALIAN


def test_page_spec_keeps_language_through_render_conversion():
    from services.rendering.layout.model.models import RenderLayoutBlock, RenderPageSpec
    from services.rendering.output.typst.emitter import build_typst_source_from_page_specs
    from services.rendering.source.prewarm_page_specs import render_layout_block_to_manifest

    block = RenderLayoutBlock(
        block_id="annotation", page_index=0, background_rect=[0, 0, 100, 40],
        content_rect=[0, 0, 100, 40], content_kind="markdown", content_text=ITALIAN,
        plain_text=ITALIAN, math_map=[], font_size_pt=10, leading_em=0.6, target_lang="it-IT",
        fit_to_box=True, fit_min_font_size_pt=8, fit_min_leading_em=0.4, fit_max_height_pt=40,
    )
    assert render_layout_block_to_manifest(block)["target_lang"] == "it-IT"
    spec = RenderPageSpec(page_index=0, page_width_pt=300, page_height_pt=400, background_pdf_path=None, blocks=[block])
    work_dir = Path(__file__).resolve().parent
    source = build_typst_source_from_page_specs(
        background_pdf_path=work_dir / "background.pdf", page_specs=[spec], work_dir=work_dir,
    )
    assert 'lang: "it", hyphenate: true' in source
    assert "emergency_min_size" in source
    assert ITALIAN in source
