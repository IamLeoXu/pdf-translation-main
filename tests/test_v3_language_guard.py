"""Offline regressions for the text-layer engine's Chinese-output failure."""
from __future__ import annotations

import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from babeldoc.translator import language_guard as guard

ROOT = Path(__file__).resolve().parents[1]
MIDEND = "babeldoc.format.pdf.document_il.midend"
SOURCE = "Schlauchliner und Aushärtung"
ITALIAN = "Rivestimento tubolare e indurimento"
CHINESE = "软管内衬和固化"


class MemoryCache:
    def __init__(self, *args):
        self.values = {}
        self.params = dict(args[1]) if len(args) > 1 else {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def add_params(self, key, value):
        self.params[key] = value


class Glossary:
    def __init__(self, name, entries):
        self.name, self.entries = name, entries

    def get_active_entries_for_text(self, text):
        return [(entry.source, entry.target) for entry in self.entries if entry.source in text]


def entry(source, target, target_language=None):
    return SimpleNamespace(source=source, target=target, target_language=target_language)


@pytest.fixture
def v3_modules(monkeypatch):
    """Run actual engine methods; replace network, model and font dependencies only."""
    def stub(name, **members):
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    stub("babeldoc.translator.cache", TranslationCache=MemoryCache)
    stub("openai", RateLimitError=type("RateLimitError", (Exception,), {}),
         BadRequestError=type("BadRequestError", (Exception,), {}))
    stub("tenacity", retry=lambda **kw: lambda fn: fn,
         before_sleep_log=lambda *a: None, retry_if_exception_type=lambda *a: None,
         stop_after_attempt=lambda *a: None, wait_exponential=lambda **kw: None)
    translator = load("babeldoc.translator.translator", "babeldoc/translator/translator.py")
    monkeypatch.setattr(translator._translate_rate_limiter, "wait", lambda *a: None)
    stub("tiktoken", encoding_for_model=lambda *a: SimpleNamespace(encode=lambda text, **kw: list(text)))
    stub("tqdm", tqdm=object)
    stub("Levenshtein", distance=lambda a, b: len(a) + len(b))
    stub("babeldoc.format.pdf.translation_config", TranslationConfig=object, TitleContextSnapshot=object)
    stub("babeldoc.utils.priority_thread_pool_executor", PriorityThreadPoolExecutor=object)
    utils = "babeldoc.format.pdf.document_il.utils"
    stub(utils + ".fontmap", FontMapper=lambda config: None)
    stub(utils + ".layout_helper", get_char_unicode_string=lambda chars: "",
         get_paragraph_unicode=lambda para: para.unicode,
         is_same_style=lambda *a: True, is_same_style_except_font=lambda *a: True,
         is_same_style_except_size=lambda *a: True)
    paragraph_helpers = load("native_minimum_helper", "babeldoc/format/pdf/document_il/utils/paragraph_helper.py")
    stub(utils + ".paragraph_helper", is_placeholder_only_paragraph=lambda p: False,
         is_pure_numeric_paragraph=lambda p: False, is_cid_paragraph=lambda p: False,
         minimum_translation_length=paragraph_helpers.minimum_translation_length)
    stub(utils + ".style_helper", GRAY80=None)
    il = load(MIDEND + ".il_translator", "babeldoc/format/pdf/document_il/midend/il_translator.py")
    import babeldoc.format.pdf.document_il.midend as package
    monkeypatch.setattr(package, "il_translator", il, raising=False)
    batch = load(MIDEND + ".il_translator_llm_only", "babeldoc/format/pdf/document_il/midend/il_translator_llm_only.py")
    term = load(MIDEND + ".automatic_term_extractor", "babeldoc/format/pdf/document_il/midend/automatic_term_extractor.py")
    return SimpleNamespace(translator=translator, il=il, batch=batch, term=term)


def make_translator(modules, responses, language="it"):
    class Translator(modules.translator.BaseTranslator):
        def __init__(self):
            super().__init__("de", language, False)
            self.responses, self.requests = iter(responses), []

        def do_llm_translate(self, text, rate_limit_params=None):
            self.requests.append(text)
            return next(self.responses)

        def do_translate(self, text, rate_limit_params=None):
            return self.do_llm_translate(text, rate_limit_params)

    return Translator()


@pytest.mark.parametrize("locale,name", [
    ("en_US", "English (US)"), ("it_IT", "Italian"), ("de_DE", "German"),
    ("zh_CN", "Simplified Chinese"), ("zh_TW", "Traditional Chinese"),
    ("ja_JP", "Japanese"), ("tr_TR", "Turkish"), ("ar_SA", "Arabic"),
])
def test_unambiguous_target_names_and_localized_notices(locale, name):
    assert guard.language_name(locale) == name
    notice = guard.content_filter_notice(locale)
    assert notice
    assert not guard.unexpected_han(locale, "", notice)


@pytest.mark.parametrize("language", ["it", "it_IT", "en", "de", "tr", "ar"])
def test_rejects_new_chinese_for_non_cjk_targets(language):
    with pytest.raises(guard.WrongTargetLanguageError):
        guard.ensure_target_language(language, SOURCE, CHINESE)


@pytest.mark.parametrize("language", ["zh_CN", "zh_TW", "ja", "ko"])
def test_chinese_and_other_cjk_targets_remain_supported(language):
    guard.ensure_target_language(language, SOURCE, CHINESE)


def test_original_han_and_explicit_literals_are_preserved():
    guard.ensure_target_language("it", "Hersteller 上海", "Produttore 上海")
    guard.ensure_target_language("it", "Hersteller ACME", "Produttore 上海", ["上海"])
    with pytest.raises(guard.WrongTargetLanguageError):
        guard.ensure_target_language("it", "上海", "上海上海")
    with pytest.raises(guard.WrongTargetLanguageError):
        guard.ensure_target_language("it", SOURCE, "神")


@pytest.mark.parametrize("method", ["translate", "llm_translate"])
def test_actual_translator_rejects_cache_and_retries_before_caching(v3_modules, method):
    translator = make_translator(v3_modules, [CHINESE, ITALIAN])
    translator.cache.set(SOURCE, CHINESE)
    assert getattr(translator, method)(SOURCE) == ITALIAN
    assert len(translator.requests) == 2
    if method == "llm_translate":
        assert "Italian only" in translator.requests[1]
    assert translator.cache.get(SOURCE) == ITALIAN
    assert getattr(translator, method)(SOURCE) == ITALIAN
    assert translator.translate_cache_call_count == 1
    assert translator.cache.params["target_language_policy"] == guard.LANGUAGE_POLICY_VERSION


def test_repeated_chinese_never_enters_cache(v3_modules):
    translator = make_translator(v3_modules, [CHINESE, CHINESE])
    with pytest.raises(guard.WrongTargetLanguageError):
        translator.llm_translate(SOURCE)
    assert translator.cache.get(SOURCE) is None


def test_batch_json_checks_output_and_preserves_format_and_tags(v3_modules):
    good = '[{"id":0,"output":"{v1}<style>Indurimento</style>"}]'
    translator = make_translator(v3_modules, ['[{"id":0,"output":"固化"}]', good])
    assert translator.llm_translate(SOURCE) == good
    assert len(translator.requests) == 2


def test_chinese_instructions_do_not_authorize_chinese_translations(v3_modules):
    prompt = "请翻译德语成意大利语；禁止出现软管内衬和固化。\n" + SOURCE
    translator = make_translator(v3_modules, [CHINESE, ITALIAN])
    translator.cache.set(prompt, CHINESE)
    assert translator.llm_translate(prompt, rate_limit_params={"source_text": SOURCE}) == ITALIAN
    assert len(translator.requests) == 2
    assert translator.translate_cache_call_count == 0


def test_scoped_literal_can_pass_response_validation(v3_modules):
    translator = make_translator(v3_modules, ["Produttore 上海"])
    assert translator.llm_translate("Translate ACME into Italian", rate_limit_params={
        "source_text": "ACME", "allowed_target_literals": ["上海"],
    }) == "Produttore 上海"


def test_ignored_cache_does_not_read_or_write(v3_modules):
    translator = make_translator(v3_modules, [ITALIAN])
    translator.cache.set(SOURCE, "cached")
    assert translator.llm_translate(SOURCE, ignore_cache=True) == ITALIAN
    assert translator.cache.get(SOURCE) == "cached"


def test_glossary_filters_auto_chinese_but_keeps_scoped_names():
    terms = Glossary("terms", [entry("Schlauchliner", "软管内衬"),
                              entry("Aushärtung", "indurimento"),
                              entry("ACME", "上海", "it"),
                              entry("Rohr", "管道", "zh")])
    filtered = guard.filter_glossaries([terms, terms], "it")
    assert len(filtered) == 1
    assert [e.target for e in filtered[0].entries] == ["indurimento", "上海"]
    assert len(terms.entries) == 4
    assert guard.allowed_glossary_literals(filtered, "ACME") == ["上海"]
    assert guard.allowed_glossary_literals(filtered, SOURCE) == ["indurimento"]


def make_extractor(modules, response, language="it"):
    captured, accepted = [], []
    tracker = modules.term.ParagraphTermExtractTracker()
    config = SimpleNamespace(lang_out=language, raise_if_cancelled=lambda: None,
                             shared_context_cross_split_part=SimpleNamespace(
                                 user_glossaries=[],
                                 add_raw_extracted_term_pair=lambda a, b: accepted.append((a, b))))
    engine = SimpleNamespace(llm_translate=lambda prompt, **kw: captured.append(prompt) or response)
    extractor = modules.term.AutomaticTermExtractor(engine, config)
    return extractor, tracker, captured, accepted


def test_actual_term_extraction_filters_chinese_in_live_batch(v3_modules):
    response = json.dumps([{"src": "Schlauchliner", "tgt": "软管内衬"},
                           {"src": "Aushärtung", "tgt": "indurimento"}])
    extractor, tracker, captured, accepted = make_extractor(v3_modules, response)
    extractor.shared_context.user_glossaries = [Glossary("legacy", [entry("Schlauchliner", "软管内衬")])]
    batch = SimpleNamespace(paragraphs=[SimpleNamespace(unicode=SOURCE)], tracker=tracker)
    extractor.extract_terms_from_paragraphs(batch, SimpleNamespace(advance=lambda n: None))
    assert accepted == [("Aushärtung", "indurimento")]
    assert "into Italian" in captured[0]
    assert not guard.unexpected_han("it", SOURCE, captured[0])


def test_actual_alternate_term_ingestion_rejects_bad_terms(v3_modules):
    extractor, _, _, accepted = make_extractor(v3_modules, "[]")
    extractor._process_llm_response('[{"src":"Schlauchliner","tgt":"软管内衬"}]', "offline")
    assert accepted == []


def test_actual_batch_prompt_has_no_chinese_example(v3_modules):
    engine = v3_modules.batch.ILTranslatorLLMOnly.__new__(v3_modules.batch.ILTranslatorLLMOnly)
    engine.translation_config = SimpleNamespace(lang_out="it", custom_system_prompt=None)
    engine._cached_glossaries = []
    prompt = engine._build_llm_prompt('[{"id":0,"input":"Aushärtung"}]', None, None, SOURCE)
    assert "into Italian" in prompt
    assert "{v1}<style id='2'>OpenAI</style> 42" in prompt
    assert not guard.unexpected_han("it", SOURCE, prompt)


def test_basic_translator_prompt_names_italian(v3_modules):
    messages = v3_modules.translator.OpenAITranslator.prompt(SimpleNamespace(lang_out="it_IT"), SOURCE)
    assert "into Italian" in messages[1]["content"]


@pytest.mark.parametrize("mode", ["il", "batch"])
def test_engine_raises_before_rendering_after_unresolved_language_error(v3_modules, monkeypatch, tmp_path, mode):
    module = getattr(v3_modules, mode)
    cls = module.ILTranslator if mode == "il" else module.ILTranslatorLLMOnly
    engine = cls.__new__(cls)
    shared = SimpleNamespace(first_paragraph=True)
    engine.shared_context_cross_split_part = shared
    engine._cached_glossaries = []
    engine.translation_config = SimpleNamespace(
        shared_context_cross_split_part=shared, lang_out="it", debug=False,
        working_dir=str(tmp_path), pool_max_workers=1,
        progress_monitor=SimpleNamespace(stage_start=lambda *a: nullcontext(None)),
        get_working_file_path=lambda name: tmp_path / name,
    )
    monkeypatch.setattr(module, "PriorityThreadPoolExecutor", lambda **kw: nullcontext(None))

    def reject_page(page, executor, pbar, tracker, *args):
        paragraph = tracker.new_paragraph()
        paragraph.set_pdf_unicode(SOURCE)
        paragraph.set_input(SOURCE)
        paragraph.target_language_error = "Unresolved wrong target language"

    engine.process_page = reject_page
    if mode == "batch":
        engine.il_translator = SimpleNamespace()
        engine.process_cross_page_paragraph = lambda *a: None
        engine.process_cross_column_paragraph = lambda *a: None
        engine._flush_tracking_loop = lambda *a: None
    docs = SimpleNamespace(page=[SimpleNamespace(pdf_paragraph=[SimpleNamespace(debug_id="1", unicode=SOURCE)])])
    with pytest.raises(guard.WrongTargetLanguageError):
        engine.translate(docs)


def test_postprocessing_blocks_chinese_before_mutating_paragraph(v3_modules):
    il = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    il.translation_config = SimpleNamespace(lang_out="it")
    il._cached_glossaries = []
    il.parse_translate_output = lambda *args: []
    paragraph = SimpleNamespace(unicode=SOURCE, pdf_paragraph_composition=["original"])
    tracker = v3_modules.il.ParagraphTranslateTracker()
    tracker.set_pdf_unicode(SOURCE)
    translation_input = SimpleNamespace(unicode=SOURCE, glossary_term_tokens={})
    with pytest.raises(guard.WrongTargetLanguageError):
        il.post_translate_paragraph(paragraph, tracker, translation_input, CHINESE)
    assert paragraph.unicode == SOURCE
    assert paragraph.pdf_paragraph_composition == ["original"]
    assert not getattr(tracker, "output", None)
    il.post_translate_paragraph(paragraph, tracker, translation_input, ITALIAN)
    assert paragraph.unicode == ITALIAN
    assert tracker.output == ITALIAN


def test_hard_glossary_restoration_is_checked_before_mutation(v3_modules):
    il = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    il.translation_config = SimpleNamespace(lang_out="it")
    il._cached_glossaries = []
    il.parse_translate_output = lambda *args: []
    paragraph = SimpleNamespace(unicode="Aushärtung", pdf_paragraph_composition=[])
    tracker = v3_modules.il.ParagraphTranslateTracker()
    tracker.set_pdf_unicode("Aushärtung")
    text = SimpleNamespace(unicode="{g0}", glossary_term_tokens={"{g0}": "固化"})
    with pytest.raises(guard.WrongTargetLanguageError):
        il.post_translate_paragraph(paragraph, tracker, text, "{g0}")
    assert paragraph.unicode == "Aushärtung"


def test_failed_fallback_is_tracked_and_prevents_publication(v3_modules, tmp_path):
    il = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    il.translation_config = SimpleNamespace(raise_if_cancelled=lambda: None)
    il._cached_glossaries = []
    il.use_as_fallback, il.support_llm_translate = False, True
    il.pre_translate_paragraph = lambda *a: (SOURCE, SimpleNamespace(unicode=SOURCE))
    il.generate_prompt_for_llm = lambda *a: SOURCE
    il.translate_engine = make_translator(v3_modules, [CHINESE, CHINESE])
    tracker = v3_modules.il.ParagraphTranslateTracker()
    tracker.set_pdf_unicode(SOURCE)
    tracker.set_input(SOURCE)
    paragraph = SimpleNamespace(unicode=SOURCE, debug_id="p1")
    il.translate_paragraph(paragraph, None, pbar=SimpleNamespace(advance=lambda: None), tracker=tracker)
    assert tracker.target_language_error
    document = v3_modules.il.DocumentTranslateTracker()
    document.new_page().paragraph.append(tracker)
    (tmp_path / "translate_tracking.json").write_text(document.to_json(), encoding="utf-8")
    with pytest.raises(guard.WrongTargetLanguageError):
        guard.validate_tracking_outputs(tmp_path, "it")


@pytest.mark.parametrize("group", ["page", "cross_page", "cross_column"])
def test_final_check_rejects_mixed_chinese_paragraphs(tmp_path, group):
    path = tmp_path / "translate_tracking.json"
    path.write_text(json.dumps({group: [{"paragraph": [
        {"pdf_unicode": SOURCE, "output": "Indurimento 固化"}]}]}), encoding="utf-8")
    with pytest.raises(guard.WrongTargetLanguageError):
        guard.validate_tracking_outputs(tmp_path, "it")
    assert guard.validate_tracking_outputs(tmp_path, "zh") == 1


@pytest.mark.parametrize("failure", [ConnectionError("Connection error."), RuntimeError("HTTP 401")])
def test_fallback_records_transport_and_service_errors(v3_modules, failure):
    il = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    il.translation_config = SimpleNamespace(raise_if_cancelled=lambda: None)
    il._cached_glossaries = []
    il.use_as_fallback, il.support_llm_translate = False, True
    il.pre_translate_paragraph = lambda *a: (SOURCE, SimpleNamespace(unicode=SOURCE))
    il.generate_prompt_for_llm = lambda *a: SOURCE

    def fail(*args, **kwargs):
        raise failure

    il.translate_engine = SimpleNamespace(llm_translate=fail)
    tracker = v3_modules.il.ParagraphTranslateTracker()
    tracker.set_pdf_unicode(SOURCE)
    tracker.set_input(SOURCE)
    paragraph = SimpleNamespace(unicode=SOURCE, debug_id="p1")
    il.translate_paragraph(paragraph, None, pbar=SimpleNamespace(advance=lambda: None), tracker=tracker)
    assert tracker.translation_error == f"{type(failure).__name__}: {failure}"
    assert paragraph.unicode == SOURCE
    assert tracker.last_llm_translate_tracker().has_error
    document = v3_modules.il.DocumentTranslateTracker()
    document.new_page().paragraph.append(tracker)
    with pytest.raises(guard.TranslationFailedError, match="1 of 1 attempted paragraphs failed"):
        guard.validate_tracker_payload(json.loads(document.to_json()), "it")


def test_empty_fallback_response_is_not_a_success(v3_modules):
    il = v3_modules.il.ILTranslator.__new__(v3_modules.il.ILTranslator)
    il.translation_config = SimpleNamespace(raise_if_cancelled=lambda: None)
    il._cached_glossaries = []
    il.use_as_fallback, il.support_llm_translate = False, True
    il.pre_translate_paragraph = lambda *a: (SOURCE, SimpleNamespace(unicode=SOURCE))
    il.generate_prompt_for_llm = lambda *a: SOURCE
    il.translate_engine = SimpleNamespace(llm_translate=lambda *a, **kw: "")
    tracker = v3_modules.il.ParagraphTranslateTracker()
    tracker.set_pdf_unicode(SOURCE)
    tracker.set_input(SOURCE)
    paragraph = SimpleNamespace(unicode=SOURCE, debug_id="p1")
    il.translate_paragraph(paragraph, None, pbar=SimpleNamespace(advance=lambda: None), tracker=tracker)
    assert "empty response" in tracker.translation_error
    assert not getattr(tracker, "output", None)
    assert paragraph.unicode == SOURCE


def test_captured_failure_shape_cannot_publish_untranslated_document():
    payload = {"page": [{"paragraph": [
        {"pdf_unicode": SOURCE, "input": SOURCE, "output": None,
         "llm_translate_trackers": [
             {"error_message": "Error Connection error. during translation. try fallback"},
             {"error_message": "", "output": ""},
         ]} for _ in range(72)
    ]}]}
    with pytest.raises(guard.TranslationFailedError, match="72 of 72 attempted paragraphs failed"):
        guard.validate_tracker_payload(payload, "it")


def test_successful_fallback_after_batch_error_is_accepted():
    payload = {"page": [{"paragraph": [
        {"pdf_unicode": SOURCE, "output": ITALIAN,
         "llm_translate_trackers": [{"error_message": "Connection error"}, {"output": ITALIAN}]},
        {"pdf_unicode": "200 mm", "output": None, "llm_translate_trackers": []},
    ]}]}
    assert guard.validate_tracker_payload(payload, "it") == 2


def test_partial_failure_is_not_hidden_by_successful_paragraphs():
    payload = {"page": [{"paragraph": [
        {"pdf_unicode": SOURCE, "output": ITALIAN, "llm_translate_trackers": [{"output": ITALIAN}]},
        {"pdf_unicode": SOURCE, "output": None, "translation_error": "HTTP 401",
         "llm_translate_trackers": [{"error_message": "HTTP 401"}]},
    ]}]}
    with pytest.raises(guard.TranslationFailedError, match="1 of 2 attempted paragraphs failed"):
        guard.validate_tracker_payload(payload, "it")
