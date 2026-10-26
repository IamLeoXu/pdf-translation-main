"""Target-language checks shared by BabelDOC prompts, terms, caches and output."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
import unicodedata

from babeldoc.babeldoc_exception.BabelDOCException import TranslationFailedError

logger = logging.getLogger(__name__)
LANGUAGE_POLICY_VERSION = "target-language-v1"
_HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
_NAMES = {
    "en": "English (US)", "it": "Italian", "de": "German", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "ar": "Arabic",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "vi": "Vietnamese", "th": "Thai",
}
_FILTER_NOTICES = {
    "en": "The translation service declined this content. The original text was retained.",
    "it": "Il servizio di traduzione ha rifiutato questo contenuto. Il testo originale è stato mantenuto.",
    "de": "Der Übersetzungsdienst hat diesen Inhalt abgelehnt. Der Originaltext wurde beibehalten.",
    "zh": "翻译服务拒绝处理此内容，已保留原文。",
    "zh-tw": "翻譯服務拒絕處理此內容，已保留原文。",
    "ja": "翻訳サービスがこの内容の処理を拒否したため、原文を保持しました。",
    "tr": "Çeviri hizmeti bu içeriği reddetti. Özgün metin korundu.",
    "ar": "رفضت خدمة الترجمة هذا المحتوى. تم الاحتفاظ بالنص الأصلي.",
}


def language_main(language: str) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    aliases = {name.lower(): code for code, name in _NAMES.items()}
    aliases.update({"english": "en", "italiano": "it", "deutsch": "de",
                    "simplified chinese": "zh", "traditional chinese": "zh",
                    "简体中文": "zh", "繁體中文": "zh"})
    return aliases.get(value, value.split("-", 1)[0])


def language_name(language: str) -> str:
    code = str(language or "").replace("_", "-")
    if code.lower() in {"zh-tw", "zh-hk", "zh-hant"}:
        return "Traditional Chinese"
    if language_main(code) == "zh":
        return "Simplified Chinese"
    return _NAMES.get(language_main(code), code)


def content_filter_notice(language: str) -> str:
    code = str(language).lower().replace("_", "-")
    if code in {"zh-tw", "zh-hk", "zh-hant"}:
        return _FILTER_NOTICES["zh-tw"]
    return _FILTER_NOTICES.get(language_main(code), _FILTER_NOTICES["en"])


def unexpected_han(language: str, source: str, output: str, allowed_literals=()) -> str:
    if language_main(language) in {"zh", "ja", "ko"}:
        return ""
    allowed = str(source or "") + "".join(str(value) for value in allowed_literals)
    source_han = Counter(_HAN.findall(unicodedata.normalize("NFKC", allowed)))
    output_han = Counter(_HAN.findall(unicodedata.normalize("NFKC", str(output or ""))))
    return "".join((output_han - source_han).elements())


class WrongTargetLanguageError(ValueError):
    pass


def ensure_target_language(language: str, source: str, output: str, allowed_literals=()) -> None:
    if unexpected_han(language, source, output, allowed_literals):
        raise WrongTargetLanguageError(f"Unexpected Han text in translation requested in {language_name(language)}")


def _response_text(output: str) -> str:
    """Validate translations in JSON without counting echoed source fields."""
    raw = str(output or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return str(output or "")
    rows = parsed if isinstance(parsed, list) else [parsed]
    texts = []
    for row in rows:
        if not isinstance(row, dict):
            return str(output or "")
        for key in ("output", "tgt", "translated_text"):
            if key in row:
                texts.append(str(row[key] or ""))
                break
        else:
            return str(output or "")
    return "\n".join(texts)


def guarded_request(text, language, request, *, llm=False, cache=None, on_cache_hit=None,
                    source_text=None, allowed_literals=()):
    """Reject poisoned cache entries and retry a wrong-language response once."""
    source = text if source_text is None else source_text
    if cache is not None:
        try:
            cached = cache.get(text)
        except Exception:
            cached = None
        if cached is not None:
            if not unexpected_han(language, source, _response_text(cached), allowed_literals):
                if on_cache_hit is not None:
                    on_cache_hit()
                return cached
            logger.warning("Ignoring cached translation in an unexpected target language")
    for attempt in range(2):
        prompt = text
        if attempt and llm:
            prompt += (f"\n\nCorrection: translate into {language_name(language)} only. "
                       "Do not introduce Chinese characters absent from the source. "
                       "Preserve the requested output format and placeholders.")
        output = request(prompt)
        try:
            ensure_target_language(language, source, _response_text(output), allowed_literals)
        except WrongTargetLanguageError:
            if not attempt:
                logger.warning("Retrying response in an unexpected target language")
                continue
            raise
        if cache is not None:
            try:
                cache.set(text, output)
            except Exception:
                logger.debug("Could not store validated translation in cache", exc_info=True)
        return output


def filter_glossaries(glossaries, language):
    result = []
    seen = set()
    for glossary in glossaries or ():
        if id(glossary) in seen:
            continue
        seen.add(id(glossary))
        entries = []
        for entry in glossary.entries:
            explicit = getattr(entry, "target_language", None)
            if explicit and language_main(explicit) != language_main(language):
                continue
            if explicit or not unexpected_han(language, entry.source, entry.target):
                entries.append(entry)
        if entries:
            result.append(glossary if len(entries) == len(glossary.entries)
                          else type(glossary)(glossary.name, entries))
    return result


def allowed_glossary_literals(glossaries, source):
    result = []
    for glossary in glossaries or ():
        spans = getattr(glossary, "get_active_spans_for_text", None)
        if callable(spans):
            result.extend(target for _, target, _, _ in spans(source))
        else:
            result.extend(target for _, target in glossary.get_active_entries_for_text(source))
    return result


def validate_tracker_payload(payload, language, glossaries=()):
    checked = 0
    attempted = 0
    failures = []
    active_glossaries = filter_glossaries(glossaries, language)
    for group in ("page", "cross_page", "cross_column"):
        for page in payload.get(group, ()):
            for paragraph in page.get("paragraph", ()):
                if paragraph.get("target_language_error"):
                    raise WrongTargetLanguageError(paragraph["target_language_error"])
                source = str(paragraph.get("pdf_unicode") or paragraph.get("input") or "")
                output = str(paragraph.get("output") or "")
                attempts = paragraph.get("llm_translate_trackers") or ()
                if attempts:
                    attempted += 1
                error = paragraph.get("translation_error")
                if error or (attempts and not output.strip()):
                    reason = error or next(
                        (item.get("error_message") for item in reversed(attempts)
                         if item.get("error_message")),
                        "No accepted translation returned for an attempted paragraph",
                    )
                    failures.append(str(reason))
                    continue
                ensure_target_language(language, source, output,
                                       allowed_glossary_literals(active_glossaries, source))
                checked += 1
    if failures:
        raise TranslationFailedError(
            f"{len(failures)} of {max(attempted, len(failures))} attempted paragraphs failed. "
            f"First failure: {failures[0]}"
        )
    return checked


def validate_tracking_outputs(working_dir, language, glossaries=()):
    checked = 0
    for path in Path(working_dir).rglob("translate_tracking.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checked += validate_tracker_payload(payload, language, glossaries)
    return checked
