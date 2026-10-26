"""Language metadata shared by translation, validation and rendering."""
from __future__ import annotations

import os
import re
from collections import Counter


LANGUAGE_NAMES = {
    "zh": "简体中文", "zh-CN": "简体中文", "zh-TW": "繁體中文",
    "en": "English (US)", "it": "Italiano", "de": "Deutsch",
    "ja": "日本語", "ko": "한국어", "tr": "Türkçe", "ar": "العربية",
    "fr": "Français", "es": "Español", "pt": "Português", "ru": "Русский",
    "th": "ภาษาไทย", "vi": "Tiếng Việt",
}


def normalize_language_code(value: str) -> str:
    parts = str(value or "").strip().replace("_", "-").split("-")
    return "-".join([parts[0].lower(), *(
        part.upper() if len(part) == 2 else part.title() for part in parts[1:]
    )])


def target_language_code(item: dict | None = None) -> str:
    return normalize_language_code(
        (item or {}).get("target_lang") or os.environ.get("RETAIN_TARGET_LANG") or "zh-CN"
    )


def language_name(code: str, explicit_name: str = "") -> str:
    if str(explicit_name or "").strip():
        return explicit_name.strip()
    normalized = normalize_language_code(code)
    return LANGUAGE_NAMES.get(normalized, LANGUAGE_NAMES.get(normalized.split("-")[0], normalized))


def target_language_name(item: dict | None = None) -> str:
    source = item or {}
    # An explicit item language wins over the worker's environment defaults.
    explicit = source.get("target_language_name") or (
        os.environ.get("RETAIN_TARGET_LANGUAGE_NAME", "") if not source.get("target_lang") else ""
    )
    return language_name(target_language_code(source), str(explicit or ""))


def is_han_language(item: dict | None = None) -> bool:
    return target_language_code(item).split("-")[0] in {"zh", "ja", "ko"}


def unexpected_han_characters(item: dict, source_text: str, translated_text: str) -> str:
    if is_han_language(item):
        return ""
    allowed = source_text + "".join(str(value) for value in item.get("translation_allowed_literals", []))
    output_han = Counter(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", translated_text))
    allowed_han = Counter(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", allowed))
    return "".join((output_han - allowed_han).elements())


def bind_language_context(item: dict, context) -> dict:
    if context is None:
        return {**item, "target_lang": target_language_code(item), "target_language_name": target_language_name(item)}
    return {
        **item,
        "source_lang": context.source_lang if context.source_lang != "auto" else item.get("source_lang", "auto"),
        "target_lang": normalize_language_code(context.target_lang),
        "target_language_name": language_name(context.target_lang, context.target_language_name),
        "translation_allowed_literals": [
            entry.target for entry in context.glossary_entries
            if any("\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in entry.target)
        ],
    }
