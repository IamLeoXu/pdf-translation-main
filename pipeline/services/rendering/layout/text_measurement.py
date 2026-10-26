"""Target-font width estimates, with glyph fallback and bounded font-aware caches."""
from __future__ import annotations

from functools import lru_cache
import unicodedata

from foundation.shared.font_policy import current_font_profile, resolve_font_profile


@lru_cache(maxsize=64)
def _measurement_font(identity: tuple):
    try:
        import fitz
        return fitz.Font(fontfile=identity[0])
    except (ImportError, RuntimeError, ValueError, OSError):
        return None


def _fallback_width(char: str) -> float:
    if unicodedata.combining(char) or char in "\r\n":
        return 0.0
    if char.isspace():
        return 0.25
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 1.0
    if char in "ilI.,:;'!|":
        return 0.28
    if char in "MWmw@%":
        return 0.85
    return 0.55


@lru_cache(maxsize=16384)
def _text_width_cached(text: str, fingerprint: tuple) -> float:
    available = [font for identity in fingerprint[2] if (font := _measurement_font(identity)) is not None]
    # Use ordered, explicit glyph fallback, matching the Typst family list.
    runs: list[tuple[object, str]] = []
    for char in text:
        selected = next((font for font in available if font.has_glyph(ord(char), fallback=False)), None)
        if runs and runs[-1][0] is selected:
            runs[-1] = (selected, runs[-1][1] + char)
        else:
            runs.append((selected, char))
    return max(0.0, sum(float(font.text_length(run, fontsize=1)) if font is not None
                        else sum(_fallback_width(char) for char in run) for font, run in runs))


def text_width_em(text: str, *, target_lang: str = "", font_family: str | None = None) -> float:
    profile = resolve_font_profile(target_lang, font_family) if target_lang or font_family else current_font_profile()
    return _text_width_cached(text.replace("\r", "").replace("\n", ""), profile.fingerprint)
