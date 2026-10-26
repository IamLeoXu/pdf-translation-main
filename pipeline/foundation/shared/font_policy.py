"""One font policy for target-language layout, rendering and cache identity."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
from inspect import signature
import os
from pathlib import Path
import shutil
import subprocess

from foundation.config import fonts
from foundation.shared.language import normalize_language_code, target_language_code


FONT_POLICY_VERSION = "target-fonts-v1"


@dataclass(frozen=True)
class FontProfile:
    families: tuple[str, ...]
    paths: tuple[str, ...]

    @cached_property
    def fingerprint(self) -> tuple:
        styles = []
        for path in self.paths:
            regular = Path(path)
            if regular.stem.endswith("-Regular"):
                for style in ("Bold", "Italic", "BoldItalic"):
                    sibling = regular.with_name(regular.name.replace("-Regular", "-" + style))
                    if sibling.is_file():
                        styles.append(_file_identity(str(sibling)))
        return (FONT_POLICY_VERSION, self.families, tuple(_file_identity(path) for path in self.paths), tuple(styles))


def _file_identity(path: str) -> tuple:
    try:
        stat = Path(path).stat()
        return (path, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (path, 0, 0)


def _font_dirs() -> list[Path]:
    return [fonts.BACKEND_FONTS_DIR, *(
        Path(part.strip()) for part in os.environ.get("RETAIN_PDF_TYPST_FONT_DIRS", "").split(os.pathsep)
        if part.strip()
    ), fonts.INSTALLED_FONTS_DIR, Path("/usr/share/fonts/truetype/noto")]


@lru_cache(maxsize=64)
def _system_font_path(family: str) -> str:
    executable = shutil.which("fc-match")
    if not executable:
        return ""
    try:
        result = subprocess.run([executable, "-f", "%{family}\n%{file}", family], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=5, check=True)
        actual, path = result.stdout.strip().split("\n", 1)
        if family.casefold() in {name.strip().casefold() for name in actual.split(",")} and Path(path).is_file():
            return path
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return ""


def _family_path(family: str, *, bold: bool = False) -> str:
    stems = {"Noto Serif": "NotoSerif", "Noto Sans": "NotoSans", "Source Han Serif SC": "SourceHanSerifSC",
             "Noto Naskh Arabic": "NotoNaskhArabic"}
    stem = stems.get(family)
    if stem:
        for directory in _font_dirs():
            for extension in ("ttf", "otf"):
                path = directory / f"{stem}-{'Bold' if bold else 'Regular'}.{extension}"
                if path.is_file():
                    return str(path)
    return _system_font_path(family)


def resolve_font_profile(target_lang: str = "", font_family: str | None = None) -> FontProfile:
    code = normalize_language_code(target_lang or target_language_code())
    main = code.split("-")[0].lower()
    requested = (font_family or os.environ.get("RETAIN_PDF_TYPST_FONT_FAMILY") or fonts.TYPST_DEFAULT_FONT_FAMILY).strip()
    if requested and requested.casefold() != "auto":
        primary = requested
    elif main == "zh":
        primary = "Noto Serif CJK TC" if code.lower().endswith(("-tw", "-hk", "-hant")) else "Source Han Serif SC"
    elif main == "ja":
        primary = "Noto Serif CJK JP"
    elif main == "ko":
        primary = "Noto Serif CJK KR"
    elif main in {"ar", "fa", "ur"}:
        primary = "Noto Naskh Arabic"
    else:
        primary = os.environ.get("RETAIN_PDF_LATIN_FONT_FAMILY", "").strip() or "Noto Serif"
    families = tuple(dict.fromkeys((primary, "Noto Serif", "Noto Sans", "Source Han Serif SC")))
    custom_path = os.environ.get("RETAIN_PDF_FONT_PATH", "").strip() if requested.casefold() != "auto" else ""
    paths = tuple(path for index, family in enumerate(families)
                  if (path := (custom_path if index == 0 and custom_path else _family_path(family))))
    return FontProfile(families=families, paths=paths)


_ACTIVE_PROFILE: ContextVar[FontProfile | None] = ContextVar("render_font_profile", default=None)
_ACTIVE_FAMILY: ContextVar[str | None] = ContextVar("render_font_override", default=None)


def current_font_profile() -> FontProfile:
    return _ACTIVE_PROFILE.get() or resolve_font_profile()


@contextmanager
def font_context(items: list[dict] | None = None, *, font_family: str | None = None, profile: FontProfile | None = None):
    items = items or []
    language = next((target_language_code(item) for item in items if item.get("target_lang")), target_language_code())
    family = font_family or _ACTIVE_FAMILY.get() or next((str(item["_render_font_family"]) for item in items if item.get("_render_font_family")), None)
    profile_token = _ACTIVE_PROFILE.set(profile or resolve_font_profile(language, family))
    family_token = _ACTIVE_FAMILY.set(family)
    try:
        yield current_font_profile()
    finally:
        _ACTIVE_FAMILY.reset(family_token)
        _ACTIVE_PROFILE.reset(profile_token)


def with_target_font(function):
    """Bind layout entry points without changing their public call signatures."""
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs).arguments
        items = bound.get("translated_items")
        if items is None and "translated_pages" in bound:
            items = [item for page in bound["translated_pages"].values() for item in page]
        if items is None and "item" in bound:
            items = [bound["item"]]
        with font_context(items, font_family=bound.get("font_family")):
            return function(*args, **kwargs)
    return wrapped


def item_font_fingerprint(item: dict) -> tuple:
    return resolve_font_profile(target_language_code(item), item.get("_render_font_family")).fingerprint
