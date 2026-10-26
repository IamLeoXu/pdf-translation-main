import os
from pathlib import Path

BACKEND_FONTS_DIR = Path(__file__).resolve().parents[3] / "fonts"
INSTALLED_FONTS_DIR = Path("/usr/local/share/fonts/pdf-translation")


def _bundled_font_path(name: str) -> Path:
    return next((directory / name for directory in (BACKEND_FONTS_DIR, INSTALLED_FONTS_DIR)
                 if (directory / name).is_file()), BACKEND_FONTS_DIR / name)


DEFAULT_FONT_PATH = Path(
    os.environ.get("RETAIN_PDF_FONT_PATH", "").strip()
    or str(_bundled_font_path("SourceHanSerifSC-Regular.otf"))
)
TITLE_BOLD_FONT_PATH = Path(
    os.environ.get("RETAIN_PDF_TITLE_BOLD_FONT_PATH", "").strip()
    or str(_bundled_font_path("SourceHanSerifSC-Bold.otf"))
)
DEFAULT_FONT_SIZE = 11.4
MIN_FONT_SIZE = 8.5
TYPST_DEFAULT_FONT_FAMILY = os.environ.get("RETAIN_PDF_TYPST_FONT_FAMILY", "").strip() or "auto"
