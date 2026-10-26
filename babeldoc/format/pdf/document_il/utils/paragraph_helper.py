import logging
import re

from babeldoc.format.pdf.document_il import il_version_1

logger = logging.getLogger(__name__)


def is_cid_paragraph(paragraph: il_version_1.PdfParagraph):
    chars: list[il_version_1.PdfCharacter] = []
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_line:
            chars.extend(composition.pdf_line.pdf_character)
        elif composition.pdf_same_style_characters:
            chars.extend(composition.pdf_same_style_characters.pdf_character)
        elif composition.pdf_same_style_unicode_characters:
            continue
        #     chars.extend(composition.pdf_same_style_unicode_characters.unicode)
        elif composition.pdf_formula:
            chars.extend(composition.pdf_formula.pdf_character)
        elif composition.pdf_character:
            chars.append(composition.pdf_character)
        else:
            logger.error(
                f"Unknown composition type. "
                f"Composition: {composition}. "
                f"Paragraph: {paragraph}. ",
            )
            continue

    cid_count = 0
    for char in chars:
        if re.match(r"^\(cid:\d+\)$", char.char_unicode):
            cid_count += 1

    return cid_count > len(chars) * 0.8


def minimum_translation_length(paragraph, translation_config) -> int:
    """Keep table and explicit field labels eligible across source languages."""
    table_labels = {"table_text", "table_cell", "wired_table_cell", "wireless_table_cell", "table_cell_hybrid"}
    if (getattr(translation_config, "enable_table_translation", False)
            and getattr(paragraph, "layout_label", None) in table_labels):
        return 1
    text = str(getattr(paragraph, "unicode", "") or "").strip()
    # A short label ending in a colon is readable text, even when a wrapped
    # heading leaves only its final word on a separate line. Bare numbers and
    # variable-like short body fragments keep the existing minimum.
    if text.endswith((":", "：")) and any(character.isalpha() for character in text):
        return 1
    return translation_config.min_text_length


NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def is_pure_numeric_paragraph(paragraph) -> bool:
    """只检查段落是否为纯数字（支持整数、小数、负数）"""

    if not paragraph or not getattr(paragraph, "unicode", None):
        return False

    text = paragraph.unicode.strip()
    if not text:
        return False

    return bool(NUMERIC_PATTERN.match(text))


def is_placeholder_only_paragraph(paragraph: il_version_1.PdfParagraph) -> bool:
    """Check if a paragraph contains only placeholders and whitespace.

    Args:
        paragraph: PDF paragraph to check

    Returns:
        True if the paragraph contains only placeholders (formula or style tags)
        and whitespace, False otherwise
    """
    if not paragraph or not paragraph.unicode:
        return False

    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_formula:
            # Formula composition is allowed
            continue
        elif composition.pdf_character:
            # Check if single character is whitespace
            if not composition.pdf_character.char_unicode.isspace():
                return False
        elif composition.pdf_line:
            # Check if all characters in the line are whitespace
            for char in composition.pdf_line.pdf_character:
                if not char.char_unicode.isspace():
                    return False
        elif composition.pdf_same_style_characters:
            # Check if all characters in the group are whitespace
            for char in composition.pdf_same_style_characters.pdf_character:
                if not char.char_unicode.isspace():
                    return False
        elif composition.pdf_same_style_unicode_characters:
            # Check if the unicode content is only whitespace
            if not composition.pdf_same_style_unicode_characters.unicode.isspace():
                return False
        else:
            # Unknown composition type, conservatively return False
            return False

    return True
