from __future__ import annotations

from foundation.shared.font_policy import with_target_font

from services.rendering.layout.fit_decision import plan_chinese_body_fit
from services.rendering.layout.typography.geometry import inner_bbox
from services.rendering.layout.chinese_body_fit import estimate_chinese_body_height_pt


@with_target_font
def should_fit_wrapped_markdown(item: dict, markdown_text: str, *, font_size_pt: float, leading_em: float) -> bool:
    content_rect = inner_bbox(item)
    if len(content_rect) != 4:
        return False
    width = max(1.0, content_rect[2] - content_rect[0])
    height = max(1.0, content_rect[3] - content_rect[1])
    if not markdown_text.strip() or font_size_pt <= 0:
        return False
    estimate = estimate_chinese_body_height_pt(width, markdown_text, [], font_size_pt, leading_em)
    return estimate.estimated_height_pt > height * 0.92


@with_target_font
def fit_body_font_size_pt(
    item: dict,
    *,
    markdown_text: str,
    formula_map: list[dict],
    font_size_pt: float,
    leading_em: float,
    page_font_size: float,
) -> float:
    content_rect = inner_bbox(item)
    fit_decision = plan_chinese_body_fit(
        bbox_width_pt=max(1.0, content_rect[2] - content_rect[0]),
        bbox_height_pt=max(1.0, content_rect[3] - content_rect[1]),
        text=markdown_text,
        formula_map=formula_map,
        font_size_pt=font_size_pt,
        leading_em=leading_em,
        max_growth_font_size_pt=page_font_size + 0.25 if page_font_size > 0 else None,
    )
    return fit_decision.font_size_pt


__all__ = [
    "fit_body_font_size_pt",
    "should_fit_wrapped_markdown",
]
