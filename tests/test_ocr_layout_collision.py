from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from services.rendering.layout.payload.collision import mark_adjacent_collision_risk
from services.rendering.layout.payload.emit import payload_to_render_block
from services.document_schema.provider_adapters.paddle.content_extract import tighten_text_bbox
from services.rendering.layout.model.models import RenderLayoutBlock, RenderPageSpec
from services.rendering.output.typst.emitter import build_typst_source_from_page_specs


def payload(item_id="short-list", bbox=None, **extra):
    bbox = bbox or [120.0, 245.0, 499.0, 261.0]
    text = "— dovrebbe essere garantito un efficace trasferimento delle conoscenze di processo e di prodotto."
    return {
        "item": {"item_id": item_id, "bbox": bbox, "block_type": "text", "block_kind": "text",
                 "semantic_role": "body", "target_lang": "it", "source_text": "— process and product knowledge transfer."},
        "index": 0, "bbox": bbox, "inner_bbox": bbox, "cover_bbox": bbox,
        "translated_text": text, "formula_map": [], "font_size_pt": 11.9,
        "leading_em": 0.54, "is_body": True, "page_body_font_size_pt": 11.9,
        "dense_small_box": False, "heavy_dense_small_box": False,
        "render_kind": "markdown", "prefer_typst_fit": False, "adjacent_collision_risk": False,
        "adjacent_available_height_pt": None, "_body_font_unified": True,
        "_allow_short_text_bbox_overflow": True, **extra,
    }


def test_single_line_ocr_box_with_wrapped_translation_cannot_skip_measured_fit():
    current = payload()
    following = payload("paragraph-1.7", [105.0, 269.0, 517.0, 318.0])
    mark_adjacent_collision_risk([current, following])
    block = payload_to_render_block(current)
    assert block.fit_to_box
    assert 0 < block.fit_max_height_pt <= 269.0 - 245.0
    assert block.inner_bbox[3] > 261.0
    assert block.inner_bbox[3] < 269.0
    # Consistent body size is retained as the *preferred* size, not as a reason
    # to render an overflowing paragraph without measuring it.
    assert current["font_size_pt"] == 11.9
    assert block.font_size_pt == 11.9


def test_column_neighbor_is_found_even_with_interleaved_other_column():
    left = payload(bbox=[20.0, 20.0, 220.0, 60.0])
    right = payload("right", [300.0, 30.0, 500.0, 80.0])
    next_left = payload("left-next", [20.0, 70.0, 220.0, 110.0])
    mark_adjacent_collision_risk([left, right, next_left])
    assert left["adjacent_collision_risk"]
    assert 0 < left["adjacent_available_height_pt"] < 50
    assert not right["adjacent_collision_risk"]


def test_side_by_side_blocks_and_equal_top_labels_are_not_vertical_neighbors():
    blocks = [payload(bbox=[0, 0, 100, 20]), payload("same-row", [0, 0, 100, 20]),
              payload("right", [120, 10, 220, 30])]
    original = deepcopy(blocks)
    mark_adjacent_collision_risk(blocks)
    assert blocks == original


def test_low_height_estimate_does_not_bypass_real_markdown_list_measurement():
    current = payload(translated_text="- short\n\n- another short item")
    following = payload("next", [105.0, 269.0, 517.0, 318.0])
    mark_adjacent_collision_risk([current, following])
    assert current["prefer_typst_fit"]
    assert payload_to_render_block(current).fit_to_box


def test_pseudo_line_estimate_cannot_remove_bottom_of_source_ocr_box():
    bbox = [105.468, 619.441, 516.844, 652.438]
    text = "1.12 Some of the responsibilities outlined in this document for the SU may also be considered to be part of the management unit responsibilities."
    result = tighten_text_bbox(bbox=bbox, text=text, block_type="text", sub_type="body")
    assert result == bbox
    assert result is not bbox


def test_source_covers_use_full_boxes_and_are_painted_before_all_translations(tmp_path):
    blocks = [RenderLayoutBlock(
        block_id=f"row-{index}", page_index=0, background_rect=[10, 20+index*40, 120, 55+index*40],
        content_rect=[12, 22+index*40, 118, 48+index*40], content_kind="markdown", content_text="Traduzione",
        plain_text="Traduzione", math_map=[], font_size_pt=10, leading_em=0.5, target_lang="it",
    ) for index in range(2)]
    spec = RenderPageSpec(page_index=0, page_width_pt=200, page_height_pt=200, background_pdf_path=None, blocks=blocks)
    source = build_typst_source_from_page_specs(background_pdf_path=tmp_path/'source.pdf', page_specs=[spec], work_dir=tmp_path)
    assert source.index("rp0_row_1_1_cover") < source.index("rp0_row_0_0_md")
    assert "rect(width: 110pt, height: 35pt" in source


