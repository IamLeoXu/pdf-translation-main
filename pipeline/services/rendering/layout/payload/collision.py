from __future__ import annotations

from services.rendering.layout.payload.collision_context import adjacent_collision_context


VERTICAL_COLLISION_TRIGGER_RATIO = 0.9


def mark_adjacent_collision_risk(ordered_payloads: list[dict]) -> None:
    for index, current in enumerate(ordered_payloads):
        # Reading-order neighbors can be in different columns. Find the nearest
        # block below this one that actually shares its horizontal space.
        contexts = [
            context for nxt in ordered_payloads[index + 1:]
            if (context := adjacent_collision_context(current, nxt)) is not None
        ]
        if not contexts:
            continue
        context = min(contexts, key=lambda candidate: candidate.max_height_pt)
        if current.get("_allow_short_text_bbox_overflow"):
            left, top, right, bottom = current["inner_bbox"]
            if context.max_height_pt > bottom - top:
                # A translated one-line source may need two lines. It may use
                # the measured empty gap, but never the next paragraph's space.
                current["inner_bbox"] = [left, top, right, top + context.max_height_pt]
                current["_relaxed_fit_height_pt"] = context.max_height_pt
        current["adjacent_collision_risk"] = True
        # Let Typst measure the actual Markdown, including list indentation,
        # paragraph spacing, hyphenation and the chosen font. Keep the preferred
        # size when it fits; the heuristic must not bypass this final check.
        current["prefer_typst_fit"] = True
        _remember_adjacent_height_limit(current, context.max_height_pt)


def _remember_adjacent_height_limit(payload: dict, max_height_pt: float) -> None:
    previous_limit = payload.get("adjacent_available_height_pt")
    if previous_limit is None or max_height_pt < previous_limit:
        payload["adjacent_available_height_pt"] = max(6.0, max_height_pt)
