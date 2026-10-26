import json
from pathlib import Path

import pytest

from store import TaskStore


def test_public_view_uses_latest_retain_pipeline_event_for_stage_text(tmp_path: Path) -> None:
    test_root = tmp_path
    work_dir = test_root / "task"
    logs_dir = work_dir / "retain_work" / "logs"
    logs_dir.mkdir(parents=True)
    events_path = logs_dir / "pipeline_events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "stage": "translating",
                        "user_stage": "translation",
                        "substage": "translation_batches",
                        "message": "开始批量翻译",
                        "progress_current": 0,
                        "progress_total": 4,
                        "progress_unit": "batch",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "stage": "rendering",
                        "user_stage": "render",
                        "substage": "render_pages",
                        "message": "正在渲染页面",
                        "progress_current": 2,
                        "progress_total": 4,
                        "progress_unit": "page",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    store = TaskStore()
    store.add(
        {
            "task_id": "task-1",
            "status": "running",
            "work_dir": str(work_dir),
            "total_pages": 4,
            "translated_pages": 1,
            "inputs": {},
        }
    )

    view = store.public_view("task-1")

    assert view["stage_text"] == "正在渲染页面: 2 / 4 页"
    assert view["pipeline_stage"] == "rendering"
    assert view["pipeline_user_stage"] == "render"
    assert view["pipeline_progress_current"] == 2
    assert view["pipeline_progress_total"] == 4
    assert view["pipeline_progress_percent"] == 50.0



def _ocr_task(tmp_path: Path, **fields):
    store = TaskStore()
    store.add(
        {
            "task_id": "ocr-task",
            "status": "running",
            "work_dir": str(tmp_path),
            "total_pages": 3,
            "translated_pages": 0,
            "progress": 5.0,
            "retain_progress": 5.0,
            "inputs": {},
            **fields,
        }
    )
    return store


def _write_ocr_event(tmp_path: Path, **fields):
    events_path = tmp_path / "retain_work" / "logs" / "pipeline_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "stage": "ocr_processing",
        "user_stage": "ocr",
        "substage": "local_provider",
        "message": "OCR",
        "progress_current": 1,
        "progress_total": 3,
        "progress_unit": "page",
        **fields,
    }
    with events_path.open("a", encoding="utf-8") as events:
        events.write(json.dumps(event) + "\n")


@pytest.mark.parametrize("status", ["pending", "running"])
def test_local_ocr_pages_advance_overall_without_counting_translated_pages(tmp_path, status):
    store = _ocr_task(tmp_path, status=status)
    for current, expected in [(1, 11.7), (2, 18.3), (3, 25.0)]:
        _write_ocr_event(tmp_path, progress_current=current)
        view = store.public_view("ocr-task")
        assert view["progress"] == expected
        assert view["retain_progress"] == expected
        assert view["translated_pages"] == 0
        assert view["page_progress"] == 0.0
        assert view["pipeline_progress_current"] == current
        assert "_local_ocr_progress_highwater" not in view


def test_local_ocr_highwater_survives_heartbeat_and_following_stage(tmp_path):
    store = _ocr_task(tmp_path)
    _write_ocr_event(tmp_path, progress_current=2)
    assert store.public_view("ocr-task")["progress"] == 18.3
    _write_ocr_event(tmp_path, progress_current=None, progress_total=None)
    assert store.public_view("ocr-task")["retain_progress"] == 18.3
    _write_ocr_event(tmp_path, progress_current=3)
    assert store.public_view("ocr-task")["progress"] == 25.0
    store.update("ocr-task", retain_progress=5.0)
    _write_ocr_event(tmp_path, stage="normalizing", substage="normalize", progress_current=None)
    assert store.public_view("ocr-task")["progress"] == 25.0
    assert store.public_view("ocr-task")["retain_progress"] == 25.0
    store.update("ocr-task", retain_progress=35.0)
    assert store.public_view("ocr-task")["progress"] == 35.0
    store.update("ocr-task", retain_progress=25.0)
    assert store.public_view("ocr-task")["retain_progress"] == 35.0


def test_local_ocr_never_reduces_existing_retain_progress(tmp_path):
    store = _ocr_task(tmp_path, retain_progress=40.0)
    _write_ocr_event(tmp_path)
    view = store.public_view("ocr-task")
    assert view["retain_progress"] == 40.0
    assert view["progress"] == 40.0


@pytest.mark.parametrize(
    "fields",
    [
        {"substage": "local_provider_image_reocr", "progress_unit": "region"},
        {"substage": "local_provider_image_reocr"},
        {"progress_unit": "region"},
        {"stage": "rendering"},
        {"substage": "cloud_provider"},
        {"progress_current": None},
        {"progress_total": None},
        {"progress_total": 0},
        {"progress_current": -1},
        {"progress_current": 4},
    ],
)
def test_non_whole_page_local_ocr_and_invalid_counts_do_not_map_progress(tmp_path, fields):
    store = _ocr_task(tmp_path)
    _write_ocr_event(tmp_path, **fields)
    view = store.public_view("ocr-task")
    assert view["progress"] == 5.0
    assert view["retain_progress"] == 5.0
    assert view["translated_pages"] == 0


@pytest.mark.parametrize("status", ["failed", "succeeded"])
def test_finished_task_does_not_start_mapping_ocr_progress(tmp_path, status):
    store = _ocr_task(tmp_path, status=status)
    _write_ocr_event(tmp_path, progress_current=3)
    view = store.public_view("ocr-task")
    assert view["retain_progress"] == 5.0
    assert view["progress"] == (100.0 if status == "succeeded" else 5.0)
    assert view["translated_pages"] == (3 if status == "succeeded" else 0)


def test_failed_task_preserves_already_reported_ocr_highwater(tmp_path):
    store = _ocr_task(tmp_path)
    _write_ocr_event(tmp_path, progress_current=2)
    assert store.public_view("ocr-task")["progress"] == 18.3
    store.mark_failed("ocr-task", "OCR failed")
    _write_ocr_event(tmp_path, progress_current=3)
    view = store.public_view("ocr-task")
    assert view["progress"] == 18.3
    assert view["retain_progress"] == 18.3
    assert view["translated_pages"] == 0
    assert view["stage_text"] == "翻译失败: 0 / 3 页"


def test_existing_non_ocr_progress_and_translated_files_are_unchanged(tmp_path):
    store = _ocr_task(tmp_path, retain_progress=30.0, v3_progress=50.0)
    translated = tmp_path / "retain_work" / "translated"
    translated.mkdir(parents=True)
    (translated / "page-1-result.json").write_text("{}", encoding="utf-8")
    (translated / "page-1-other.json").write_text("{}", encoding="utf-8")
    view = store.public_view("ocr-task")
    assert view["progress"] == 50.0
    assert view["retain_progress"] == 30.0
    assert view["translated_pages"] == 1
    assert view["page_progress"] == 33.3
