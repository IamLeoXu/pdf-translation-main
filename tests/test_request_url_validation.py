"""Requests with Swagger placeholders must fail before a task is queued."""
from __future__ import annotations

import importlib.util
from itertools import count
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(monkeypatch, tmp_path):
    records, queued = {}, []
    ids = count(1)

    def new_work_dir(task_id):
        path = tmp_path / task_id
        path.mkdir()
        return path

    store = ModuleType("store")
    store._now_iso = lambda: "2026-10-23T00:00:00"
    store._task_queue = SimpleNamespace(put=queued.append)
    store.new_task_id = lambda: f"task-{next(ids)}"
    store.new_work_dir = new_work_dir
    store.start_background_workers = lambda: None
    store.tasks_store = SimpleNamespace(
        add=lambda task: records.update({task["task_id"]: task}),
        public_view=lambda task_id: {"task_id": task_id, "status": records[task_id]["status"]},
    )
    monkeypatch.setitem(sys.modules, "store", store)
    image = ModuleType("image_translate")
    image.image_dimensions = lambda data: (100, 100)
    monkeypatch.setitem(sys.modules, "image_translate", image)
    spec = importlib.util.spec_from_file_location("app_url_validation_under_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_pdf_page_count", lambda path: 3)
    monkeypatch.setattr(module.settings, "llm_base_url", "https://api.example.com/v1")
    return SimpleNamespace(client=TestClient(module.app), records=records, queued=queued,
                           root=tmp_path, settings=module.settings)


@pytest.mark.parametrize("endpoint,field,value", [
    ("/tasks", "openai_base_url", "string"),
    ("/tasks", "openai_base_url", "api.example.com/v1"),
    ("/tasks", "openai_base_url", "ftp://api.example.com/v1"),
    ("/tasks", "callback_url", "string"),
    ("/tasks", "paddle_api_url", "string"),
    ("/images/translate", "openai_base_url", "string"),
    ("/images/translate", "paddle_api_url", "string"),
])
def test_invalid_urls_are_rejected_before_task_creation(api, endpoint, field, value):
    filename = "input.pdf" if endpoint == "/tasks" else "input.png"
    response = api.client.post(endpoint, data={
        "lang_in": "de", "lang_out": "it", "openai_api_key": "offline-unused", field: value,
    }, files={"file": (filename, b"offline test file")})
    assert response.status_code == 422
    assert any(item["loc"][-1] == field for item in response.json()["detail"])
    assert api.queued == []
    assert api.records == {}
    assert list(api.root.iterdir()) == []


@pytest.mark.parametrize("endpoint", ["/tasks", "/images/translate"])
@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://host.docker.internal:8080/v1"])
def test_valid_urls_and_language_parameters_reach_worker_unchanged(api, endpoint, url):
    filename = "input.pdf" if endpoint == "/tasks" else "input.png"
    response = api.client.post(endpoint, data={
        "lang_in": "de", "lang_out": "it", "openai_api_key": "offline-unused", "openai_base_url": url,
    }, files={"file": (filename, b"offline test file")})
    assert response.status_code == 202
    inputs = api.records[response.json()["task_id"]]["inputs"]
    assert inputs["openai_base_url"] == url
    assert isinstance(inputs["openai_base_url"], str)
    assert (inputs["lang_in"], inputs["lang_out"]) == ("de", "it")


@pytest.mark.parametrize("value", [None, ""])
def test_empty_optional_model_url_preserves_server_default(api, value):
    data = {"lang_in": "de", "lang_out": "it", "openai_api_key": "offline-unused"}
    if value is not None:
        data["openai_base_url"] = value
    response = api.client.post("/tasks", data=data, files={"file": ("input.pdf", b"offline")})
    assert response.status_code == 202
    inputs = api.records[response.json()["task_id"]]["inputs"]
    assert inputs["openai_base_url"] is None


def test_invalid_server_default_is_rejected_before_task_creation(api, monkeypatch):
    monkeypatch.setattr(api.settings, "llm_base_url", "string")
    response = api.client.post("/tasks", data={"openai_api_key": "offline-unused"},
                               files={"file": ("input.pdf", b"offline")})
    assert response.status_code == 500
    assert response.json()["detail"]["field"] == "LLM_BASE_URL"
    assert api.queued == []


def test_request_url_overrides_invalid_server_default(api, monkeypatch):
    monkeypatch.setattr(api.settings, "llm_base_url", "string")
    response = api.client.post("/tasks", data={"openai_api_key": "offline-unused",
        "openai_base_url": "https://api.example.com/v1"}, files={"file": ("input.pdf", b"offline")})
    assert response.status_code == 202


@pytest.mark.parametrize("value", ["", "   ", " https://api.example.com/v1 "])
def test_empty_server_address_uses_existing_default(monkeypatch, value):
    monkeypatch.setenv("LLM_BASE_URL", value)
    spec = importlib.util.spec_from_file_location("config_address_under_test", ROOT / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = value.strip() or "https://api.deepseek.com/v1"
    assert module.settings.llm_base_url == expected
