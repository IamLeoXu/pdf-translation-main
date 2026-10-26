from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from services.ocr_provider.local_command_driver import _command_error_detail


PROJECT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "local_ocr" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embedded = load("embedded_ocr_under_test", "embedded_paddleocr.py")
wrapper = load("local_ocr_wrapper_under_test", "paddlex_paddleocr.py")


@pytest.mark.parametrize("language,expected", [
    ("de", "de"), ("it-IT", "it"), ("en_US", "en"), ("zh-CN", "ch"),
    ("zh-TW", "chinese_cht"), ("ja", "japan"), ("tr", "tr"), ("ar", "ar"),
])
def test_source_language_selects_recognizer(language, expected):
    assert embedded.recognition_language(language) == expected


def test_local_paddle_result_keeps_layout_table_and_pixel_coordinates():
    result = SimpleNamespace(json={"res": {"parsing_res_list": [
        {"block_label": "table", "block_content": "<table><tr><td>Öl</td></tr></table>", "block_bbox": [30, 40, 600, 300]}
    ]}})
    page = embedded.paddle_page_payload(result, width=1654, height=2339)
    assert page["prunedResult"]["width"] == 1654
    assert page["prunedResult"]["height"] == 2339
    assert page["prunedResult"]["parsing_res_list"][0]["block_bbox"] == [30, 40, 600, 300]


def test_cpu_pipeline_disables_remote_and_gpu_runtime(monkeypatch):
    monkeypatch.delenv("RETAIN_LOCAL_OCR_MKLDNN", raising=False)
    seen = {}
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=lambda **kw: seen.update(kw)))
    embedded.create_pipeline("de")
    assert seen["device"] == "cpu"
    assert seen["enable_mkldnn"]
    assert seen["lang"] == "de"
    assert seen["use_table_recognition"]
    assert not seen["use_formula_recognition"]


def test_default_local_mode_never_calls_old_placeholder_http_service(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAIN_LOCAL_OCR_BACKEND", raising=False)
    monkeypatch.setenv("RETAIN_LOCAL_OCR_URL", "http://host.docker.internal:8080")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_PYTHON", sys.executable)
    monkeypatch.setenv("RETAIN_SOURCE_LANG", "de")
    output = tmp_path / "result.json"
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        output.write_text(json.dumps({"layoutParsingResults": []}), "utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wrapper, "run_streamed_command", run)
    monkeypatch.setattr(wrapper.requests, "post", lambda *a, **k: pytest.fail("unexpected HTTP request"))
    result, _log_id = wrapper._recognize(tmp_path / "source.pdf", output)
    assert result == {"layoutParsingResults": []}
    assert seen["command"][-2:] == ["--lang", "de"]


def test_existing_local_http_service_remains_available(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_BACKEND", "service")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_URL", "http://ocr-server:8080")
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF")
    seen = {}

    def post(url, **kwargs):
        seen["url"] = url
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "result": {"layoutParsingResults": []}, "logId": "local-test",
        })

    monkeypatch.setattr(wrapper.requests, "post", post)
    result, log_id = wrapper._recognize(source, tmp_path / "result.json")
    assert seen["url"] == "http://ocr-server:8080/layout-parsing"
    assert log_id == "local-test"


def test_failure_reports_actual_ocr_process_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_BACKEND", "embedded")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_PYTHON", sys.executable)
    monkeypatch.setattr(wrapper, "run_streamed_command", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="missing-model-file"))
    with pytest.raises(RuntimeError, match="missing-model-file"):
        wrapper._recognize(tmp_path / "input.pdf", tmp_path / "result.json")


def test_local_command_error_keeps_reason_and_removes_credentials():
    detail = _command_error_detail("model download failed; secret-123", {"LLM_API_KEY": "secret-123"})
    assert "model download failed" in detail
    assert "secret-123" not in detail


@pytest.mark.parametrize("returncode,reason", [(1,"onednn kernel unsupported"),(-11,"native error"),(3221225477,"access violation")])
def test_cpu_kernel_failure_retries_standard_kernels_once(tmp_path, monkeypatch, returncode, reason):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_BACKEND", "embedded")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_PYTHON", sys.executable)
    monkeypatch.delenv("RETAIN_LOCAL_OCR_MKLDNN", raising=False)
    output=tmp_path/"result.json"
    calls=[]
    def run(command, **kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            return SimpleNamespace(returncode=returncode,stdout="",stderr=reason)
        output.write_text(json.dumps({"layoutParsingResults":[]}),"utf-8")
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(wrapper,"run_streamed_command",run)
    result,_=wrapper._recognize(tmp_path/"input.pdf",output)
    assert len(calls)==2
    assert calls[1]["env"]["RETAIN_LOCAL_OCR_MKLDNN"]=="0"
    assert result=={"layoutParsingResults":[]}


def test_regular_model_error_does_not_retry_cpu_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_BACKEND", "embedded")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_PYTHON", sys.executable)
    monkeypatch.delenv("RETAIN_LOCAL_OCR_MKLDNN", raising=False)
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=1,stdout="",stderr="missing-model-file")
    monkeypatch.setattr(wrapper,"run_streamed_command",run)
    with pytest.raises(RuntimeError,match="missing-model-file"):
        wrapper._recognize(tmp_path/"input.pdf",tmp_path/"result.json")
    assert len(calls)==1


def test_cpu_optimization_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_MKLDNN", "0")
    seen={}
    monkeypatch.setitem(sys.modules,"paddleocr",SimpleNamespace(PPStructureV3=lambda **kwargs:seen.update(kwargs)))
    embedded.create_pipeline("en")
    assert not seen["enable_mkldnn"]
    assert seen["use_table_recognition"]


def test_disabled_optimization_does_not_repeat_failed_process(tmp_path,monkeypatch):
    monkeypatch.setenv("RETAIN_LOCAL_OCR_BACKEND","embedded")
    monkeypatch.setenv("RETAIN_LOCAL_OCR_PYTHON",sys.executable)
    monkeypatch.setenv("RETAIN_LOCAL_OCR_MKLDNN","0")
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=-11,stdout="",stderr="native error")
    monkeypatch.setattr(wrapper,"run_streamed_command",run)
    with pytest.raises(RuntimeError,match="native error"):
        wrapper._recognize(tmp_path/"input.pdf",tmp_path/"result.json")
    assert len(calls)==1
