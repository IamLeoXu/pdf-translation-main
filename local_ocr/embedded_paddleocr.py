"""CPU document OCR, run with the dedicated /opt/local-ocr interpreter."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def recognition_language(source_language: str) -> str:
    code = str(source_language or "auto").strip().lower().replace("_", "-")
    aliases = {
        "auto": "ch", "zh": "ch", "zh-cn": "ch", "zh-hans": "ch",
        "zh-tw": "chinese_cht", "zh-hant": "chinese_cht", "zh-hk": "chinese_cht",
        "ja": "japan", "ko": "korean", "german": "de", "italian": "it",
        "english": "en", "arabic": "ar", "japanese": "japan",
        "chinese": "ch", "turkish": "tr",
    }
    return aliases.get(code, code.split("-")[0])


def create_pipeline(language: str, *, table_recognition: bool = True, mkldnn: bool | None = None):
    # Import only in the isolated OCR process, never in the translation worker.
    from paddleocr import PPStructureV3

    return PPStructureV3(
        lang=recognition_language(language),
        device="cpu",
        enable_mkldnn=(os.environ.get("RETAIN_LOCAL_OCR_MKLDNN", "1").strip().lower() in {"1", "true", "yes"}) if mkldnn is None else mkldnn,
        cpu_threads=max(1, int(os.environ.get("RETAIN_LOCAL_OCR_CPU_THREADS", "4"))),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_region_detection=False,
        use_table_recognition=table_recognition,
        # The compact table model supports ruled and unruled tables and avoids
        # loading the 350MB SLANeXt model for CPU-only deployments.
        wired_table_structure_recognition_model_name="SLANet_plus",
    )


def paddle_page_payload(result, *, width: int, height: int) -> dict:
    value = result.json if hasattr(result, "json") else result
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise RuntimeError("local PaddleOCR returned a non-object result")
    pruned = dict(value.get("res", value))
    if not isinstance(pruned.get("parsing_res_list"), list):
        raise RuntimeError("local PaddleOCR result missing parsing_res_list")
    pruned.update(width=width, height=height)
    return {"prunedResult": pruned}


def _table_layout_required(page_payload: dict) -> bool:
    pruned = page_payload["prunedResult"]
    # Keep table support conservative if the engine omits its layout trace.
    layout = pruned.get("layout_det_res")
    if not isinstance(layout, dict) or not isinstance(layout.get("boxes"), list):
        return True
    boxes = layout["boxes"]
    if any(not isinstance(box, dict) or "label" not in box for box in boxes):
        return True
    return any(str(box["label"]).strip().lower() == "table" for box in boxes)


def _progress(phase: str, current: int, total: int, *, scope: str = "document", **extra) -> None:
    print("PDFTR_OCR_PROGRESS " + json.dumps({"phase": phase, "current": current, "total": total,
                                             "scope": scope, **extra}), flush=True)


def recognize_pdf(source: Path, language: str, *, progress_scope: str = "document") -> dict:
    import numpy as np
    import pymupdf

    started = time.perf_counter()
    dpi = max(96, min(300, int(os.environ.get("RETAIN_LOCAL_OCR_DPI", "200"))))
    pages, metadata, page_timings = [], [], []
    with pymupdf.open(source) as doc:
        total = len(doc)
        _progress("loading_models", 0, total, scope=progress_scope)
        lazy_tables = os.environ.get("RETAIN_LOCAL_OCR_LAZY_TABLES", "0").strip().lower() not in {"0", "false", "no"}
        tables_ready = not lazy_tables
        pipeline = create_pipeline(language, table_recognition=tables_ready)
        model_seconds = time.perf_counter() - started
        _progress("models_ready", 0, total, scope=progress_scope, phase_elapsed_ms=round(model_seconds * 1000))
        try:
            for index, page in enumerate(doc):
                page_started = time.perf_counter()
                table_load_seconds = 0.0
                _progress("page_started", index, total, scope=progress_scope, page=index + 1)
                pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
                predictions = iter(pipeline.predict(input=image[:, :, ::-1].copy()))
                result = next(predictions, None)
                if result is None:
                    raise RuntimeError(f"local PaddleOCR returned no result for page {index + 1}")
                page_payload = paddle_page_payload(result, width=pixmap.width, height=pixmap.height)
                if not tables_ready and _table_layout_required(page_payload):
                    # No quality downgrade: a detected table is recognized again
                    # with the original full table pipeline and identical pixels.
                    _progress("loading_models", index, total, scope=progress_scope, page=index + 1)
                    table_started = time.perf_counter()
                    previous_pipeline, pipeline = pipeline, None
                    previous_pipeline.close()
                    del previous_pipeline, result
                    import gc
                    gc.collect()
                    pipeline = create_pipeline(language, table_recognition=True)
                    tables_ready = True
                    table_load_seconds = time.perf_counter() - table_started
                    model_seconds += table_load_seconds
                    _progress("models_ready", index, total, scope=progress_scope, page=index + 1)
                    result = next(iter(pipeline.predict(input=image[:, :, ::-1].copy())), None)
                    if result is None:
                        raise RuntimeError(f"local PaddleOCR returned no table result for page {index + 1}")
                    page_payload = paddle_page_payload(result, width=pixmap.width, height=pixmap.height)
                pages.append(page_payload)
                metadata.append({"width": pixmap.width, "height": pixmap.height})
                seconds = time.perf_counter() - page_started
                page_timings.append(round(seconds - table_load_seconds, 3))
                _progress("page_completed", index + 1, total, scope=progress_scope,
                          phase_elapsed_ms=round(seconds * 1000))
        finally:
            if pipeline is not None:
                pipeline.close()
    timings = {"model_load_seconds": round(model_seconds, 3), "page_seconds": page_timings,
               "total_seconds": round(time.perf_counter() - started, 3), "dpi": dpi, "table_models_loaded": tables_ready,
               "mkldnn": os.environ.get("RETAIN_LOCAL_OCR_MKLDNN", "1").strip().lower() in {"1", "true", "yes"}}
    _progress("complete", total, total, scope=progress_scope, phase_elapsed_ms=round(timings["total_seconds"] * 1000))
    return {"layoutParsingResults": pages, "dataInfo": {"pages": metadata, "numPages": len(pages)},
            "_meta": {"local_ocr_timings": timings}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lang", default="auto")
    parser.add_argument("--progress-scope", choices=("document", "image"), default="document")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--languages", default="en,de,it,zh-CN,zh-TW,ja,tr,ar")
    args = parser.parse_args()
    if args.warmup:
        from paddlex.inference import create_predictor
        # Table OCR initializes these two helpers lazily on the first table.
        # Download them now so an offline container does not fetch at runtime.
        for model_name in ("PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_textline_ori"):
            helper = create_predictor(model_name=model_name, device="cpu")
            helper.close()
        for language in args.languages.split(","):
            pipeline = create_pipeline(language.strip(), mkldnn=False)
            # Execute inference during the build, not just model downloads.
            # A runtime/backend failure must fail the image build itself.
            import numpy as np
            result = next(iter(pipeline.predict(input=np.full((128, 512, 3), 255, dtype=np.uint8))), None)
            if result is None:
                raise RuntimeError(f"local PaddleOCR smoke test returned no page: {language}")
            print(f"Local PaddleOCR models ready: {language}", flush=True)
            pipeline.close()
            del result, pipeline
            import gc
            gc.collect()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required for recognition")
    result = recognize_pdf(args.input, args.lang, progress_scope=args.progress_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
