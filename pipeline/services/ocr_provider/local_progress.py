"""Translate embedded OCR progress records into the existing pipeline event API."""
from __future__ import annotations

import json
import os
import time
from services.pipeline_shared.events import emit_stage_progress

PROGRESS_PREFIX = "PDFTR_OCR_PROGRESS "
_MESSAGES = {
    "en_US": ("Starting local OCR", "Loading local OCR models", "Local OCR models ready", "Recognizing page {page}", "Local OCR page completed", "Local image-region OCR", "{message} (elapsed {seconds}s)"),
    "it_IT": ("Avvio OCR locale", "Caricamento dei modelli OCR locali", "Modelli OCR locali pronti", "Riconoscimento pagina {page}", "Pagina OCR locale completata", "OCR locale della regione immagine", "{message} (tempo trascorso: {seconds}s)"),
    "de_DE": ("Lokale OCR wird gestartet", "Lokale OCR-Modelle werden geladen", "Lokale OCR-Modelle sind bereit", "Seite {page} wird erkannt", "Lokale OCR-Seite abgeschlossen", "Lokale OCR des Bildbereichs", "{message} (vergangen: {seconds}s)"),
    "zh_CN": ("正在启动本地 OCR", "正在加载本地 OCR 模型", "本地 OCR 模型已就绪", "正在识别第 {page} 页", "本地 OCR 识别进度", "正在执行图像区域本地 OCR", "{message}（已用 {seconds} 秒）"),
    "zh_TW": ("正在啟動本機 OCR", "正在載入本機 OCR 模型", "本機 OCR 模型已就緒", "正在辨識第 {page} 頁", "本機 OCR 辨識進度", "正在執行影像區域本機 OCR", "{message}（已用 {seconds} 秒）"),
    "ja_JP": ("ローカル OCR を開始中", "ローカル OCR モデルを読み込み中", "ローカル OCR モデルの準備完了", "{page} ページを認識中", "ローカル OCR の認識進捗", "画像領域のローカル OCR を実行中", "{message}（経過 {seconds} 秒）"),
    "tr_TR": ("Yerel OCR başlatılıyor", "Yerel OCR modelleri yükleniyor", "Yerel OCR modelleri hazır", "{page}. sayfa tanınıyor", "Yerel OCR sayfası tamamlandı", "Görüntü bölgesi için yerel OCR", "{message} (geçen süre: {seconds} sn)"),
    "ar_SA": ("بدء التعرف المحلي", "تحميل نماذج التعرف المحلي", "نماذج التعرف المحلي جاهزة", "التعرف على الصفحة {page}", "اكتملت صفحة التعرف المحلي", "التعرف المحلي على منطقة الصورة", "{message} (الوقت المنقضي: {seconds} ثانية)"),
}
_PHASE_INDEX = {"starting": 0, "loading_models": 1, "models_ready": 2, "page_started": 3, "page_completed": 4, "complete": 4}


class LocalOcrProgress:
    def __init__(self, provider: str, *, locale: str | None = None):
        self.provider = provider
        self.messages = _MESSAGES.get(locale or os.getenv("RETAIN_UI_LOCALE", "zh_CN"), _MESSAGES["zh_CN"])
        self.started = time.monotonic()
        self.state = {"phase": "starting", "current": 0, "total": 0, "scope": "document"}

    def stdout(self, line: str) -> None:
        if not self.accept(line):
            print(line, end="", flush=True)

    def accept(self, line: str) -> bool:
        if not line.startswith(PROGRESS_PREFIX):
            return False
        try:
            record = json.loads(line[len(PROGRESS_PREFIX):])
            phase = str(record["phase"])
            current, total = int(record.get("current", 0)), int(record.get("total", 0))
            if phase not in _PHASE_INDEX or not 0 <= current <= total <= 1000000:
                return False
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        self.state = {**record, "phase": phase, "current": current, "total": total}
        self._emit(heartbeat=False)
        return True

    def tick(self) -> None:
        self._emit(heartbeat=True)

    def _emit(self, *, heartbeat: bool) -> None:
        state = self.state
        image = state.get("scope") == "image"
        message = self.messages[5 if image else _PHASE_INDEX[state["phase"]]].format(page=state.get("page", state["current"] + 1))
        elapsed = int(time.monotonic() - self.started)
        detail = self.messages[6].format(message=message, seconds=elapsed)
        emit_stage_progress(
            stage="ocr_processing", substage="local_provider_image_reocr" if image else "local_provider",
            message=message, stage_detail=detail, provider=self.provider,
            provider_stage=state["phase"],
            progress_current=state["current"] if state["total"] else None,
            progress_total=state["total"] or None,
            elapsed_ms=elapsed * 1000,
            payload={"progress_unit": "region" if image else "page", "local_ocr_phase": state["phase"],
                     "heartbeat": heartbeat, "phase_elapsed_ms": state.get("phase_elapsed_ms")},
        )
