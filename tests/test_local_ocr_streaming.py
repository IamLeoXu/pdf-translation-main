from pathlib import Path
import json, os, subprocess, sys, threading, time
from types import SimpleNamespace
import importlib.util

import pytest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'pipeline'))
from foundation.shared.process_stream import run_streamed_command
from services.ocr_provider.local_progress import LocalOcrProgress, PROGRESS_PREFIX, _MESSAGES
from services.pipeline_shared.events import PipelineEventWriter, pipeline_event_writer_scope


def test_child_output_is_forwarded_before_process_exits(tmp_path):
    done=tmp_path/'done'
    code='import pathlib,time; print("first",flush=True); time.sleep(0.4); pathlib.Path('+repr(str(done))+').write_text("done"); print("last",flush=True)'
    live=[]
    thread=threading.get_ident()
    def receive(line):
        live.append((line.strip(),done.exists(),threading.get_ident()))
    result=run_streamed_command([sys.executable,'-u','-c',code],on_stdout=receive,timeout=3)
    assert result.returncode==0
    assert live[0]==('first',False,thread)
    assert live[-1][0]=='last'
    assert result.stdout=='first\nlast\n'


def test_streamed_worker_progress_is_saved_while_child_is_running(tmp_path):
    done=tmp_path/'done'
    events=tmp_path/'pipeline_events.jsonl'
    frame=PROGRESS_PREFIX+json.dumps({'phase':'page_completed','current':1,'total':3,'scope':'document'})
    code='import pathlib,time; print('+repr(frame)+',flush=True); time.sleep(0.4); pathlib.Path('+repr(str(done))+').write_text("done")'
    progress=LocalOcrProgress('local')
    seen=[]
    def receive(line):
        progress.stdout(line)
        records=[json.loads(item) for item in events.read_text('utf-8').splitlines()]
        seen.append((records[-1]['progress_current'],done.exists()))
    writer=PipelineEventWriter(job_id='streaming-test',job_root=tmp_path,logs_dir=tmp_path)
    with pipeline_event_writer_scope(writer):
        run_streamed_command([sys.executable,'-u','-c',code],on_stdout=receive,timeout=3)
    assert seen==[(1,False)]
    record=json.loads(events.read_text('utf-8').splitlines()[-1])
    assert record['substage']=='local_provider'
    assert record['progress_total']==3
    assert record['progress_unit']=='page'


def test_timeout_retains_error_tail_and_stops_child():
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_streamed_command([sys.executable,'-u','-c','import sys,time; print("started",flush=True); print("reason",file=sys.stderr,flush=True); time.sleep(5)'],timeout=0.35)
    assert 'started' in error.value.output
    assert 'reason' in error.value.stderr


def test_tick_does_not_invent_completed_pages(monkeypatch):
    records=[]
    from services.ocr_provider import local_progress
    monkeypatch.setattr(local_progress,'emit_stage_progress',lambda **kwargs: records.append(kwargs))
    progress=LocalOcrProgress('local')
    progress.accept(PROGRESS_PREFIX+json.dumps({'phase':'page_started','current':0,'page':1,'total':3}))
    run_streamed_command([sys.executable,'-u','-c','import time; time.sleep(0.22)'],on_tick=progress.tick,tick_interval=0.05,timeout=3)
    assert len(records)>1
    assert {record['progress_current'] for record in records}=={0}
    assert any(record['payload']['heartbeat'] for record in records)


def test_image_crop_progress_has_separate_unit_and_substage(monkeypatch):
    records=[]
    from services.ocr_provider import local_progress
    monkeypatch.setattr(local_progress,'emit_stage_progress',lambda **kwargs: records.append(kwargs))
    progress=LocalOcrProgress('local')
    assert progress.accept(PROGRESS_PREFIX+json.dumps({'phase':'page_completed','current':1,'total':1,'scope':'image'}))
    assert records[-1]['substage']=='local_provider_image_reocr'
    assert records[-1]['payload']['progress_unit']=='region'


@pytest.mark.parametrize('locale',['en_US','it_IT','de_DE','zh_CN','zh_TW','ja_JP','tr_TR','ar_SA'])
def test_local_ocr_progress_copy_covers_supported_locales(locale):
    assert locale in _MESSAGES
    assert len(_MESSAGES[locale])==7
    assert all(_MESSAGES[locale])


def test_invalid_progress_is_not_interpreted_as_success():
    progress=LocalOcrProgress('local')
    for frame in ('garbage','[]','null','{"phase":"page_completed","current":4,"total":3}'):
        assert not progress.accept(PROGRESS_PREFIX+frame)


def test_worker_uses_one_model_set_per_document_and_closes_it(monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('latency_worker_test',PROJECT/'local_ocr/embedded_paddleocr.py')
    worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    class Page:
        def get_pixmap(self,**kwargs):
            assert kwargs['dpi']==200
            return SimpleNamespace(samples=b'\xff\xff\xff',height=1,width=1)
    class Document:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __len__(self): return 2
        def __iter__(self): return iter([Page(),Page()])
    monkeypatch.setitem(sys.modules,'pymupdf',SimpleNamespace(open=lambda source:Document(),csRGB='RGB'))
    class Pipeline:
        calls=0; closed=False
        def predict(self,**kwargs):
            self.calls+=1
            return [SimpleNamespace(json={'res':{'parsing_res_list':[], 'layout_det_res':{'boxes':[]}}})]
        def close(self): self.closed=True
    pipeline=Pipeline(); created=[]
    monkeypatch.setattr(worker,'create_pipeline',lambda lang,**kwargs:created.append(lang) or pipeline)
    result=worker.recognize_pdf(Path('input.pdf'),'en')
    frames=[json.loads(line[len(PROGRESS_PREFIX):]) for line in capsys.readouterr().out.splitlines()]
    assert created==['en']
    assert pipeline.calls==2 and pipeline.closed
    assert [frame['current'] for frame in frames if frame['phase']=='page_completed']==[1,2]
    assert len(result['_meta']['local_ocr_timings']['page_seconds'])==2
    assert result['dataInfo']['numPages']==2



def test_table_detection_uses_raw_layout_and_is_conservative():
    spec=importlib.util.spec_from_file_location("lazy_table_detection_test",PROJECT/"local_ocr/embedded_paddleocr.py")
    worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    assert worker._table_layout_required({"prunedResult":{"parsing_res_list":[]}})
    assert worker._table_layout_required({"prunedResult":{"layout_det_res":{"boxes":[{"cls_id":1}]}}})
    assert worker._table_layout_required({"prunedResult":{"layout_det_res":{"boxes":[{"label":"table"}]},"parsing_res_list":[]}})
    assert not worker._table_layout_required({"prunedResult":{"layout_det_res":{"boxes":[{"label":"text"}]}}})


def test_lazy_table_page_is_rerun_with_original_full_pipeline(monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location("lazy_table_worker_test",PROJECT/"local_ocr/embedded_paddleocr.py")
    worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    class Page:
        def get_pixmap(self,**kwargs): return SimpleNamespace(samples=b"\xff\xff\xff",height=1,width=1)
    class Document:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __len__(self): return 2
        def __iter__(self): return iter([Page(),Page()])
    monkeypatch.setitem(sys.modules,"pymupdf",SimpleNamespace(open=lambda source:Document(),csRGB="RGB"))
    pipelines=[]
    class Pipeline:
        def __init__(self,table): self.table=table; self.calls=0; self.closed=False
        def predict(self,**kwargs):
            self.calls+=1
            return [SimpleNamespace(json={"res":{"layout_det_res":{"boxes":[{"label":"table"}]},"parsing_res_list":[
                {"block_label":"table","block_content":"<table><tr><td>Wert</td></tr></table>" if self.table else "","block_bbox":[0,0,1,1]}
            ]}})]
        def close(self): self.closed=True
    def create(lang,*,table_recognition):
        assert lang=="de"
        if table_recognition: assert pipelines[0].closed
        pipeline=Pipeline(table_recognition); pipelines.append(pipeline); return pipeline
    monkeypatch.setattr(worker,"create_pipeline",create)
    result=worker.recognize_pdf(Path("source.pdf"),"de")
    assert [pipeline.table for pipeline in pipelines]==[False,True]
    assert [pipeline.calls for pipeline in pipelines]==[1,2]
    assert all(pipeline.closed for pipeline in pipelines)
    assert all("Wert" in page["prunedResult"]["parsing_res_list"][0]["block_content"] for page in result["layoutParsingResults"])
    assert result["_meta"]["local_ocr_timings"]["table_models_loaded"]
