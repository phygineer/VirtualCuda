"""OpenAI-compatible API (same /v1 shape Ollama exposes) plus the visualizer pages.

Usage:
    .venv/bin/python server.py [--model <hf-repo-id>]     # http://localhost:8700

    # development: restart on .py changes, pages reload themselves on page/server changes
    VC_DEV=1 VC_MODEL=<hf-repo-id> .venv/bin/uvicorn server:app --port 8700 --reload --reload-exclude '.venv/*' --reload-exclude 'data/*'

Endpoints:
    GET  /                       chat page (static/index.html)
    GET  /sim                    step-by-step simulation page (static/sim.html)
    GET  /data                   dataset generation page (static/data.html)
    GET  /v1/models              OpenAI model list (loaded + cached causal LMs)
    POST /v1/chat/completions    OpenAI chat completions: streaming, tools / tool_calls.
                                 Naming another cached model switches to it, like Ollama.
                                 Send X-Session-Id to group calls into one recorded conversation.
    GET  /api/info               loaded model's architecture
    GET  /api/compat             loaded model's self-test report (compat.py)
    GET  /api/models             cached models (+ supported or not) and which one is loaded
    GET  /api/models/check       is a Hub repo supported? ?model=<hf-repo-id> (no download)
    POST /api/models/load        switch model {"model": "<hf-repo-id>"}; refuses unsupported models
    GET  /docs/supported-models.md   the supported-models guide shown in the "?" popup
    GET  /static/{name}.js       scripts shared by both pages ("?" popup, live reload)
    GET  /dev/livereload         dev mode (VC_DEV=1): event stream that makes pages reload on changes
    GET  /media/{file}           images / audio sent in chats (stored by conversation.py)
    GET  /api/tools              built-in tool schemas (tools.py)
    POST /api/tools/run          execute a built-in tool {"name", "arguments"}
    GET  /api/sessions           recorded conversations (?q= searches titles and messages)
    POST /api/sessions/delete    forget several conversations {"ids": [...]} or all {"all": true}
    GET  /api/sessions/{id}      one conversation's generations
    PATCH /api/sessions/{id}     rename a conversation {"title": "..."}
    DELETE /api/sessions/{id}    forget a conversation
    POST /api/trace              SSE trace of a fresh prompt, or of a whole conversation's next reply
    POST /api/replay             SSE trace replaying a recorded generation (teacher-forced)
    GET  /api/datasets           prompt CSVs (dataset.py) with how many rows have output, + the job
    POST /api/datasets           upload a CSV {"filename", "csv"}; it needs a "prompt" column
    GET  /api/datasets/{name}    one dataset's rows
    GET  /api/datasets/{name}/csv    download it
    DELETE /api/datasets/{name}  delete it
    POST /api/datasets/{name}/generate   fill in the model's reply to every prompt (background job)
    GET  /api/datasets/{name}/training      how much of its training data (training.py) is built
    POST /api/datasets/{name}/capture       build it: one teacher-forced pass per generated row (job)
    GET  /api/datasets/{name}/training.zip  download it: data.csv + manifest.json + rows/*.safetensors
    GET  /api/job                that job's progress
    POST /api/job/stop           stop it after the current prompt (that prompt is redone next time)
"""

import argparse
import asyncio
import csv
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import compat
import conversation as conv
import dataset
import store
import tools
import training
from models import DEFAULT_MODEL, LoadedModel, cached_models
from tracer import Tracer

PORT = 8700
STATIC = Path(__file__).parent / "static"
DOCS = Path(__file__).parent / "docs"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

lm: LoadedModel = None
tracer: Tracer = None
# The model isn't reentrant: one generation (or trace, or model switch) at a time.
gen_lock = threading.Lock()
# Dev mode (VC_DEV=1): pages reload themselves when a page file changes or the server restarts.
DEV = os.environ.get("VC_DEV") == "1"
BOOT_ID = uuid.uuid4().hex  # changes every time the process starts (e.g. uvicorn --reload)


@asynccontextmanager
async def lifespan(app):
    # Under `uvicorn server:app` (e.g. with --reload) __main__ doesn't run, so load here.
    if lm is None:
        model_id = os.environ.get("VC_MODEL", DEFAULT_MODEL)
        try:
            load_model(model_id)
        except Unsupported as e:
            print_unsupported(model_id, e.report)
            raise SystemExit(1)
    yield


app = FastAPI(title="VirtualCuda", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Unsupported(Exception):
    def __init__(self, report):
        super().__init__(f"{report['model']} is not supported")
        self.report = report


def _load_checked(model_id):
    """Load a model and self-test it; returns (model or None, report)."""
    try:
        new = LoadedModel(model_id)
    except Exception as e:
        return None, compat.report(model_id, [compat.check("Load", False, f"Couldn't load: {e}")])
    try:
        rep = compat.self_test(new)
    except Exception as e:
        rep = compat.report(model_id, [compat.check("Self-test", False, f"Forward pass with hooks failed: {e}")])
    if not rep["supported"]:
        new.unload()
        return None, rep
    new.compat = rep
    return new, rep


def load_model(model_id):
    """Swap the loaded model, only if it's supported (compat.py). Caller must hold gen_lock.
    On failure the previous model is reloaded and Unsupported carries the report."""
    global lm, tracer
    if lm is not None and lm.id == model_id:
        return
    pre = compat.precheck(model_id)
    if not pre["supported"]:
        raise Unsupported(pre)
    prev = lm.id if lm else None
    if lm is not None:
        lm.unload()
        lm = tracer = None
    new, rep = _load_checked(model_id)
    if new is None:
        if prev:
            lm, _ = _load_checked(prev)
            tracer = Tracer(lm)
        raise Unsupported(rep)
    lm = new
    tracer = Tracer(lm)


class Message(BaseModel):
    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    tools: list[dict] | None = None
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream_options: dict | None = None


class TraceRequest(BaseModel):
    prompt: str = ""
    messages: list[dict] | None = None  # a conversation (OpenAI messages): trace its next reply
    chat: bool = True
    max_tokens: int = 48
    temperature: float = 0.7
    top_p: float = 0.9


class ReplayRequest(BaseModel):
    session: str
    generation: int


class LoadRequest(BaseModel):
    model: str


class Rename(BaseModel):
    title: str


class BulkDelete(BaseModel):
    ids: list[str] = []
    all: bool = False


class DatasetUpload(BaseModel):
    filename: str
    csv: str


class DatasetGenerate(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.0  # greedy by default; replay is exact either way once ids are saved
    top_p: float = 1.0
    redo: bool = False  # clear every row's output first


class ToolRun(BaseModel):
    name: str
    arguments: dict | str | None = None


class StopFlag(StoppingCriteria):
    """Lets us abort generate() when the client disconnects (e.g. the UI's Stop button)."""

    def __init__(self):
        self.event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool,
                          device=input_ids.device)


def specials():
    """Regex of the loaded tokenizer's special tokens (cached per model)."""
    if getattr(lm, "_specials", None) is None:
        lm._specials = conv.special_pattern(lm.tok)
    return lm._specials


def start_generation(req: ChatRequest, session_id):
    """Kick off generate() on a background thread. Caller must already hold gen_lock;
    the thread releases it. The finished generation is recorded to the session log."""
    model, tok = lm.model, lm.tok
    messages = conv.normalize([m.model_dump(exclude_none=True) for m in req.messages])
    enc = conv.encode(lm, messages, req.tools)
    prompt_ids = enc.prompt_ids
    inputs = torch.tensor([prompt_ids], device=lm.device)
    extra = {k: v.to(lm.device) for k, v in enc.model_inputs.items()}

    if req.temperature > 0:
        sampling = dict(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        sampling = dict(do_sample=False)

    # Keep special tokens in the stream so tool-call markers can be spotted; sse() strips them.
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    stop = StopFlag()
    result = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0, "content": "",
              "tool_calls": [], "finish_reason": "stop", "generation": None, "error": None,
              "done": threading.Event()}
    t0 = time.time()

    def run():
        try:
            with torch.no_grad():
                out = model.generate(inputs, attention_mask=torch.ones_like(inputs), streamer=streamer,
                                     max_new_tokens=req.max_tokens,
                                     stopping_criteria=StoppingCriteriaList([stop]), **extra, **sampling)
            completion_ids = out[0, len(prompt_ids):].tolist()
            raw = tok.decode(completion_ids, skip_special_tokens=False)
            if req.tools:
                content, calls = conv.parse_tool_calls(raw, specials())
            else:
                content, calls = conv.strip_special(raw, specials()), []
            n = len(completion_ids)
            reason = ("stopped" if stop.event.is_set() else "tool_calls" if calls
                      else "length" if n >= req.max_tokens else "stop")
            result.update(completion_tokens=n, content=content, tool_calls=calls, finish_reason=reason)
            result["generation"] = store.record(
                session_id, lm.id,
                {"messages": messages, "tools": req.tools, "temperature": req.temperature,
                 "top_p": req.top_p, "max_tokens": req.max_tokens},
                prompt_ids, completion_ids, content, calls, reason, time.time() - t0)
        except Exception as e:
            result["error"] = repr(e)
            streamer.end()
        finally:
            result["done"].set()
            gen_lock.release()

    Thread(target=run, daemon=True).start()
    return streamer, stop, result


class Progress(StoppingCriteria):
    """Reports each new token of a dataset row to the Data page, and stops on request."""

    def __init__(self, start, on_step, stop_event):
        self.start, self.on_step, self.stop_event = start, on_step, stop_event

    def __call__(self, input_ids, scores, **kwargs):
        new = input_ids[0, self.start:]
        if len(new) % 4 == 1:  # decoding every token would cost more than it shows
            self.on_step(len(new), conv.strip_special(lm.tok.decode(new, skip_special_tokens=False), specials()))
        else:
            self.on_step(len(new), None)
        return torch.full((input_ids.shape[0],), self.stop_event.is_set(), dtype=torch.bool,
                          device=input_ids.device)


def generate_prompt(prompt, settings, on_step, stop_event):
    """One dataset row: the loaded model's reply to `prompt` as a single user turn. Takes gen_lock
    itself, per prompt, so other requests can run between rows of a job."""
    with gen_lock:
        ids = conv.encode(lm, [{"role": "user", "content": prompt}]).prompt_ids
        inputs = torch.tensor([ids], device=lm.device)
        if settings["temperature"] > 0:
            sampling = dict(do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"])
        else:
            sampling = dict(do_sample=False)
        t0 = time.time()
        with torch.no_grad():
            out = lm.model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                    max_new_tokens=settings["max_tokens"],
                                    stopping_criteria=StoppingCriteriaList([Progress(len(ids), on_step, stop_event)]),
                                    **sampling)
        completion = out[0, len(ids):].tolist()
        text = conv.strip_special(lm.tok.decode(completion, skip_special_tokens=False), specials())
        return {"model": lm.id, "prompt_ids": ids, "completion_ids": completion, "text": text.strip(),
                "finish_reason": "length" if len(completion) >= settings["max_tokens"] else "stop",
                "stopped": stop_event.is_set(), "seconds": time.time() - t0}


job = dataset.Job()


def usage(result):
    return {"prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}


async def acquire_and_maybe_switch(model_id):
    """Take the model lock; if a different cached model was requested, load it first (Ollama-style)."""
    await asyncio.to_thread(gen_lock.acquire)
    if model_id and model_id != lm.id and model_id in {m["id"] for m in cached_models()}:
        try:
            await asyncio.to_thread(load_model, model_id)
        except Unsupported as e:
            gen_lock.release()
            raise HTTPException(400, e.report)
        except Exception:
            gen_lock.release()
            raise


NO_CACHE = {"Cache-Control": "no-store"}  # pages change often while we iterate


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/sim")
def sim():
    return FileResponse(STATIC / "sim.html", headers=NO_CACHE)


@app.get("/data")
def data_page():
    return FileResponse(STATIC / "data.html", headers=NO_CACHE)


@app.get("/media/{name}")
def media(name: str):
    """Images / audio that were sent in chats (content-addressed, see conversation.py)."""
    try:
        return FileResponse(conv.media_file(f"/media/{name}"))
    except conv.BadRequest:
        raise HTTPException(404, "no such media")


@app.get("/static/{name}.js")
def page_script(name: str):
    """Scripts shared by the pages (docs.js, livereload.js)."""
    path = STATIC / f"{name}.js"
    if not re.fullmatch(r"[a-z]+", name) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/javascript", headers=NO_CACHE)


def page_files_signature():
    """Changes whenever a page file (static/, docs/) is edited."""
    files = [p for d in (STATIC, DOCS) for p in d.rglob("*") if p.is_file()]
    return f"{len(files)}:{max((p.stat().st_mtime_ns for p in files), default=0)}"


@app.get("/dev/livereload")
async def livereload():
    """Dev mode only: an event stream the pages watch, so they reload on page edits or restarts."""
    if not DEV:
        return Response(status_code=204)  # pages see this once and stop listening

    # Each stream ends after a few seconds and the page reconnects (livereload.js). An endless
    # stream would keep uvicorn's graceful shutdown waiting forever, so --reload would hang.
    async def events():
        last = None
        for _ in range(20):
            sig = page_files_signature()
            if sig != last:
                last = sig
                yield f"data: {json.dumps({'boot': BOOT_ID, 'pages': sig})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream", headers=NO_CACHE)


@app.get("/api/info")
def info():
    return lm.info()


@app.get("/api/compat")
def current_compat():
    return lm.compat


@app.get("/api/models")
def list_models():
    cached = []
    for m in cached_models():
        checks = compat.config_checks(m.pop("config"))
        failed = [c["detail"] for c in checks if not c["ok"]]
        cached.append({**m, "supported": not failed, "reason": failed[0] if failed else None})
    return {"loaded": lm.id, "cached": cached}


@app.get("/api/models/check")
def check_model(model: str):
    """Pre-download check: is this Hub repo supported? Reads only config.json + metadata."""
    return compat.precheck(model)


@app.post("/api/models/load")
async def switch_model(req: LoadRequest):
    await asyncio.to_thread(gen_lock.acquire)
    try:
        await asyncio.to_thread(load_model, req.model)
    except Unsupported as e:
        raise HTTPException(400, e.report)
    finally:
        gen_lock.release()
    return lm.info()


@app.get("/docs/supported-models.md")
def supported_models_doc():
    return FileResponse(DOCS / "supported-models.md", media_type="text/markdown; charset=utf-8", headers=NO_CACHE)


@app.get("/api/tools")
def list_tools():
    return tools.schemas()


@app.post("/api/tools/run")
def run_tool(req: ToolRun):
    args = req.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return {"result": f"Error: arguments are not valid JSON: {args}"}
    return {"result": tools.run(req.name, args)}


@app.get("/api/sessions")
def sessions(q: str = ""):
    return store.list_sessions(q=q.strip())


@app.post("/api/sessions/delete")
def delete_sessions(req: BulkDelete):
    return {"deleted": store.delete_sessions(None if req.all else req.ids)}


@app.get("/api/sessions/{session_id}")
def session(session_id: str):
    s = store.get_session(session_id)
    if not s:
        raise HTTPException(404, "no such session")
    return s


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: Rename):
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "title can't be empty")
    if not store.rename_session(session_id, title):
        raise HTTPException(404, "no such session")
    return {"id": session_id, "title": title[:80]}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not store.delete_session(session_id):
        raise HTTPException(404, "no such session")
    return {"deleted": session_id}


def stream_trace(make_events):
    """Run a tracer generator on a worker thread (holding gen_lock) and stream it as SSE."""
    events = queue.Queue()
    stop = threading.Event()

    def produce():
        try:
            for ev in make_events(stop):
                events.put(ev)
        except Exception as e:
            events.put({"type": "error", "message": repr(e)})
        finally:
            events.put(None)
            gen_lock.release()

    Thread(target=produce, daemon=True).start()

    async def sse():
        try:
            while (ev := await asyncio.to_thread(events.get)) is not None:
                yield f"data: {json.dumps(ev)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            stop.set()  # aborts the trace if the page went away mid-run

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/api/trace")
async def trace(req: TraceRequest):
    await asyncio.to_thread(gen_lock.acquire)
    try:
        if req.messages:  # same template path as /v1/chat/completions
            enc = conv.encode(lm, conv.normalize(req.messages))
            ids, inputs, media = enc.prompt_ids, enc.model_inputs, enc.media
        else:
            ids, inputs, media = tracer.prompt_ids(req.prompt, req.chat), None, None
    except conv.BadRequest as e:
        gen_lock.release()
        raise HTTPException(400, str(e))
    except Exception:
        gen_lock.release()
        raise
    return stream_trace(lambda stop: tracer.run(ids, req.max_tokens, req.temperature, req.top_p, stop,
                                                model_inputs=inputs, media_files=media))


@app.post("/api/replay")
async def replay(req: ReplayRequest):
    s = store.get_session(req.session)
    g = store.get_generation(req.session, req.generation)
    if not s or not g:
        raise HTTPException(404, "no such generation")
    await acquire_and_maybe_switch(s["model"])
    if lm.id != s["model"]:
        gen_lock.release()
        raise HTTPException(409, f"recorded with {s['model']}, which is not in the local cache")

    # Context passing: how much of this prompt is the previous call's prompt + reply, re-fed?
    reused = 0
    if req.generation > 0:
        p = store.get_generation(req.session, req.generation - 1)
        prev = p["prompt_ids"] + p["completion_ids"]
        while reused < min(len(prev), len(g["prompt_ids"])) and prev[reused] == g["prompt_ids"][reused]:
            reused += 1
    extra = {"replay": {
        "session": req.session, "generation": req.generation,
        "generations": len(s["generations"]), "reused": reused,
        "finish_reason": g["finish_reason"], "tool_calls": g["tool_calls"],
        "tools": [t["function"]["name"] for t in g["request"].get("tools") or []],
        "messages": [{"role": m["role"], "tool_calls": bool(m.get("tool_calls"))}
                     for m in g["request"]["messages"]],
    }}
    r = g["request"]
    # Images/audio: rebuild the processor tensors from the stored files; they must reproduce the
    # exact prompt ids that were recorded, or the replay wouldn't be the same forward pass.
    try:
        enc = conv.encode(lm, r["messages"], r.get("tools"))
    except conv.BadRequest as e:
        gen_lock.release()
        raise HTTPException(409, f"can't rebuild this prompt: {e}")
    if enc.prompt_ids != g["prompt_ids"]:
        gen_lock.release()
        raise HTTPException(409, "rebuilt prompt doesn't match the recorded token ids (template or media changed)")
    return stream_trace(lambda stop: tracer.run(g["prompt_ids"], 0, r["temperature"], r["top_p"],
                                                stop, forced=g["completion_ids"], extra_meta=extra,
                                                model_inputs=enc.model_inputs, media_files=enc.media))


@app.get("/api/datasets")
def datasets():
    return {"datasets": dataset.list_all(), "job": job.state()}


@app.post("/api/datasets")
def upload_dataset(req: DatasetUpload):
    try:
        name = dataset.save_upload(req.filename, req.csv)
    except (dataset.BadDataset, csv.Error) as e:
        raise HTTPException(400, str(e))
    return dataset.summary(name)


def _dataset_path(name):
    try:
        p = dataset.path(name)
    except dataset.BadDataset as e:
        raise HTTPException(400, str(e))
    if not p.exists():
        raise HTTPException(404, f"no dataset {name}")
    return p


@app.get("/api/datasets/{name}")
def get_dataset(name: str):
    _dataset_path(name)
    fields, rows = dataset.read(name)
    return {**dataset.summary(name), "fields": fields, "items": rows}


@app.get("/api/datasets/{name}/csv")
def download_dataset(name: str):
    return FileResponse(_dataset_path(name), media_type="text/csv", filename=f"{name}.csv",
                        headers=NO_CACHE)


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    _dataset_path(name)
    if job.running() and job.st["name"] == name:
        raise HTTPException(409, "stop the job before deleting its dataset")
    dataset.delete(name)
    training.delete(name)
    return {"deleted": name}


@app.post("/api/datasets/{name}/generate")
def generate_dataset(name: str, req: DatasetGenerate):
    _dataset_path(name)
    try:
        job.start(name, req.model_dump(), generate_prompt)
    except dataset.BadDataset as e:
        raise HTTPException(409, str(e))
    return job.state()


@app.get("/api/datasets/{name}/training")
def training_status(name: str):
    _dataset_path(name)
    return training.status(name)


def capture_one(name):
    def work(index, row):
        prompt_ids, output_ids = training.token_ids(row)
        with gen_lock:
            if row.get("model") != lm.id:
                raise RuntimeError(f"row {row.get('id')} was generated by {row.get('model')}, "
                                   f"but {lm.id} is loaded; switch models to capture it")
            tensors = training.capture_row(lm, prompt_ids, output_ids)
        return training.save_row(name, index, row, tensors, len(prompt_ids))
    return work


@app.post("/api/datasets/{name}/capture")
def capture_dataset(name: str):
    _dataset_path(name)
    if job.running():
        raise HTTPException(409, f"busy with {job.st['name']}")
    fields, rows = dataset.read(name)
    # Rows generated before prompt ids were saved: rebuild them (the chat template is deterministic).
    missing = [r for r in rows if dataset.is_done(r) and not r.get("prompt_token_ids")]
    if missing:
        with gen_lock:
            for r in missing:
                ids = conv.encode(lm, [{"role": "user", "content": r["prompt"]}]).prompt_ids
                if r.get("prompt_tokens") and int(r["prompt_tokens"]) != len(ids):
                    raise HTTPException(409, f"row {r.get('id')}: prompt re-encodes to {len(ids)} tokens, "
                                             f"not {r['prompt_tokens']}; regenerate it")
                r["prompt_token_ids"] = json.dumps(ids)
        fields += [f for f in dataset.OUT_FIELDS if f not in fields]
        dataset.write(name, fields, rows)
    todo = [(i, r) for i, r in enumerate(rows, 1) if dataset.is_done(r) and not training.is_fresh(name, i, r)]
    total = sum(map(dataset.is_done, rows))
    if not total:
        raise HTTPException(409, "generate some outputs first")
    try:
        job.start_capture(name, todo, total, capture_one(name))
    except dataset.BadDataset as e:
        raise HTTPException(409, str(e))
    return job.state()


@app.get("/api/datasets/{name}/training.zip")
async def download_training(name: str):
    _dataset_path(name)
    if job.running() and job.st["name"] == name:
        raise HTTPException(409, "wait for the current job on this dataset to finish")
    st = training.status(name)
    if not st["captured"]:
        raise HTTPException(409, "build the training data first")
    pipe = training.download_zip(name, training.manifest(name, lm))

    async def body():
        try:
            while (chunk := await asyncio.to_thread(pipe.q.get)) is not None:
                yield chunk
        finally:
            pipe.aborted.set()  # client went away: stop the writer thread

    return StreamingResponse(body(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}-training.zip"',
                                      **NO_CACHE})


@app.get("/api/job")
def job_state():
    return job.state()


@app.post("/api/job/stop")
def stop_job():
    job.stop()
    return job.state()


@app.get("/v1/models")
def models():
    ids = [lm.id] + [m["id"] for m in cached_models() if m["id"] != lm.id]
    return {"object": "list", "data": [
        {"id": i, "object": "model", "created": 0, "owned_by": "virtualcuda"} for i in ids]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    session_id = request.headers.get("x-session-id", "")
    if not SESSION_ID.match(session_id):
        session_id = f"api-{uuid.uuid4().hex[:12]}"

    await acquire_and_maybe_switch(req.model)
    try:
        streamer, stop, result = start_generation(req, session_id)
    except conv.BadRequest as e:
        gen_lock.release()
        raise HTTPException(400, str(e))
    except Exception:
        gen_lock.release()
        raise

    model_id = lm.id
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta, reason=None, **extra):
        body = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}], **extra}
        return f"data: {json.dumps(body)}\n\n"

    async def wait_done():
        await asyncio.to_thread(result["done"].wait)
        if result["error"]:
            raise HTTPException(500, result["error"])

    def x_session():
        return {"id": session_id, "generation": result["generation"]}

    if not req.stream:
        async for _ in _aiter(streamer):
            pass
        await wait_done()
        message = {"role": "assistant", "content": result["content"] or None}
        if result["tool_calls"]:
            message["tool_calls"] = result["tool_calls"]
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}],
            "usage": usage(result), "x_session": x_session(),
        }

    async def sse():
        try:
            yield chunk({"role": "assistant", "content": ""})
            raw, sent = "", ""
            async for piece in _aiter(streamer):
                raw += piece
                if req.tools:  # stop showing text where a tool call begins
                    cuts = [raw.index(s) for s in conv.TOOL_STARTS if s in raw]
                    safe = min(cuts) if cuts else len(raw) - conv.held_back(raw)
                else:
                    safe = len(raw)
                visible = conv.strip_special(raw[:safe], specials())
                if len(visible) > len(sent) and visible.startswith(sent):
                    yield chunk({"content": visible[len(sent):]})
                    sent = visible
            await wait_done()
            if not result["tool_calls"] and result["content"].startswith(sent) and len(result["content"]) > len(sent):
                yield chunk({"content": result["content"][len(sent):]})
            if result["tool_calls"]:
                yield chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(result["tool_calls"])]})
            yield chunk({}, result["finish_reason"], x_session=x_session())
            if (req.stream_options or {}).get("include_usage"):
                body = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": model_id, "choices": [], "usage": usage(result)}
                yield f"data: {json.dumps(body)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            stop.event.set()  # no-op if finished; aborts generate() if the client went away

    return StreamingResponse(sse(), media_type="text/event-stream")


async def _aiter(streamer):
    """Iterate the blocking TextIteratorStreamer without blocking the event loop."""
    while True:
        piece = await asyncio.to_thread(next, streamer, None)
        if piece is None:
            return
        yield piece


def print_unsupported(model_id, report):
    print(f"{model_id} is not supported (see docs/supported-models.md):")
    for c in report["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['label']}: {c['detail']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id to load at start")
    args = parser.parse_args()
    try:
        load_model(args.model)
    except Unsupported as e:
        print_unsupported(args.model, e.report)
        raise SystemExit(1)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
