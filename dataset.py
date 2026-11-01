"""Prompt datasets for activation capture (the Data page): CSV files under data/datasets/.

A dataset is any CSV with a `prompt` column (build_prompts.py makes one). Generating fills in,
for every row, the loaded model's reply to that prompt as a single user turn:

    output_text         the reply, special tokens removed
    output_token_ids    JSON list of the exact token ids the model produced
    prompt_token_ids    JSON list of the prompt's ids with the chat template applied
    model, prompt_tokens, completion_tokens, finish_reason, seconds

With the token ids saved, any row's activations can be reproduced exactly by teacher-forced
replay, so the CSV is the dataset; training.py turns it into per-token activations.

One job runs at a time on a background thread. The server's generate function takes the model
lock per prompt, so chat and Deep dive still get a turn between rows. Rows that already have
output are skipped, so a job resumes where it left off (after Stop, or a uvicorn --reload).
"""

import csv
import io
import json
import re
import threading
import time
from pathlib import Path

DIR = Path(__file__).parent / "data" / "datasets"
NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
OUT_FIELDS = ["output_text", "output_token_ids", "prompt_token_ids", "model", "prompt_tokens",
              "completion_tokens", "finish_reason", "seconds"]


class BadDataset(ValueError):
    pass


def path(name):
    if not NAME.match(name) or name.startswith("."):
        raise BadDataset(f"bad dataset name: {name!r}")
    return DIR / f"{name}.csv"


def read(name):
    with path(name).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write(name, fields, rows):
    """Atomic: a crash or reload mid-write never leaves a half-written CSV."""
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = path(name).with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path(name))


def is_done(row):
    return bool(row.get("output_token_ids"))


def summary(name):
    fields, rows = read(name)
    p = path(name)
    return {"name": name, "rows": len(rows), "done": sum(map(is_done, rows)),
            "tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
            "bytes": p.stat().st_size, "modified": p.stat().st_mtime}


def list_all():
    if not DIR.exists():
        return []
    out = [summary(p.stem) for p in DIR.glob("*.csv")]
    return sorted(out, key=lambda d: -d["modified"])


def save_upload(filename, text):
    """Store an uploaded CSV under a free name; returns that name."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        raise BadDataset("the CSV has no rows")
    fields = list(rows[0].keys())
    col = next((f for f in fields if f and f.strip().lower() == "prompt"), None)
    if col is None:
        raise BadDataset(f"the CSV needs a 'prompt' column (found: {', '.join(f for f in fields if f)})")
    if col != "prompt":
        fields[fields.index(col)] = "prompt"
        rows = [{("prompt" if k == col else k): v for k, v in r.items()} for r in rows]
    rows = [r for r in rows if (r["prompt"] or "").strip()]
    if not rows:
        raise BadDataset("every prompt is empty")
    if "id" not in fields:
        fields.insert(0, "id")
        for i, r in enumerate(rows, 1):
            r["id"] = str(i)
    fields += [f for f in OUT_FIELDS if f not in fields]

    base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip("-.") or "prompts"
    base, name, n = base[:70], base[:70], 2
    while path(name).exists():
        name, n = f"{base}-{n}", n + 1
    write(name, fields, rows)
    return name


def delete(name):
    path(name).unlink(missing_ok=True)


class Job:
    """The one running job (generating replies, or capturing training data), which the Data page
    polls to show progress. Both use the model, so only one runs at a time."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.st = {"running": False, "name": None}

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def state(self):
        st = dict(self.st, running=self.running())
        if st.get("started"):
            st["elapsed"] = (st.get("finished") or time.time()) - st["started"]
        return st

    def start(self, name, settings, generate):
        """generate(prompt, settings, on_step, stop_event) -> result dict (see server.py)."""
        if self.running():
            raise BadDataset(f"already generating {self.st['name']}")
        fields, rows = read(name)
        if settings.get("redo"):
            for r in rows:
                r.update({f: "" for f in OUT_FIELDS})
            write(name, fields, rows)
        self.stop_event.clear()
        self.st = {"running": True, "kind": "generate", "name": name, "settings": settings, "total": len(rows),
                   "done": sum(map(is_done, rows)), "done_at_start": sum(map(is_done, rows)),
                   "current": None, "tokens": 0, "partial": "", "gen_tokens": 0, "gen_seconds": 0.0,
                   "started": time.time(), "finished": None, "error": None, "recent": []}
        self.thread = threading.Thread(target=self._run, args=(name, settings, generate), daemon=True)
        self.thread.start()

    def start_capture(self, name, todo, total, work):
        """todo: [(row index, row)] still to capture; total: rows that will have data when done;
        work(index, row) -> bytes written (it takes the model lock itself)."""
        if self.running():
            raise BadDataset(f"busy with {self.st['name']}")
        self.stop_event.clear()
        self.st = {"running": True, "kind": "capture", "name": name, "total": total,
                   "done": total - len(todo), "done_at_start": total - len(todo), "current": None,
                   "bytes": 0, "started": time.time(), "finished": None, "error": None}
        self.thread = threading.Thread(target=self._run_capture, args=(todo, work), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run_capture(self, todo, work):
        st = self.st
        try:
            for index, row in todo:
                if self.stop_event.is_set():
                    break
                st["current"] = {"id": row.get("id"), "prompt": row["prompt"]}
                st["bytes"] += work(index, row)
                st["done"] += 1
        except Exception as e:
            st["error"] = repr(e)
        finally:
            st.update(current=None, finished=time.time(), stopped=self.stop_event.is_set())

    def _run(self, name, settings, generate):
        st = self.st
        try:
            fields, rows = read(name)
            fields += [f for f in OUT_FIELDS if f not in fields]
            for r in rows:
                if self.stop_event.is_set():
                    break
                if is_done(r):
                    continue
                st.update(current={"id": r.get("id"), "prompt": r["prompt"]}, tokens=0, partial="")

                def on_step(n, partial):  # partial is None on steps where the text wasn't re-decoded
                    st["tokens"] = n
                    if partial is not None:
                        st["partial"] = partial

                res = generate(r["prompt"], settings, on_step, self.stop_event)
                if res["stopped"]:
                    break  # leave this row empty so a later run redoes it
                r.update(output_text=res["text"], output_token_ids=json.dumps(res["completion_ids"]),
                         prompt_token_ids=json.dumps(res["prompt_ids"]),
                         model=res["model"], prompt_tokens=len(res["prompt_ids"]),
                         completion_tokens=len(res["completion_ids"]),
                         finish_reason=res["finish_reason"], seconds=f"{res['seconds']:.2f}")
                write(name, fields, rows)
                st["done"] += 1
                st["gen_tokens"] += len(res["completion_ids"])
                st["gen_seconds"] += res["seconds"]
                st["recent"] = [{"id": r.get("id"), "tokens": len(res["completion_ids"]),
                                 "seconds": res["seconds"], "finish_reason": res["finish_reason"]}
                                ] + st["recent"][:19]
        except Exception as e:
            st["error"] = repr(e)
        finally:
            st.update(current=None, partial="", finished=time.time(),
                      stopped=self.stop_event.is_set())
