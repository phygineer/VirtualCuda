"""Training data for the next phase: what the model computes for every token of a dataset.

For each generated row of a dataset (dataset.py), one teacher-forced forward pass over the
prompt (chat template applied) + the generated ids, with hooks on every decoder layer. Saved per
row as data/training/<dataset>/rows/NNNNN.safetensors, in bf16, i.e. exactly the values the
model computed:

    token_ids   [T]             int32   prompt + generated ids (the first n_prompt are the prompt)
    resid       [L+1, T, H]     bf16    residual stream (the Journey view's data, full width):
                                        resid[0] = embedding = input to layer 0,
                                        resid[i+1] = output of layer i
    attn_out    [L, T, H]       bf16    what attention adds to the residual at layer i,
                                        the only part of a layer that mixes positions
    scalar      [L]             float32 per-layer output multiplier (1 where the model has none)
    top_ids     [T, 10]         int32   the model's 10 most likely next tokens at each position
    top_logits  [T, 10]         float32 ... and their logits
    match       [T]             int8    1 if top_ids[t, 0] == token_ids[t + 1], else 0; -1 at the
                                        last position. Over the generated part this shows where
                                        the one-pass replay disagrees with step-by-step generation.

The per-token part of layer i (MLP + per-layer input) follows from the identity
    resid[i+1] = (resid[i] + attn_out[i] + mlp_out[i] + ple_out[i]) * scalar[i]
so  mlp_out[i] + ple_out[i] = resid[i+1] / scalar[i] - resid[i] - attn_out[i]  (up to bf16 rounding).

A row's file records the hash of its token ids, so files go stale (and are redone) when a row
is regenerated. download_zip() packs the CSV, manifest.json and every up-to-date row file.
"""

import hashlib
import json
import platform
import queue
import shutil
import threading
import time
import zipfile
from pathlib import Path

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from safetensors.torch import save_file

import dataset

DIR = Path(__file__).parent / "data" / "training"
TOP = 10
FORMAT = "virtualcuda.training"
VERSION = 1


def rows_dir(name):
    dataset.path(name)  # validates the name
    return DIR / name / "rows"


def row_file(name, index):
    return rows_dir(name) / f"{index:05d}.safetensors"


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def token_ids(row):
    """(prompt ids, output ids) of a generated row, or None if it has no output yet."""
    if not dataset.is_done(row) or not row.get("prompt_token_ids"):
        return None
    return json.loads(row["prompt_token_ids"]), json.loads(row["output_token_ids"])


def file_meta(path):
    try:
        with safe_open(str(path), "pt") as f:
            return f.metadata() or {}
    except Exception:
        return None


def is_fresh(name, index, row):
    ids = token_ids(row)
    if ids is None:
        return False
    meta = file_meta(row_file(name, index))
    return bool(meta) and meta.get("ids_sha") == ids_sha(ids[0] + ids[1]) and meta.get("model") == row.get("model")


def status(name):
    """How much of a dataset's training data is built."""
    _, rows = dataset.read(name)
    generated = fresh = size = 0
    for i, r in enumerate(rows, 1):
        if not dataset.is_done(r):
            continue
        generated += 1
        if is_fresh(name, i, r):
            fresh += 1
            size += row_file(name, i).stat().st_size
    return {"rows": len(rows), "generated": generated, "captured": fresh, "bytes": size,
            "ready": generated > 0 and fresh == generated}


class _Hooks:
    """Residual entering layer 0, every layer's output, and what attention adds, at every position."""

    def __init__(self, arch):
        self.arch = arch
        self.embed = None
        self.out = [None] * arch.n_layers
        self.attn = [None] * arch.n_layers
        self.handles = []

    def __enter__(self):
        a = self.arch
        self.handles.append(a.layers[0].register_forward_pre_hook(self._embed, with_kwargs=True))
        for i in range(a.n_layers):
            self.handles.append(a.layers[i].register_forward_hook(self._keep(self.out, i)))
            # with a sandwich norm, what's added is the post-norm's output; else the raw attention output
            src = a.attn_post[i] if a.attn_post[i] is not None else a.attn[i]
            self.handles.append(src.register_forward_hook(self._keep(self.attn, i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _embed(self, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        self.embed = h[0].detach().to("cpu", torch.bfloat16)

    @staticmethod
    def _keep(store, i):
        def hook(module, args, out):
            store[i] = (out[0] if isinstance(out, tuple) else out)[0].detach().to("cpu", torch.bfloat16)
        return hook


@torch.no_grad()
def capture_row(lm, prompt_ids, output_ids):
    """One teacher-forced pass; returns the tensors saved for a row. Caller holds the model lock."""
    a = lm.arch
    ids = prompt_ids + output_ids
    x = torch.tensor([ids], device=lm.device)
    with _Hooks(a) as h:
        logits = lm.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0]
    top = torch.topk(logits.float(), TOP, dim=-1)
    top_ids = top.indices.to("cpu", torch.int32)
    ids_t = torch.tensor(ids, dtype=torch.int32)
    match = torch.full((len(ids),), -1, dtype=torch.int8)
    match[:-1] = (top_ids[:-1, 0] == ids_t[1:]).to(torch.int8)
    scalar = torch.tensor([float(s.float().flatten()[0]) if s is not None else 1.0 for s in a.scalar])
    return {
        "token_ids": ids_t,
        "resid": torch.stack([h.embed] + h.out).contiguous(),
        "attn_out": torch.stack(h.attn).contiguous(),
        "scalar": scalar,
        "top_ids": top_ids.contiguous(),
        "top_logits": top.values.cpu().contiguous(),
        "match": match,
    }


def save_row(name, index, row, tensors, n_prompt):
    path = row_file(name, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = tensors["token_ids"].tolist()
    meta = {"row_id": str(row.get("id", index)), "model": row.get("model", ""), "n_prompt": str(n_prompt),
            "ids_sha": ids_sha(ids)}
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(path)
    return path.stat().st_size


def revision(model_id):
    """Commit hash of the cached weights (the snapshot folder name), so data stays tied to them."""
    path = try_to_load_from_cache(model_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def manifest(name, lm):
    _, rows = dataset.read(name)
    a = lm.arch
    files = []
    for i, r in enumerate(rows, 1):
        if is_fresh(name, i, r):
            meta = file_meta(row_file(name, i))
            n = len(json.loads(r["prompt_token_ids"])) + len(json.loads(r["output_token_ids"]))
            files.append({"file": f"rows/{i:05d}.safetensors", "row": i, "id": r.get("id"),
                          "n_prompt": int(meta["n_prompt"]), "n_tokens": n})
    return {
        "format": FORMAT, "version": VERSION, "dataset": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": lm.id, "revision": revision(lm.id),
        "architecture": type(lm.model).__name__, "dtype": "bfloat16", "device": str(lm.device),
        "layers": a.n_layers, "hidden_size": a.hidden_size, "vocab_size": a.vocab_size,
        "layer_kind": a.layer_kind, "kv_source": a.kv_source, "softcap": a.softcap,
        "identity": "resid[i+1] = (resid[i] + attn_out[i] + mlp_out[i] + ple_out[i]) * scalar[i]",
        "tensors": {
            "token_ids": "[T] int32: prompt (chat template applied) + generated ids; first n_prompt are the prompt",
            "resid": "[L+1, T, H] bf16: resid[0] = embedding (input to layer 0), resid[i+1] = output of layer i",
            "attn_out": "[L, T, H] bf16: what attention adds to the residual at layer i (after its post-norm)",
            "scalar": "[L] float32: per-layer output multiplier (1 where the model has none)",
            "top_ids": f"[T, {TOP}] int32: most likely next tokens at each position",
            "top_logits": f"[T, {TOP}] float32: their logits (after soft-capping)",
            "match": "[T] int8: top_ids[t,0] == token_ids[t+1] (1/0), -1 at the last position",
        },
        "capture": "one teacher-forced forward pass per row (no KV cache), hooks on every decoder layer",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        "rows": files,
    }


class _Pipe:
    """File-like sink that zipfile writes into; the bytes come out of chunks() as they're made."""

    CHUNK = 1 << 20

    def __init__(self):
        self.q = queue.Queue(maxsize=16)
        self.buf = bytearray()
        self.aborted = threading.Event()

    def write(self, b):
        if self.aborted.is_set():
            raise BrokenPipeError("download cancelled")
        self.buf += b
        if len(self.buf) >= self.CHUNK:
            self._put(bytes(self.buf))
            self.buf.clear()
        return len(b)

    def flush(self):
        pass

    def close_stream(self):
        if self.buf:
            self._put(bytes(self.buf))
            self.buf.clear()
        self._put(None)

    def _put(self, item):
        while not self.aborted.is_set():
            try:
                self.q.put(item, timeout=1)
                return
            except queue.Full:
                pass


def download_zip(name, man):
    """Start writing the zip on a thread; returns the pipe whose queue yields its bytes (None at the end).
    Stored, not deflated: bf16 activations barely compress and this keeps it fast."""
    pipe = _Pipe()

    def run():
        try:
            with zipfile.ZipFile(pipe, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                zf.write(dataset.path(name), "data.csv")
                zf.writestr("manifest.json", json.dumps(man, indent=2))
                for f in man["rows"]:
                    zf.write(DIR / name / f["file"], f["file"])
        except BrokenPipeError:
            return
        finally:
            pipe.close_stream()

    threading.Thread(target=run, daemon=True).start()
    return pipe


def delete(name):
    shutil.rmtree(DIR / name, ignore_errors=True)
