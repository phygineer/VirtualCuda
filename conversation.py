"""OpenAI-format conversations -> model inputs, and model outputs -> OpenAI-format replies.

Media: OpenAI content parts carry images as `image_url` (data: URLs) and audio as `input_audio`
(base64 WAV). We store each file once under data/media/<sha>.<ext> and replace the inline data
with its /media/ URL, so recordings stay small and replays can re-run the exact same pixels and
samples through the encoders.

Tool calls are parsed from the raw completion (special tokens kept) in three formats:
    Qwen / Hermes   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    Gemma 4         <|tool_call>call:NAME{key:<|"|>value<|"|>, n: 3}<tool_call|>
    Llama 3         a bare {"name": ..., "parameters": {...}} reply
"""

import base64
import hashlib
import io
import json
import re
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MEDIA_DIR = Path(__file__).parent / "data" / "media"
MEDIA_NAME = re.compile(r"^[0-9a-f]{16}\.(png|jpg|jpeg|webp|gif|wav)$")
SAMPLE_RATE = 16000
TOOL_STARTS = ["<tool_call>", "<|tool_call>"]


class BadRequest(ValueError):
    pass


# ---------------------------------------------------------------- media files

def save_media(data: bytes, ext: str) -> str:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{hashlib.sha256(data).hexdigest()[:16]}.{ext}"
    path = MEDIA_DIR / name
    if not path.exists():
        path.write_bytes(data)
    return f"/media/{name}"


def media_file(url: str) -> Path:
    name = url.rsplit("/", 1)[-1]
    if not url.startswith("/media/") or not MEDIA_NAME.match(name) or not (MEDIA_DIR / name).exists():
        raise BadRequest(f"unknown media {url}")
    return MEDIA_DIR / name


def _decode_data_url(url: str):
    m = re.match(r"^data:([\w/+.-]+);base64,(.*)$", url, re.S)
    if not m:
        raise BadRequest("images must be data: URLs (base64) or /media/ paths from this server")
    mime, data = m.groups()
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(mime)
    if not ext:
        raise BadRequest(f"unsupported image type {mime}")
    return base64.b64decode(data), ext


def normalize(messages: list[dict]) -> list[dict]:
    """Store inline media and return the same OpenAI messages with /media/ URLs in their place."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                t = p.get("type")
                if t == "image_url":
                    url = p["image_url"]["url"] if isinstance(p.get("image_url"), dict) else p.get("image_url")
                    if not url.startswith("/media/"):
                        url = save_media(*_decode_data_url(url))
                    media_file(url)
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                elif t == "input_audio":
                    a = p["input_audio"]
                    url = a.get("url") or save_media(base64.b64decode(a["data"]), a.get("format", "wav"))
                    if not url.endswith(".wav"):
                        raise BadRequest("audio must be WAV (the chat page converts recordings to 16 kHz WAV)")
                    media_file(url)
                    parts.append({"type": "input_audio", "input_audio": {"url": url, "format": "wav"}})
                elif t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
            m["content"] = parts
        out.append(m)
    return out


def load_audio(path: Path) -> np.ndarray:
    """WAV file -> mono float32 at 16 kHz (linear resampling if it was recorded at another rate)."""
    with wave.open(str(path)) as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise BadRequest(f"unsupported WAV sample width {width}")
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        t = np.arange(0, len(x) * SAMPLE_RATE / sr) * sr / SAMPLE_RATE
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


# ---------------------------------------------------------------- encoding

def has_media(messages) -> list[str]:
    kinds = []
    for m in messages:
        content = m.get("content")
        for p in content if isinstance(content, list) else []:
            if p["type"] == "image_url":
                kinds.append("image")
            elif p["type"] == "input_audio":
                kinds.append("audio")
    return kinds


def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def template_messages(messages, multimodal):
    """OpenAI messages -> what HF chat templates expect: tool arguments as dicts, tool results
    named after their call, and (for multimodal models) images/audio as loaded objects."""
    from PIL import Image

    names = {}
    out = []
    for m in messages:
        if multimodal and isinstance(m.get("content"), list):
            content = []
            for p in m["content"]:
                if p["type"] == "text":
                    content.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image_url":
                    content.append({"type": "image", "image": Image.open(media_file(p["image_url"]["url"])).convert("RGB")})
                elif p["type"] == "input_audio":
                    content.append({"type": "audio", "audio": load_audio(media_file(p["input_audio"]["url"]))})
        else:
            content = _text_of(m.get("content"))
        msg = {"role": m["role"], "content": content}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                names[c.get("id")] = fn.get("name")
                calls.append({**c, "function": fn})
            msg["tool_calls"] = calls
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        name = m.get("name") or names.get(m.get("tool_call_id"))
        if name:
            msg["name"] = name
        out.append(msg)
    return out


@dataclass
class Encoded:
    prompt_ids: list[int]
    model_inputs: dict = field(default_factory=dict)   # extra tensors: pixel_values, input_features, ...
    media: dict = field(default_factory=dict)          # {"image": [urls], "audio": [urls]} in prompt order


def encode(lm, messages, tools=None) -> Encoded:
    """Normalized OpenAI messages -> token ids (+ processor tensors for images/audio)."""
    kinds = has_media(messages)
    if kinds and lm.processor is None:
        raise BadRequest(f"{lm.id} is text-only; it can't take {', '.join(sorted(set(kinds)))}")
    for k in set(kinds):
        if k not in lm.modalities:
            raise BadRequest(f"{lm.id} doesn't accept {k}")
    if lm.processor is None:
        ids = lm.tok.apply_chat_template(template_messages(messages, False), tools=tools or None,
                                         add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        return Encoded(prompt_ids=list(ids))

    inputs = lm.processor.apply_chat_template(template_messages(messages, True), tools=tools or None,
                                              add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt")
    ids = inputs.pop("input_ids")[0].tolist()
    inputs.pop("attention_mask", None)
    media = {"image": [], "audio": []}
    for m in messages:
        for p in m.get("content") if isinstance(m.get("content"), list) else []:
            if p["type"] == "image_url":
                media["image"].append(p["image_url"]["url"])
            elif p["type"] == "input_audio":
                media["audio"].append(p["input_audio"]["url"])
    extra = {k: v for k, v in inputs.items() if torch.is_tensor(v)} if kinds else {}
    return Encoded(prompt_ids=ids, model_inputs=extra, media=media)


# ---------------------------------------------------------------- outputs

def special_pattern(tok):
    specials = sorted({s for s in tok.all_special_tokens if s}, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in specials)) if specials else None


def strip_special(text, pattern):
    return pattern.sub("", text) if pattern else text


def _gemma_args(body):
    """Gemma 4's argument syntax -> dict. Strings are wrapped in <|"|>, keys are bare."""
    parts = body.split('<|"|>')
    js = []
    for i, part in enumerate(parts):
        if i % 2:  # inside a string
            js.append(json.dumps(part))
        else:
            js.append(re.sub(r'([A-Za-z_][\w]*)\s*:', r'"\1":', part))
    return json.loads("{" + "".join(js) + "}")


def _call(name, args):
    return {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def parse_tool_calls(raw, pattern):
    """Model output with special tokens kept -> (visible content, OpenAI tool_calls)."""
    starts = [raw.index(s) for s in TOOL_STARTS if s in raw]
    calls = []
    for name, body in re.findall(r"<\|tool_call>call:([\w.-]+)\{(.*?)\}<tool_call\|>", raw, re.S):
        try:
            calls.append(_call(name, _gemma_args(body)))
        except (json.JSONDecodeError, ValueError):
            pass
    for body in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", raw, re.S):
        try:
            obj = json.loads(body.strip())
            if isinstance(obj, dict) and "name" in obj:
                calls.append(_call(obj["name"], obj.get("arguments", obj.get("parameters", {}))))
        except json.JSONDecodeError:
            pass
    if calls:
        return strip_special(raw[:min(starts)], pattern).strip(), calls

    text = strip_special(raw, pattern)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "name" in obj:
                return "", [_call(obj["name"], obj.get("arguments", obj.get("parameters", {})))]
        except json.JSONDecodeError:
            pass
    return text, []


def held_back(text):
    """How much of the tail might be the start of a tool-call marker (don't stream it yet)."""
    for marker in TOOL_STARTS:
        for k in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:k]):
                return k
    return 0
