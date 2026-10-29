"""Load a supported Hugging Face decoder-only LLM and locate its parts generically.

Supported layouts (see compat.py and docs/supported-models.md for the exact list):

    plain pre-norm (Llama, Qwen, ...)
        embed -> N x [ x += attn(norm(x));  x += mlp(norm(x)) ] -> norm -> lm_head
    Gemma 4 E2B / E4B
        embed -> N x [ x += post_norm(attn(norm(x)));  x += post_norm(mlp(norm(x)));
                       x += norm(proj(act(gate(x)) * per_layer_input));  x *= layer_scalar ]
              -> norm -> lm_head -> soft-cap
        plus image and audio encoders whose outputs fill placeholder token slots

Rather than hardcoding `model.model.layers[i].mlp.down_proj`, `Arch` finds each piece by
structure (the ModuleList with num_hidden_layers entries, the child named like attention, the
MLP's projection back to hidden size, ...), so the tracer works on every supported family.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_ATTN = re.compile(r"(self_)?attn|attention", re.I)
_MLP = re.compile(r"mlp|feed_forward|ffn", re.I)
_DOWN = ["down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "out_proj"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Arch:
    decoder: nn.Module  # the text decoder (whose attention implementation the tracer switches)
    embed: nn.Module
    layers: nn.ModuleList
    attn: list
    mlp: list
    down: list          # MLP projection back to hidden size
    final_norm: nn.Module
    lm_head: nn.Module
    n_layers: int
    hidden_size: int
    intermediate_size: int
    heads: int
    kv_heads: int
    vocab_size: int
    # Optional parts of richer layouts (None / empty when the layer is plain pre-norm):
    attn_post: list     # norm applied to attention output before it's added (sandwich norm)
    mlp_post: list      # norm applied to MLP output before it's added
    ple_mix: list       # per-layer-input sublayer: projection whose input is act(gate(x)) * per_layer_input
    ple_out: list       # ... and the norm whose output is added to the residual
    scalar: list        # per-layer output multiplier (tensor) or None
    softcap: float | None  # final logit soft-capping: cap * tanh(logits / cap)
    kv_source: list     # for KV-shared layers, the layer whose keys/values they reuse; else None
    layer_kind: list    # "full" / "sliding" attention per layer (or None)
    encoders: dict      # {"image": Encoder, "audio": Encoder} for multimodal models

    def logits(self, h):
        """What the model's head does to a final residual vector (norm, lm_head, soft-cap)."""
        x = self.lm_head(self.final_norm(h)).float()
        return self.softcap * torch.tanh(x / self.softcap) if self.softcap else x


@dataclass
class Encoder:
    """A non-text input pipeline: tower (its own layer stack) -> embedder (into the LLM's space)."""
    tower: nn.Module
    layers: nn.ModuleList
    embedder: nn.Module
    token_id: int       # placeholder token whose slots the soft tokens fill


def _child(module, pattern):
    for name, child in module.named_children():
        if pattern.search(name) and "norm" not in name.lower():
            return child
    return None


def _decoder(model):
    """The text decoder: the LLM itself, or the language model inside a multimodal wrapper."""
    base = getattr(model, "model", model)
    return getattr(base, "language_model", None) or base


def _encoders(model, cfg):
    base = getattr(model, "model", model)
    found = {}
    for kind, tower, embedder, token in [("image", "vision_tower", "embed_vision", "image_token_id"),
                                         ("audio", "audio_tower", "embed_audio", "audio_token_id")]:
        t, e = getattr(base, tower, None), getattr(base, embedder, None)
        if t is None or e is None:
            continue
        stacks = [m for m in t.modules() if isinstance(m, nn.ModuleList)]
        layers = max(stacks, key=len) if stacks else None
        found[kind] = Encoder(tower=t, layers=layers, embedder=e, token_id=getattr(cfg, token))
    return found


def discover(model):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    decoder = _decoder(model)

    layers_name, layers = next(((name, m) for name, m in decoder.named_modules()
                                if isinstance(m, nn.ModuleList) and len(m) == n), (None, None))
    if layers is None:
        raise ValueError(f"no ModuleList of {n} decoder layers found")

    attn, mlp, down = [], [], []
    attn_post, mlp_post, ple_mix, ple_out, scalar, kv_source, layer_kind = [], [], [], [], [], [], []
    for i, layer in enumerate(layers):
        a, m = _child(layer, _ATTN), _child(layer, _MLP)
        if a is None or m is None:
            raise ValueError(f"layer {type(layer).__name__} has no attention/MLP child")
        attn.append(a)
        mlp.append(m)
        d = next((getattr(m, name) for name in _DOWN if isinstance(getattr(m, name, None), nn.Linear)), None)
        if d is None:
            raise ValueError(f"{type(m).__name__} has no linear down-projection (MoE or Conv1D MLP)")
        down.append(d)

        # Sandwich norms: a pre- *and* post-norm around the MLP means post_attention_layernorm is
        # the attention's post-norm (Gemma 2/3/4), not the MLP's pre-norm (Llama, Qwen).
        sandwich = hasattr(layer, "pre_feedforward_layernorm") and hasattr(layer, "post_feedforward_layernorm")
        attn_post.append(layer.post_attention_layernorm if sandwich else None)
        mlp_post.append(layer.post_feedforward_layernorm if sandwich else None)
        has_ple = hasattr(layer, "per_layer_input_gate")
        ple_mix.append(layer.per_layer_projection if has_ple else None)
        ple_out.append(layer.post_per_layer_input_norm if has_ple else None)
        s = getattr(layer, "layer_scalar", None)
        scalar.append(s if torch.is_tensor(s) else None)
        kind = getattr(a, "layer_type", None)
        layer_kind.append({"full_attention": "full", "sliding_attention": "sliding"}.get(kind))
        kv_source.append(None)
    for i, a in enumerate(attn):
        if getattr(a, "is_kv_shared_layer", False):
            # Reuses the keys/values of the last non-shared layer of the same attention kind.
            kv_source[i] = max(j for j in range(i) if not getattr(attn[j], "is_kv_shared_layer", False)
                               and layer_kind[j] == layer_kind[i])

    final_norm = getattr(decoder, "norm", None)
    if final_norm is None:  # the last norm-like module that isn't inside a decoder layer
        for name, mod in decoder.named_modules():
            if name.startswith(layers_name + ".") or mod is decoder:
                continue
            if "norm" in type(mod).__name__.lower() and not list(mod.children()):
                final_norm = mod
    if final_norm is None:
        raise ValueError("no final norm found")

    heads = cfg.num_attention_heads
    intermediate = getattr(cfg, "intermediate_size", None) or down[0].in_features
    return Arch(
        decoder=decoder,
        embed=model.get_input_embeddings(),
        layers=layers, attn=attn, mlp=mlp, down=down,
        final_norm=final_norm, lm_head=model.get_output_embeddings(),
        n_layers=n, hidden_size=hidden, intermediate_size=intermediate,
        heads=heads, kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
        vocab_size=cfg.vocab_size,
        attn_post=attn_post, mlp_post=mlp_post, ple_mix=ple_mix, ple_out=ple_out, scalar=scalar,
        softcap=getattr(cfg, "final_logit_softcapping", None),
        kv_source=kv_source, layer_kind=layer_kind,
        encoders=_encoders(model, model.config),
    )


class LoadedModel:
    def __init__(self, model_id, device=None):
        self.id = model_id
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        self.arch = discover(self.model)
        # Multimodal models turn images/audio into pixel/feature tensors + placeholder tokens.
        self.processor = AutoProcessor.from_pretrained(model_id) if self.arch.encoders else None
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)

    @property
    def modalities(self):
        return ["text", *self.arch.encoders]

    def info(self):
        a = self.arch
        return {
            "model": self.id,
            "architecture": type(self.model).__name__,
            "device": self.device,
            "dtype": str(self.model.dtype).replace("torch.", ""),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "layers": a.n_layers,
            "hidden_size": a.hidden_size,
            "intermediate_size": a.intermediate_size,
            "attention_heads": a.heads,
            "kv_heads": a.kv_heads,
            "vocab_size": a.vocab_size,
            "modalities": self.modalities,
        }

    def unload(self):
        del self.model
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


def cached_models():
    """Causal LMs already in the local Hugging Face cache that transformers can load."""
    from huggingface_hub import scan_cache_dir

    found = []
    for repo in scan_cache_dir().repos:
        if repo.repo_type != "model" or "mlx" in repo.repo_id.lower():
            continue
        rev = max(repo.revisions, key=lambda r: r.last_modified)
        snapshot = Path(rev.snapshot_path)
        cfg_file = snapshot / "config.json"
        has_weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
        if not cfg_file.exists() or not has_weights:  # e.g. only config.json, from a pre-download check
            continue
        cfg = json.loads(cfg_file.read_text())
        archs = cfg.get("architectures") or []
        if any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs):
            found.append({"id": repo.repo_id, "size_gb": round(repo.size_on_disk / 1e9, 2),
                          "architecture": archs[0], "config": cfg})
    return sorted(found, key=lambda m: m["id"])
