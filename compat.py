"""Is a model supported? Yes or no. Human-readable list: docs/supported-models.md

Supported = a dense decoder whose every simulator panel is exact. Two layouts qualify:

    plain pre-norm   embed -> N x [ x += attn(norm(x));  x += mlp(norm(x)) ] -> norm -> lm_head
    Gemma 4 E2B/E4B  the same with post-norms, a per-layer-input sublayer, a layer scalar and a
                     soft-capped head, plus image and audio encoders (see models.py)

Two gates:

    precheck(repo_id)   before downloading: config.json + size from the Hub. Rejects known
                        unsupported kinds (MoE, non-attention layers, quantized, multimodal, ...).
    self_test(lm)       after loading: one tiny forward pass with the tracer's hooks, proving the
                        layout above holds for this exact model. Any failure -> not supported.

Each check is {"label", "ok": bool, "detail"}. A model is supported only if every check is ok.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from tracer import Capture, EncoderCapture

# Families whose layout is exactly the one above. Anything else is not supported.
SUPPORTED = {
    "qwen2": "Qwen2 / Qwen2.5",
    "qwen3": "Qwen3 (dense)",
    "llama": "Llama 2 / 3.x, SmolLM2",
    "mistral": "Mistral (dense)",
    "phi3": "Phi-3 / Phi-3.5 / Phi-4-mini",
    "gemma": "Gemma (1st generation)",
    "smollm3": "SmolLM3",
    "olmo": "OLMo (1st generation)",
    "gemma4": "Gemma 4 E2B / E4B: text, image and audio",
}
# Supported families that ship as a multimodal *ForConditionalGeneration checkpoint.
MULTIMODAL = {"gemma4"}

# Why other families are out, for a clear rejection message.
REASONS = {
    **dict.fromkeys(["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3",
                     "gpt_oss", "granitemoe", "phimoe", "dbrx", "jetmoe", "llama4"],
                    "mixture-of-experts MLPs"),
    **dict.fromkeys(["mamba", "mamba2", "falcon_mamba", "rwkv", "jamba", "zamba", "zamba2",
                     "falcon_h1", "bamba", "qwen3_next", "lfm2", "recurrent_gemma", "nemotron_h",
                     "granitemoehybrid", "xlstm", "hymba", "minimax"],
                    "layers that aren't attention (state-space / recurrent / linear attention)"),
    **dict.fromkeys(["phi", "gptj", "gpt_neox", "falcon", "cohere", "cohere2", "codegen"],
                    "attention and MLP running in parallel"),
    **dict.fromkeys(["gpt2", "gpt_bigcode"], "Conv1D layers instead of linear projections"),
    **dict.fromkeys(["granite"], "scaled residuals and logits"),
}
# Share of system memory the weights may take: fits an 8B model in bf16 (~16 GB) on 24 GB.
MEMORY_SHARE = 0.65
ENCODERS = {"bert", "roberta", "deberta", "deberta-v2", "electra", "distilbert", "modernbert", "xlm-roberta"}


def check(label, ok, detail):
    return {"label": label, "ok": bool(ok), "detail": detail}


def report(model_id, checks):
    return {"model": model_id, "supported": all(c["ok"] for c in checks), "checks": checks}


def system_memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def config_checks(cfg, n_params=None):
    """Decide from config.json (dict) alone, without loading weights."""
    archs = cfg.get("architectures") or []
    mtype = cfg.get("model_type", "?")
    text = cfg.get("text_config") or cfg

    if cfg.get("is_encoder_decoder") or mtype in ENCODERS:
        return [check("Model kind", False, f"{mtype} is an encoder / encoder-decoder model. Only decoder-only LLMs are supported.")]
    if any(a.endswith("ForConditionalGeneration") for a in archs):
        if mtype not in MULTIMODAL:
            return [check("Model kind", False, f"{archs[0]} is multimodal, and {mtype} isn't a supported multimodal family.")]
        kinds = [k for k, key in [("image", "vision_config"), ("audio", "audio_config")] if cfg.get(key)]
        out = [check("Model kind", True, f"Multimodal decoder ({archs[0]}): text{''.join(', ' + k for k in kinds)}.")]
    elif any(a.endswith("ForCausalLM") for a in archs):
        out = [check("Model kind", True, f"Decoder-only causal LM ({archs[0]}).")]
    else:
        return [check("Model kind", False, f"{archs[0] if archs else mtype} is not a causal LM.")]

    if text.get("enable_moe_block") or text.get("num_experts"):
        out.append(check("Architecture", False, f"{mtype} with mixture-of-experts MLPs (e.g. Gemma 4 26B-A4B)."))
    elif mtype in SUPPORTED:
        out.append(check("Architecture", True, f"{SUPPORTED[mtype]}."))
    elif mtype in REASONS:
        out.append(check("Architecture", False, f"{mtype} uses {REASONS[mtype]}."))
    else:
        out.append(check("Architecture", False, f"{mtype} isn't on the supported list."))

    q = cfg.get("quantization_config")
    if q:
        out.append(check("Weights", False, f"Pre-quantized checkpoint ({q.get('quant_method', 'unknown')}). Use the original bf16/fp16 repo."))

    if n_params:
        need, mem = n_params * 2 / 1e9, system_memory_gb()
        fits = mem is None or need <= mem * MEMORY_SHARE
        out.append(check("Memory", fits, f"{n_params / 1e9:.2f}B params ≈ {need:.1f} GB in bf16"
                         + (f"; limit is {MEMORY_SHARE:.0%} of {mem:.0f} GB memory." if mem else ".")))
    return out


def precheck(repo_id):
    """Check a Hub repo before downloading it (fetches only config.json and file metadata)."""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        cached = try_to_load_from_cache(repo_id, "config.json")
        path = cached if isinstance(cached, str) else hf_hub_download(repo_id, "config.json")
        cfg = json.loads(open(path).read())
    except GatedRepoError:
        return report(repo_id, [check("Access", False,
                                      "Gated repo: accept its license on huggingface.co, then run `huggingface-cli login` on this machine.")])
    except RepositoryNotFoundError:
        return report(repo_id, [check("Access", False, "No such model on the Hugging Face Hub.")])
    except Exception as e:
        return report(repo_id, [check("Access", False, f"Couldn't read config.json: {e}")])
    n_params = None
    try:
        info = HfApi().model_info(repo_id)
        n_params = info.safetensors.total if info.safetensors else None
    except Exception:
        pass
    return report(repo_id, config_checks(cfg, n_params))


@torch.no_grad()
def self_test(lm):
    """Prove, on this exact model, that every simulator panel will be exact."""
    a, model, tok = lm.arch, lm.model, lm.tok
    checks = [c for c in config_checks(model.config.to_dict()) if c["label"] == "Architecture"]
    checks.append(check("Parts found", True,
                        f"{a.n_layers} × {type(a.layers[0]).__name__}: {type(a.attn[0]).__name__}, "
                        f"{type(a.mlp[0]).__name__}, final {type(a.final_norm).__name__}."))

    # A short probe through every input path the model has. The probe image and tone only test
    # the plumbing; they're never shown in the simulator.
    inputs = {}
    if a.encoders:
        content = []
        if "image" in a.encoders:
            g = np.linspace(0, 255, 96, dtype=np.uint8)
            content.append({"type": "image", "image": Image.fromarray(np.stack([*np.meshgrid(g, g[::-1]), g[None].repeat(96, 0)], -1))})
        if "audio" in a.encoders:
            content.append({"type": "audio", "audio": (0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)})
        content.append({"type": "text", "text": "The capital of France is"})
        inputs = lm.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                  tokenize=True, return_dict=True, return_tensors="pt")
        ids = inputs.pop("input_ids").to(lm.device)
        inputs.pop("attention_mask", None)
        inputs = {k: v.to(lm.device) for k, v in inputs.items() if torch.is_tensor(v)}
    else:
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(lm.device)

    prev = a.decoder.config._attn_implementation
    a.decoder.set_attn_implementation("eager")
    try:
        with Capture(a) as cap, EncoderCapture(a.encoders) as enc:
            out = model(input_ids=ids, **inputs)
    finally:
        a.decoder.set_attn_implementation(prev)

    def rel(x, y):
        return float((x.float() - y.float()).norm() / (y.float().norm() + 1e-6))

    checks.append(check("Attention weights", all(w is not None for w in cap.attn_w),
                        f"Every layer exposes all {a.heads} heads' weights."))
    checks.append(check("MLP neurons", all(x is not None for x in cap.mlp_act),
                        f"Every layer's {a.intermediate_size:,}+ intermediate neurons are captured."))

    terms = ["input", "attention", "MLP"]
    if a.ple_mix[0] is not None:
        terms.append("per-layer input")
    errs = []
    for i in range(a.n_layers):
        total = cap.resid_in[i] + cap.attn_out[i] + cap.mlp_out[i]
        if cap.ple_out[i] is not None:
            total = total + cap.ple_out[i]
        if a.scalar[i] is not None:
            total = total * a.scalar[i].float().flatten()[0]
        errs.append(rel(total, cap.resid_out[i]))
    rule = f"({' + '.join(terms)}) × layer scalar" if a.scalar[0] is not None else " + ".join(terms)
    post = " (attention and MLP measured after their post-norms)" if a.attn_post[0] is not None else ""
    checks.append(check("Residual stream", max(errs) < 0.02,
                        f"Every layer: output = {rule}{post}; max error {max(errs):.2%}."))

    lens = a.logits(cap.resid_out[-1][-1])
    real = out.logits[0, -1].float()
    head = "final norm + lm_head" + (f" + soft-cap {a.softcap:g}" if a.softcap else "")
    checks.append(check("Logit lens", int(lens.argmax()) == int(real.argmax()) and rel(lens, real) < 0.02,
                        f"{head} on the last layer reproduces the real logits (error {rel(lens, real):.2%})."))

    for kind, e in a.encoders.items():
        st = enc.stacks[kind]
        slots = int((ids == e.token_id).sum())
        captured = st["input"] is not None and all(x is not None for x in st["layers"])
        checks.append(check(f"{kind.capitalize()} input", captured and slots > 0,
                            f"{type(e.tower).__name__}: {len(e.layers)} layers captured; "
                            f"its output fills {slots} placeholder token slots in the sequence."))

    template = tok.chat_template or (lm.processor.chat_template if lm.processor else None)
    checks.append(check("Chat template", bool(template), "The tokenizer can format conversations."))

    result = report(lm.id, checks)
    result["tools"] = "tools" in (template or "")  # a chat feature, not a support criterion
    result["modalities"] = lm.modalities
    return result
