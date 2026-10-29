"""Token-by-token generation that records everything each layer does, for the simulation page.

Each step is one forward pass (the whole prompt on step 0, then one token at a time with the
KV cache). Hooks on every decoder layer capture, for the last position of that pass:

    resid_in    residual stream entering the layer                     [hidden]
    attn_raw    attention output (before any post-norm)                [hidden]
    attn_out    what attention adds to the residual (after post-norm)  [hidden]
    attn        attention weights of each head over the keys it sees   [heads, keys]
    mlp_act     MLP intermediate (input to down-proj)                  [intermediate] -> binned + top neurons
    mlp_raw     MLP output (before any post-norm)                      [hidden]
    mlp_out     what the MLP adds to the residual                      [hidden]
    ple_*       Gemma 4 per-layer input: its embedding slice, the gated mix, and what it adds
    scalar      layer output multiplier
    lens        logit lens: the model's own head applied to this layer's output, top 3

so that, exactly:  resid_out = (resid_in + attn_out + mlp_out + ple_out) * scalar

On step 0, multimodal inputs are also traced through their encoders (see media_summary):
the pixels / mel spectrogram the model actually received, every encoder layer, and the soft
tokens that fill the image/audio placeholder slots in the sequence.

Two modes:
    live     sample each next token (temperature / top_p), like normal generation
    replay   teacher-forced: feed the token ids (and media) recorded from a real chat (store.py).
             The forward pass is a pure function of them, so this reproduces that chat's activations.

Vectors are sent as base64 little-endian float16 to keep each step a few hundred KB.
Layer parts are located generically via models.Arch, so every supported family works (compat.py).
"""

import base64
import io

import numpy as np
import torch

MLP_BIN = 32  # intermediate neurons per bin in the MLP activation strip
LENS_TOP = 3
OUT_TOP = 10


def f16(t):
    return base64.b64encode(t.detach().float().cpu().numpy().astype("<f2").tobytes()).decode()


def u8(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def _first(x):
    return x[0] if isinstance(x, tuple) else x


class Capture:
    """Forward hooks on every decoder layer; holds batch-0 activations from the last forward."""

    def __init__(self, arch):
        self.arch = arch
        n = arch.n_layers
        for name in ["resid_in", "attn_raw", "attn_out", "attn_w", "mlp_act", "mlp_raw", "mlp_out",
                     "ple_in", "ple_mix", "ple_out", "resid_out"]:
            setattr(self, name, [None] * n)
        self.handles = []

    def __enter__(self):
        a = self.arch
        for i in range(a.n_layers):
            self.handles += [
                a.layers[i].register_forward_pre_hook(self._store_in(i), with_kwargs=True),
                a.attn[i].register_forward_hook(self._store_attn(i)),
                a.mlp[i].register_forward_hook(self._store("mlp_raw", i)),
                a.down[i].register_forward_pre_hook(self._store_input("mlp_act", i)),
                a.layers[i].register_forward_hook(self._store("resid_out", i)),
            ]
            if a.attn_post[i] is not None:
                self.handles.append(a.attn_post[i].register_forward_hook(self._store("attn_out", i)))
            if a.mlp_post[i] is not None:
                self.handles.append(a.mlp_post[i].register_forward_hook(self._store("mlp_out", i)))
            if a.ple_mix[i] is not None:
                self.handles += [
                    a.ple_mix[i].register_forward_pre_hook(self._store_input("ple_mix", i)),
                    a.ple_out[i].register_forward_hook(self._store("ple_out", i)),
                ]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        a = self.arch
        for i in range(a.n_layers):  # plain pre-norm: what's added is the raw sublayer output
            if a.attn_post[i] is None:
                self.attn_out[i] = self.attn_raw[i]
            if a.mlp_post[i] is None:
                self.mlp_out[i] = self.mlp_raw[i]

    def _store_in(self, i):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            self.resid_in[i] = h[0].detach()
            ple = args[1] if len(args) > 1 else kwargs.get("per_layer_input")
            self.ple_in[i] = ple[0].detach() if torch.is_tensor(ple) else None
        return hook

    def _store_attn(self, i):
        def hook(module, args, out):
            self.attn_raw[i] = _first(out)[0].detach()
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            self.attn_w[i] = w[0].detach() if torch.is_tensor(w) else None
        return hook

    def _store(self, name, i):
        def hook(module, args, out):
            getattr(self, name)[i] = _first(out)[0].detach()
        return hook

    def _store_input(self, name, i):
        def hook(module, args):
            getattr(self, name)[i] = args[0][0].detach()
        return hook


class EncoderCapture:
    """Hooks on each encoder's layer stack: input to the first layer and every layer's output."""

    def __init__(self, encoders):
        self.encoders = encoders
        self.stacks = {kind: {"input": None, "layers": [None] * len(e.layers)} for kind, e in encoders.items()}
        self.handles = []

    def __enter__(self):
        for kind, e in self.encoders.items():
            s = self.stacks[kind]

            def pre(module, args, kwargs, s=s):
                s["input"] = (args[0] if args else kwargs["hidden_states"]).detach()

            self.handles.append(e.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
            for j, layer in enumerate(e.layers):
                def post(module, args, out, s=s, j=j):
                    s["layers"][j] = _first(out).detach()
                self.handles.append(layer.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def pca_rgb(x):
    """Real features -> colors: project [n, d] onto its top 3 principal components, one per RGB channel."""
    x = x.float().cpu()
    x = x - x.mean(0, keepdim=True)
    if x.shape[0] < 4:
        return np.zeros((x.shape[0], 3), np.uint8)
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    p = (x @ v[:, :3]).numpy()
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    return (np.clip((p - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def layer_maps(states):
    """Per encoder layer: per-position norms (float16) and PCA colors (uint8 RGB)."""
    return [{"norm": f16(s.float().norm(dim=-1)), "rgb": u8(pca_rgb(s))} for s in states]


class Tracer:
    def __init__(self, lm):
        self.lm = lm

    def token(self, i):
        tok = self.lm.tok
        i = int(i)
        return {"id": i, "text": tok.decode([i]), "piece": tok.convert_ids_to_tokens(i)}

    def prompt_ids(self, prompt, chat):
        tok = self.lm.tok
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            return tok(text, add_special_tokens=False).input_ids
        return tok(prompt).input_ids

    def run(self, ids, max_tokens, temperature, top_p, stop, forced=None, extra_meta=None,
            model_inputs=None, media_files=None):
        """Generator of trace events. Caller must hold the model lock.

        ids:           prompt token ids
        forced:        if given, replay these ids instead of sampling (max_tokens is ignored)
        model_inputs:  extra step-0 tensors from the processor (pixel_values, input_features, ...)
        media_files:   {"image": [urls], "audio": [urls]} of the files behind them, in prompt order
        """
        lm, a = self.lm, self.lm.arch
        model = lm.model
        if forced is not None:
            max_tokens = len(forced)

        yield {
            "type": "meta",
            "model": lm.id,
            "architecture": type(model).__name__,
            "template_text": lm.tok.decode(ids),
            "tokens": [self.token(i) for i in ids],
            "layers": a.n_layers,
            "hidden_size": a.hidden_size,
            "intermediate_size": a.intermediate_size,
            "heads": a.heads,
            "kv_heads": a.kv_heads,
            "vocab_size": a.vocab_size,
            "mlp_bin": MLP_BIN,
            "temperature": temperature,
            "top_p": top_p,
            "forced": forced is not None,
            "sandwich": a.attn_post[0] is not None,
            "ple": a.ple_mix[0] is not None,
            "softcap": a.softcap,
            "layer_kind": a.layer_kind,
            "kv_source": a.kv_source,
            "media_tokens": {kind: e.token_id for kind, e in a.encoders.items()},
            **(extra_meta or {}),
        }

        # sdpa doesn't return attention weights. Only the text decoder switches: encoders keep theirs.
        prev_impl = a.decoder.config._attn_implementation
        a.decoder.set_attn_implementation("eager")
        try:
            cache = None
            input_ids = torch.tensor([ids])
            pos = 0
            for step in range(max_tokens):
                if stop.is_set():
                    return
                extra = {}
                if step == 0 and model_inputs:
                    extra = {k: v.to(lm.device) for k, v in model_inputs.items()}
                with Capture(a) as cap, EncoderCapture(a.encoders if step == 0 else {}) as enc, torch.no_grad():
                    out = model(input_ids=input_ids.to(lm.device), past_key_values=cache, use_cache=True, **extra)
                cache = out.past_key_values
                force = forced[step] if forced is not None else None
                event, next_id = self.step_event(cap, out.logits[0, -1], step, pos, input_ids[0],
                                                 temperature, top_p, force)
                if step == 0 and model_inputs:
                    event["media"] = self.media_summary(enc, cap, ids, model_inputs, media_files or [])
                yield event
                pos += input_ids.shape[1]
                if forced is None and next_id in lm.eos_ids:
                    return
                input_ids = torch.tensor([[next_id]])
        finally:
            a.decoder.set_attn_implementation(prev_impl)

    @torch.no_grad()
    def step_event(self, cap, logits, step, pos, input_ids, temperature, top_p, forced_id=None):
        a = self.lm.arch
        query_pos = pos + len(input_ids) - 1

        # Logit lens: what would the model predict if it stopped after layer i?
        stacked = torch.stack([cap.resid_out[i][-1] for i in range(a.n_layers)])
        lens_probs = torch.softmax(a.logits(stacked), dim=-1)
        lens_p, lens_i = lens_probs.topk(LENS_TOP, dim=-1)

        layers = []
        for i in range(a.n_layers):
            act = cap.mlp_act[i][-1].float()
            mag = act.abs()
            pad = (-act.numel()) % MLP_BIN
            binned = torch.nn.functional.pad(act, (0, pad)).view(-1, MLP_BIN)
            peak = binned.abs().argmax(dim=1, keepdim=True)
            top_v, top_i = mag.topk(16)
            share = mag.sort(descending=True).values.cumsum(0) / mag.sum()
            w = cap.attn_w[i][:, -1, :]
            layer = {
                "resid_in": f16(cap.resid_in[i][-1]),
                "attn_out": f16(cap.attn_out[i][-1]),
                "mlp_out": f16(cap.mlp_out[i][-1]),
                "attn": f16(w),
                "k0": query_pos + 1 - w.shape[-1],  # absolute position of the first key this layer sees
                "mlp_bins": f16(binned.gather(1, peak).squeeze(1)),  # signed peak per bin
                "mlp_top": [[int(j), round(float(act[j]), 4)] for j in top_i],
                "mlp_k50": int((share < 0.5).sum()) + 1,
                "mlp_k90": int((share < 0.9).sum()) + 1,
                "mlp_n": act.numel(),
                "lens": [[self.token(t)["text"], round(float(p), 4)]
                         for t, p in zip(lens_i[i], lens_p[i])],
            }
            if a.attn_post[i] is not None:
                layer["attn_raw"] = f16(cap.attn_raw[i][-1])
                layer["mlp_raw"] = f16(cap.mlp_raw[i][-1])
            if a.ple_mix[i] is not None:
                layer["ple_in"] = f16(cap.ple_in[i][-1])
                layer["ple_mix"] = f16(cap.ple_mix[i][-1])
                layer["ple_out"] = f16(cap.ple_out[i][-1])
            if a.scalar[i] is not None:
                layer["scalar"] = round(float(a.scalar[i].float().flatten()[0]), 6)
            layers.append(layer)

        final_resid = cap.resid_out[-1][-1]
        normed = a.final_norm(final_resid)

        # Sampling, done by hand so every number on screen is the one actually used.
        logits = logits.float()
        raw_probs = torch.softmax(logits, dim=-1)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_p, sorted_i = probs.sort(descending=True)
            keep = (sorted_p.cumsum(0) - sorted_p) < top_p
            nucleus = int(keep.sum())
            sample_probs = torch.zeros_like(probs)
            sample_probs[sorted_i[keep]] = sorted_p[keep]
            sample_probs /= sample_probs.sum()
            next_id = int(torch.multinomial(sample_probs, 1))
        else:
            nucleus = 1
            next_id = int(logits.argmax())
            sample_probs = torch.zeros_like(raw_probs)
            sample_probs[next_id] = 1.0
        if forced_id is not None:
            next_id = int(forced_id)

        top_l, top_i = logits.topk(OUT_TOP)
        candidates = [{**self.token(t), "logit": round(float(l), 3),
                       "prob": round(float(raw_probs[t]), 5),
                       "sample_prob": round(float(sample_probs[t]), 5)}
                      for t, l in zip(top_i, top_l)]
        rank = int((logits > logits[next_id]).sum()) + 1

        event = {
            "type": "step",
            "step": step,
            "prefill": step == 0,
            "position": query_pos,
            "input_tokens": [self.token(t) for t in input_ids],
            "layers": layers,
            "final_resid": f16(final_resid),
            "final_normed": f16(normed),
            "candidates": candidates,
            "nucleus": nucleus,
            "forced": forced_id is not None,
            "chosen": {**self.token(next_id), "prob": round(float(raw_probs[next_id]), 5),
                       "sample_prob": round(float(sample_probs[next_id]), 5), "rank": rank},
        }
        if step == 0:
            # Residual norm of every prompt position at every layer boundary (input + N outputs).
            bounds = [cap.resid_in[0]] + [cap.resid_out[i] for i in range(a.n_layers)]
            event["pos_norms"] = [[round(float(v), 2) for v in b.float().norm(dim=-1)] for b in bounds]
        return event, next_id

    @torch.no_grad()
    def media_summary(self, enc, cap, ids, inputs, media_files):
        """Everything real about each image / audio clip on its way into the sequence."""
        a = self.lm.arch
        ids_t = torch.tensor(ids)
        embedded = cap.resid_in[0]  # layer-0 input: text embeddings with soft tokens scattered in
        out = []
        files = {k: list(v) for k, v in (media_files or {}).items()}

        def slots(token_id):
            """Consecutive runs of a placeholder token: one run per image / clip."""
            idx = (ids_t == token_id).nonzero().flatten().tolist()
            runs = []
            for p in idx:
                if runs and p == runs[-1][-1] + 1:
                    runs[-1].append(p)
                else:
                    runs.append([p])
            return runs

        def soft(run):
            s = embedded[run[0]:run[-1] + 1]
            return {"start": run[0], "count": len(run), "norm": f16(s.float().norm(dim=-1)), "rgb": u8(pca_rgb(s))}

        if "image" in a.encoders and "pixel_values" in inputs:
            st = enc.stacks["image"]
            pv, pos_ids = inputs["pixel_values"], inputs["image_position_ids"]
            patch = int(round((pv.shape[-1] / 3) ** 0.5))
            for n, run in enumerate(slots(a.encoders["image"].token_id)):
                valid = (pos_ids[n] != -1).all(-1)
                xy = pos_ids[n][valid]
                gx, gy = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
                k = int(round((int(valid.sum()) / len(run)) ** 0.5))
                out.append({
                    "kind": "image", "file": (files.get("image") or [None]).pop(0),
                    "patch": patch, "patches_xy": u8(xy.numpy().astype(np.uint8)) if max(gx, gy) < 256 else None,
                    "patch_grid": [gx, gy], "pool": k, "grid": [gx // k, gy // k],
                    "pixels": self.reconstruct(pv[n][valid], xy, patch, gx, gy),
                    "encoder_input": layer_maps([st["input"][n][valid]])[0],
                    "encoder_layers": layer_maps([s[n][valid] for s in st["layers"]]),
                    "soft": soft(run),
                })
        if "audio" in a.encoders and "input_features" in inputs:
            st = enc.stacks["audio"]
            feats, fmask = inputs["input_features"], inputs["input_features_mask"]
            fe = getattr(self.lm.processor, "feature_extractor", None)
            for n, run in enumerate(slots(a.encoders["audio"].token_id)):
                mel = feats[n][fmask[n].bool()].float().numpy()            # [frames, mel bins]
                lo, hi = np.percentile(mel, 1), np.percentile(mel, 99)
                frames = [s[n] for s in st["layers"]]
                valid = frames[-1].shape[0]
                out.append({
                    "kind": "audio", "file": (files.get("audio") or [None]).pop(0),
                    "mel": u8(np.clip((mel - lo) / max(hi - lo, 1e-6), 0, 1) * 255),
                    "mel_shape": list(mel.shape),
                    "hop_seconds": (fe.hop_length / fe.sampling_rate) if fe and hasattr(fe, "hop_length") else None,
                    "encoder_input": layer_maps([st["input"][n]])[0],
                    "encoder_frames": valid,
                    "encoder_layers": layer_maps(frames),
                    "soft": soft(run),
                })
        return out

    @staticmethod
    def reconstruct(patches, xy, p, gx, gy):
        """The exact pixels the vision tower received, reassembled from its patch tensor (PNG)."""
        from PIL import Image

        x = patches.float().view(-1, p, p, 3).clamp(0, 1).numpy()  # processor emits (p, p, 3) patches in [0, 1]
        img = np.zeros((gy * p, gx * p, 3), np.float32)
        for (px, py), tile in zip(xy.tolist(), x):
            img[py * p:(py + 1) * p, px * p:(px + 1) * p] = tile
        buf = io.BytesIO()
        Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
