# Supported models

VirtualCuda supports a model only if **every panel in the simulator is exact** for it. A model either qualifies, and loads, or it isn't supported and won't load. Two layouts qualify:

```
plain pre-norm (Llama, Qwen, Mistral, …)
  token ids → embedding → N × [ x += attention(norm(x))
                                x += MLP(norm(x))       ] → final norm → lm_head → next token

Gemma 4 E2B / E4B
  token ids → embedding (× √d) → N × [ x += post_norm(attention(norm(x)))
                                       x += post_norm(MLP(norm(x)))
                                       x += norm(proj(act(gate(x)) × per-layer input))
                                       x *= layer scalar                              ]
            → final norm → lm_head → soft-cap → next token
  images → vision encoder → pool → soft tokens ┐
  audio  → audio encoder        → soft tokens ┴→ fill <|image|> / <|audio|> slots in the sequence
```

## Supported

| Family | `model_type` | Inputs | Examples |
|---|---|---|---|
| Gemma 4 E2B / E4B | `gemma4` | text, image, audio | google/gemma-4-E2B-it, google/gemma-4-E4B-it |
| Qwen2 / Qwen2.5 | `qwen2` | text | Qwen/Qwen2.5-0.5B / 1.5B / 3B-Instruct |
| Qwen3 (dense) | `qwen3` | text | Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B |
| Llama 2 / 3.x, SmolLM2 | `llama` | text | HuggingFaceTB/SmolLM2-360M-Instruct, meta-llama/Llama-3.2-1B-Instruct |
| Mistral (dense) | `mistral` | text | mistralai/Mistral-7B-Instruct-v0.3 |
| Phi-3 / 3.5 / 4-mini | `phi3` | text | microsoft/Phi-3.5-mini-instruct |
| Gemma (1st generation) | `gemma` | text | google/gemma-2b-it |
| SmolLM3 | `smollm3` | text | HuggingFaceTB/SmolLM3-3B |
| OLMo (1st generation) | `olmo` | text | allenai/OLMo-1B-hf |

Tested end to end: **Gemma 4 E2B** (text, image, audio), **Qwen2.5-1.5B-Instruct** and **SmolLM2-360M-Instruct**. Every other model on this list still has to pass the self-test (below) when it loads, or it's refused.

The model must also:

- be the original **bf16 / fp16** weights (not a quantized repo),
- fit in **65% of this machine's memory** in bf16 (≈ 2 bytes per parameter; about 8B parameters on 24 GB),
- have a **chat template**.

## What you see for images and audio (Gemma 4 E2B / E4B)

Everything is captured from the real forward pass. Nothing is illustrative.

- **Image:** the exact pixels the vision tower received (rebuilt from its patch tensor), the patch and soft-token grids, every vision-encoder layer's patch features (PCA colors and norms), the pooled soft tokens, and, at every decoder layer, where attention lands on the image.
- **Audio:** your stored file's waveform, the log-mel spectrogram the encoder received, every audio-encoder layer per frame, the soft tokens, and where decoder attention lands in time.
- **Decoder:** the same per-layer stages as text models, plus the raw vs. post-normed sublayer outputs, the per-layer-input stage, the layer scalar, KV-sharing and sliding-window labels, and the soft-capped logits.

Video input isn't supported yet.

## Not supported

| What | Why | Examples |
|---|---|---|
| Mixture of experts | The MLP is a router plus many experts, not one MLP | Gemma 4 26B-A4B, Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek-V2/V3, gpt-oss, Llama 4 |
| Non-attention layers | State-space, recurrent or linear-attention layers | Mamba, RWKV, Jamba, Zamba, Falcon-H1, Qwen3-Next, LFM2, RecurrentGemma |
| Other Gemma generations | Different layouts, not verified | Gemma 2, Gemma 3, Gemma 3n, Gemma 4 12B (`gemma4_unified`) |
| Parallel blocks | Attention and MLP read the same input at the same time | Phi-1 / Phi-2, GPT-J, GPT-NeoX / Pythia, Falcon, Cohere |
| Conv1D layers | No linear down-projection to hook | GPT-2, StarCoder 1 (gpt_bigcode) |
| Scaled residuals / logits | Extra multipliers in the residual stream and output | Granite |
| Other multimodal models | Encoders we haven't mapped | LLaVA, Qwen2.5-VL, Llama 3.2 Vision |
| Encoders, encoder-decoders | Not next-token generators | BERT, RoBERTa, T5, BART |
| Quantized checkpoints | Packed weights need special kernels | GPTQ, AWQ, bitsandbytes, FP8, MLX, GGUF repos |
| Too big | Over 65% of memory in bf16 | Gemma 4 31B, anything much over 8B on 24 GB |
| Anything not listed above | Unknown layout | |

Gated repos (Llama, older Gemma) need their license accepted on huggingface.co and `huggingface-cli login` on this machine before they can download. Gemma 4 E2B / E4B are Apache-2.0 and open.

## How it's checked

**Before downloading**, VirtualCuda reads only `config.json` and the parameter count from the Hub, then rejects anything from the "Not supported" table, anything not on the supported list, quantized weights, or a model too big for memory.

**After loading**, a self-test runs one short forward pass through the simulator's own hooks and must pass every check. Otherwise the model is unloaded and the previous one comes back. Multimodal models get a small probe image and tone to exercise their encoders. The probes are only for the check and never shown.

| Check | Must hold |
|---|---|
| Parts found | A list of N decoder layers, each with an attention module and an MLP with a linear down-projection, plus a final norm and lm_head |
| Attention weights | Every layer returns per-head attention weights (eager attention) |
| MLP neurons | Every layer's intermediate activations can be captured |
| Residual stream | In every layer, output = input + what attention adds + what the MLP adds (+ per-layer input, × layer scalar for Gemma 4), up to bf16 rounding |
| Logit lens | The model's head (final norm, lm_head, soft-cap) on the last layer reproduces its real logits |
| Image / audio input | Each encoder's layers are captured and its output fills the matching placeholder slots |
| Chat template | The tokenizer or processor can format a conversation |

Each one is what a simulator panel relies on. The residual check, for example, is what makes "residual after attention = in + added" true, and the logit-lens check is what makes the per-layer predictions mean something.

## Chat features (don't affect support)

- **Tool calling** is on when the model's chat template accepts tools (Gemma 4, Qwen2.5, Qwen3, Llama 3.1+, SmolLM3, …). The server reads calls written as `<tool_call>{json}</tool_call>` (Qwen, Hermes), Gemma 4's `<|tool_call>call:name{…}<tool_call|>`, or a bare `{"name": …, "arguments": …}` object.
- **Images and audio** go through the OpenAI content parts `image_url` (data: URL) and `input_audio` (base64 WAV). The chat page converts any audio, including microphone recordings, to 16 kHz mono WAV in the browser. Files are stored once under `data/media/` so replays can re-run them.
- **Replays** always use the model that recorded the chat, so that model has to stay in the local Hugging Face cache.

## Adding a family

If a family has one of the layouts above but isn't listed, add its `model_type` to `SUPPORTED` in `compat.py` and load it. The self-test decides. If the parts aren't found, the name patterns are in `models.py` (`_ATTN`, `_MLP`, `_DOWN`), and the optional parts (post-norms, per-layer input, layer scalar, encoders) are detected in `discover()`.
