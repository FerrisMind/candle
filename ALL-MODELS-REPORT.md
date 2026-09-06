# All-models matrix: G:\models on cpu / vulkan / wgpu (2026-09-06)

Branch `wgpu/vulkan`, rev fe21097c (+ earlier 159a772d/92fa25c5). Protocol: greedy (`--temperature 0`),
fixed seed where supported, same inputs across devices, sequential GPU jobs. RTX 5080 16GB.

## Results

| Model | CPU | Vulkan | WGPU | Cross-device parity |
|---|---|---|---|---|
| bert MiniLM-L6-v2 | ok | ok | ok | embeddings identical on all 3 |
| bge-small-en-v1.5 | ok | ok | ok | embeddings identical on all 3 |
| mamba2-130m | ok | 26.9 tok/s | 8.0 tok/s | identical text (after fixes, see below) |
| llama-3.2-1B | ok | 3.7 tok/s | 4.8 tok/s | identical text |
| qwen3-0.6B Q8_0 (GGUF) | ok | 89.6 tok/s | 56.6 tok/s | identical text |
| qwen3-0.6B Q4_K_M (GGUF) | ok | ok | ok | identical text |
| qwen3-moe-16B-A3B Q4_K_M | **fail** (see notes) | 8.6 tok/s | 1.7 tok/s | identical text (vk/wgpu) |
| recurrent-gemma-2b q4k | ok | 1.8 tok/s | 1.0 tok/s | identical (degenerate greedy text, same everywhere) |
| rwkv7-g1d-0.1b | ok | 49.4 tok/s | 19.2 tok/s | identical text |
| t5-small | ok | ok | ok | identical translation (encoder logits mode + greedy decode mode) |
| whisper tiny.en | ok | ok | ok | identical JFK transcription; identical segment layout on tone input |
| clip ViT-B/32 | ok | ok | ok | probabilities match to 4–5 decimals, same ranking |
| segment-anything (mobile_sam) | ok | ok | ok | mask sign agreement 100%; maxabs diff vs cpu: vk 6.7e-3, wgpu 3.2e-5; iou 0.9819 |
| encodec 24khz | ok | ok | ok | corr 1.0, maxdiff 1 int16 LSB |
| stable-diffusion v1-5 (256², 8 steps) | skipped (unseedable RNG) | 0.15 s/step | 1.4 s/step | bitwise-reproducible per backend (same md5 across runs); cross-backend images differ — per-device seeded RNG = different initial latents (inherent, not a bug) |
| resnet50 | ok | ok | ok | identical top-5, ≤0.04 pp spread |
| yolo-v8s | ok | ok | ok | (earlier campaign) clean detections, bisect maxabs 1.2e-3 |

Peak VRAM seen: 7.2 GiB (sd15) / 16B MoE fit with no OOM. No memory bloat anywhere.

## Bugs found and fixed (commit fe21097c, pushed)

1. **candle-nn layer_norm regression (b96522e6)** — the BERT gamma/beta-compat change made the
   `bias`/`beta` lookup unconditional; every biasless RMSNorm model (llama, qwen3, recurrent-gemma,
   mamba2, …) failed to load with `Failed to find weight tensor`. Bias is now only required when
   `affine = true`, matching upstream.
2. **mamba2 HF checkpoint compat** — accept `backbone.*` prefix and final norm named `norm`
   (HF `*-hf` repos) in addition to candle's flat `embeddings/layers/norm_f` layout.
3. **whisper local files** — added `--config-file/--tokenizer-file/--weight-file` (was hub-only).
4. llama hf-hub 0.5 revision fix + encodec cpal 0.18 port (follow-ups to 92fa25c5, committed here).

## Known limitations (not backend bugs, not fixed)

- quantized-qwen3-moe on **CPU**: `unsupported dtype BF16 for op matmul` — upstream candle CPU
  limitation for the GGUF MoE path; both GPU backends fine.
- SD seeds are not portable across backends: vulkan and wgpu RNG kernels produce different
  initial latents for the same seed. Per-backend determinism verified bitwise.
- recurrent-gemma GPU decode is slow (1–1.8 tok/s): sequential recurrent scan, many small ops —
  latency-bound, not a GEMM hole.

## Models in G:\models without candle support

- LTX-Video-0.9.8-2B-distilled, WAN — no candle model/example.
- OmniVoice — empty directory.
- clip-vit-base-patch32 — weights absent locally (tokenizer.json only; hub download works and was used).
- recurrent-gemma-2b-it — config/tokenizer only, no weights (used as config source for q4k run).
- peculiar-ragdoll (Qwen3.8-27B-GSQ, Tiel-Coder-35B-A3B GGUF) — exceeds 16 GB VRAM; GSQ quant format unsupported.
- qwen3_vl — model code exists in candle-transformers, no example to drive it.
