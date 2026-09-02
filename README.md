<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Candle fork with native Vulkan and WGPU / WebGPU backends.</b><br>
  CUDA-parity focus for inference on Linux, Windows, macOS, Android, and WASM.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0%20%2F%20MIT-2ea44f" alt="Apache 2.0 / MIT">
  <img src="https://img.shields.io/badge/Rust-edition%202021-93450a?logo=rust" alt="Rust edition 2021">
  <img src="https://img.shields.io/badge/Backends-Vulkan%20%2B%20WGPU-5B7CFA" alt="Vulkan and WGPU">
  <img src="https://img.shields.io/badge/Fork-0.0.174-d4730e" alt="Fork version 0.0.174">
  <img src="https://img.shields.io/badge/Upstream-Candle%200.11.0-232323" alt="Upstream Candle 0.11.0">
  <img src="https://img.shields.io/badge/Branch-wgpu%2Fvulkan-232323" alt="wgpu/vulkan branch">
</p>

<h1 align="center">Candle / wgpu · vulkan</h1>

## Platform matrix

| Backend | Feature | Linux | Windows | macOS | Android | WASM |
|---------|---------|:-----:|:-------:|:-----:|:-------:|:----:|
| CPU | (default) | ✅ | ✅ | ✅ | ✅ | ✅ |
| CUDA | `cuda` | ✅ | ✅ | ❌ | ❌ | ❌ |
| Metal | `metal` | ❌ | ❌ | ✅ | ❌ | ❌ |
| WGPU | `wgpu` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vulkan | `vulkan` | ✅ | ✅ | ❌ | ✅ | ❌ |

**CPU** and **CUDA** backends are kept in direct sync with [huggingface/candle](https://github.com/huggingface/candle) `main`. Nothing from those upstream backends has been removed; this fork only adds Vulkan and WGPU on top.

## Upstream correspondence

| Fork ([FerrisMind/candle](https://github.com/FerrisMind/candle)) | Upstream ([huggingface/candle](https://github.com/huggingface/candle)) |
|------------------------------------------------------------------|------------------------------------------------------------------------|
| Branch `wgpu/vulkan` | `main` |
| Fork crates **0.0.174** (`candle-core` / `candle-nn` / `candle-transformers` / `candle-examples` / `candle-*-kernels` for Vulkan & WGPU); unchanged crates stay **0.11.0** | Candle **0.11.0** |
| Last CPU/CUDA sync [`b3e5b40f`](https://github.com/FerrisMind/candle/commit/b3e5b40f) (2026-08-17) | Tip [`162b59b9`](https://github.com/FerrisMind/candle/commit/162b59b9) (#3892) |
| Fork-only | Native **Vulkan** + **WGPU / WebGPU** |

CPU and CUDA stay in sync with upstream `main` (nothing removed). Vulkan and WGPU exist only in this fork.

## Table of Contents

- [Platform matrix](#platform-matrix)
- [Upstream correspondence](#upstream-correspondence)
- [What is this?](#what-is-this)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Backend Performance](#backend-performance)
- [System Requirements](#system-requirements)
- [License](#license)

## What is this?

This is a fork of [huggingface/candle](https://github.com/huggingface/candle) on the default branch [`wgpu/vulkan`](https://github.com/FerrisMind/candle/tree/wgpu/vulkan). Upstream remains the source for the tensor API, models, and CPU / CUDA / Metal paths — those backends stay synchronized with `main` and are not stripped. This branch adds and hardens:

- a **native Vulkan** compute backend (`ash` + SPIR-V)
- a **WGPU / WebGPU** compute backend (`wgpu` + WGSL)
- parity docs, smoke tests, and CUDA differential coverage for those backends

Parity is tracked in three separate profiles (do not mix results):

| Profile | Meaning |
|---------|---------|
| Native Vulkan | Direct Vulkan / SPIR-V |
| Native WebGPU | Native `wgpu` with runtime feature detection |
| Portable WebGPU | Browser / WASM-safe WGSL; no native-only claims |

Normative docs:

- [`docs/backend-parity-spec.md`](./docs/backend-parity-spec.md)
- [`docs/backend-parity.md`](./docs/backend-parity.md)
- [`docs/backend-parity-manifest.json`](./docs/backend-parity-manifest.json)

## Key Features

- Cargo features `vulkan` and `wgpu` enable the new GPU backends.
- Device selection via `CANDLE_DEVICE=vulkan|wgpu` (also `cuda`, `metal`, `cpu`).
- Adapter pick: `CANDLE_VULKAN_DEVICE_NAME`, `CANDLE_WGPU_ADAPTER_NAME`.
- WGPU API override: `WGPU_BACKEND=vulkan|dx12|metal|gl`.
- No hidden CPU compute returned as GPU; no silent dtype cast to F32.
- Static parity audit, smoke tests, CUDA differential matrix, fallback audit, and example bench harness.

## Repository Layout

| Path | Purpose |
|------|---------|
| `candle-core` | Tensor API, devices, Vulkan / WGPU storage and ops |
| `candle-vulkan-kernels` | SPIR-V compute shaders |
| `candle-wgpu-kernels` | WGSL compute shaders |
| `candle-nn` | Layers (incl. MoE and flash-attn dispatch) |
| `candle-transformers` | Model implementations |
| `candle-examples` | Runnable examples for e2e backend coverage |
| `docs/` | Parity specification, manifest, and evidence |
| `scripts/` | Parity audit and related tooling |
| `bench_examples.py` | Multi-backend example throughput harness |

## Quick Start

### Run an example on Vulkan or WGPU

```powershell
$env:CANDLE_DEVICE = "vulkan"   # or "wgpu"
cargo run -p candle-examples --release --features vulkan --example quantized-qwen3 -- --model <path-to-gguf>
cargo run -p candle-examples --release --features wgpu --example quantized-qwen3 -- --model <path-to-gguf>
```

### Parity checks

```powershell
python scripts/backend_parity_audit.py

cargo test -p candle-core --features vulkan --test backend_smoke_tests
cargo test -p candle-core --features wgpu --test backend_smoke_tests

# CUDA differential matrix (requires GPUs)
$env:CANDLE_REQUIRE_CUDA_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_VULKAN_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_WGPU_TEST_DEVICE = "1"
cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_parity_matrix_tests

cargo run -p candle-core --release --features "vulkan,wgpu" --example fallback_runtime_audit
cargo run -p candle-core --release --features "cuda,vulkan,wgpu" --example backend_parity_microbench -- --suite

python bench_examples.py --models-root <models-root> --backend cuda --backend vulkan --backend wgpu
```

## Backend Performance

End-to-end throughput for [`quantized-qwen3`](./candle-examples/examples/quantized-qwen3/) (**Qwen3-0.6B-GGUF Q4_K_M**, release, same-session CUDA baseline).

Hardware: **RTX 3060 12 GB**, **Ryzen 7 3700X**, **64 GB DDR4**. `%CUDA` is relative CUDA throughput; Min / Normal / Goal are SLO tiers from [`bench_examples.py`](./bench_examples.py).

| Backend | Phase | Cell | tok/s | %CUDA | Min | Normal | Goal | Verdict |
|---------|-------|------|------:|------:|----:|-------:|-----:|---------|
| Vulkan | decode | tg128 | 121.51 | 202% | 75 | 90 | 90+ | PASS ×3 |
| Vulkan | decode | tg256 | 121.26 | 203% | 75 | 90 | 90+ | PASS ×3 |
| Vulkan | prefill | pp512 | 120.72 | 205% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp1024 | 117.55 | 207% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp2048 | 112.03 | 190% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp4096 | 96.46 | 220% | 85 | 95 | 95+ | PASS ×3 |
| WGPU | decode | tg128 | 53.85 | 90% | 10 | 18 | 30+ | PASS ×3 |
| WGPU | decode | tg256 | 52.99 | 89% | 10 | 18 | 30+ | PASS ×3 |
| WGPU | prefill | pp512 | 52.87 | 90% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp1024 | 45.18 | 80% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp2048 | 33.50 | 57% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp4096 | 22.01 | 50% | 12 | 22 | 38+ | PASS ×3 |

CUDA baseline (same session): tg128 60.09, tg256 59.57, pp512 58.79, pp1024 56.74, pp2048 58.78, pp4096 43.91 tok/s.

Release SLO targets (end-to-end vs CUDA): **Vulkan ≤ 15% slower** (stretch 10%); **native WebGPU ≤ 30% slower** (stretch 20%). Portable WebGPU has no fixed CUDA %; investigate if more than 2× slower than native WebGPU on the same GPU.

Logs: `bench_logs/qwen3-q4km_{cuda,vulkan,wgpu}_final.log`.

## System Requirements

- Rust toolchain with workspace `edition = "2021"`
- For **Vulkan**: working Vulkan loader / ICD (Linux, Windows, Android)
- For **WGPU**: a `wgpu`-supported adapter (Vulkan, DX12, Metal, or browser WebGPU)
- Optional CUDA for differential parity and SLO baselines
- Python 3.x for `scripts/backend_parity_audit.py` and `bench_examples.py`

## License

Code in this repository follows upstream Candle dual licensing: [Apache-2.0](./LICENSE-APACHE) and [MIT](./LICENSE-MIT).

Upstream model weights keep their original licenses and usage restrictions.
