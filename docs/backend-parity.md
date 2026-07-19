# Backend Parity Summary (Vulkan / Native WebGPU / Portable WebGPU)

Machine-readable source of truth: [`backend-parity-manifest.json`](./backend-parity-manifest.json)  
Normative criteria: [`backend-parity-spec.md`](./backend-parity-spec.md)

**Baseline commit:** `943192e140cde1d9a384556aeceed7331779333d` (`wgpu/vulkan`)  
**CUDA baseline:** actual `BackendStorage` / `BackendDevice` surface of that commit.

## Profiles (do not merge)

| Profile | Meaning |
| --- | --- |
| **Native Vulkan** | Direct Vulkan/SPIR-V (`candle-core` + `candle-vulkan-kernels`) |
| **Native WebGPU** | Native `wgpu` (Vulkan/DX12/Metal) with runtime feature detection |
| **Portable WebGPU** | Browser/WASM-safe WGSL subset; no native-only features claimed |

## Status vocabulary

`Native` · `Optimized` · `GPUEmulated` · `UnsupportedBySpecification` · `UnsupportedByHardware` · `CudaSpecific` · `Missing` · `Verified`

Rules of thumb:

- **Verified** requires a real test reference in the manifest.
- **Optimized** requires `bench: true` and a specialized fast path with generic GPU fallback.
- **GPUEmulated** is GPU-side composition (e.g. BF16 via F32 hub), **not** host compute and **not** silent user-visible dtype lies.
- **Missing** is unfinished work; CI audit fails if CUDA-required ops remain Missing.
- Portable **Verified** needs WASM/portable/browser evidence — native-only smokes are insufficient.

## Static trait presence (audit)

```text
python scripts/backend_parity_audit.py
```

As of the baseline, all CUDA-required `BackendStorage` methods have non-immediate Vulkan and WebGPU implementations. Depth (dtype × layout × edges × perf) is tracked in the JSON manifest.

## Policy highlights

- No hidden CPU compute fallback; counters + `fallback_runtime_audit` example.
- No silent host cast of unsupported dtypes to F32 for “fake” success.
- Storage-only dtype support ≠ arithmetic support.
- BF16 arithmetic on Vulkan/WebGPU is typically **GPUEmulated** (GPU F32 hub).
- F16 without `SHADER_F16` on native/portable WebGPU is **GPUEmulated**.
- F64 on portable WebGPU is **UnsupportedBySpecification** (base WGSL).
- NCCL / multi-GPU distributed: **CudaSpecific**.

## Key test commands

```text
# Static audit (requires docs/backend-parity-manifest.json)
python scripts/backend_parity_audit.py

# Smoke (per backend)
cargo test -p candle-core --features vulkan --test backend_smoke_tests
cargo test -p candle-core --features wgpu --test backend_smoke_tests

# CUDA differential matrix (hardware)
set CANDLE_REQUIRE_CUDA_TEST_DEVICE=1
set CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1
set CANDLE_REQUIRE_WGPU_TEST_DEVICE=1
cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_parity_matrix_tests

# Shader materialization
cargo test -p candle-core --features vulkan --test gpu_shader_validation_tests

# Zero CPU-fallback runtime audit
cargo run -p candle-core --release --features "vulkan,wgpu" --example fallback_runtime_audit

# Microbench (release; compare CUDA/Vulkan/wgpu on same GPU)
cargo run -p candle-core --release --features "cuda,vulkan,wgpu" --example backend_parity_microbench -- --suite
```

## Portable WebGPU

See [`portable-webgpu-capability-matrix.md`](./portable-webgpu-capability-matrix.md).

## SLO (release)

| Profile | End-to-end vs CUDA | Critical kernel |
| --- | --- | --- |
| Native Vulkan | ≤15% slower (hot geomean ≤20%) | ≤1.5× without profiled limit |
| Native WebGPU | ≤30% slower | ≤2× without profiled limit |
| Portable WebGPU | no universal % vs CUDA | GPU-only; profile vs native WebGPU on same HW |

Deviations must be recorded with objective API/hardware evidence.
