# Portable WebGPU Capability Matrix

**Profile:** Portable WebGPU (browser / WASM / standard WGSL)  
**Not the same as:** Native `wgpu` on Vulkan/DX12/Metal with optional features.

This matrix documents what may be claimed under `portable_webgpu_status` in
[`backend-parity-manifest.json`](./backend-parity-manifest.json).

## Portable baseline assumptions

| Area | Portable assumption |
| --- | --- |
| Shader language | WGSL as defined for WebGPU (no vendor SPIR-V extensions) |
| Bindings | Standard bind groups / storage buffers |
| F16 | **Not guaranteed** without `shader-f16`; without it → `GPUEmulated` (F32 hub) or unsupported arithmetic |
| F64 | **Not in portable WGSL core** → `UnsupportedBySpecification` for compute |
| BF16 | **Not portable native** → `GPUEmulated` via F32 on GPU |
| Subgroups | Not assumed portable |
| Cooperative matrix | Native-only (`EXPERIMENTAL_COOPERATIVE_MATRIX`) — not portable |
| Immediates / push constants | Native optimization only |
| CPU fallback | **Forbidden** for claimed portable ops |
| Steady-state pipelines | Must not recompile every dispatch |

## Feature classification

| Capability | Portable status | Notes / evidence |
| --- | --- | --- |
| Device create / alloc / H2D / D2H / sync | Supported | Core WebGPU buffers |
| F32 unary / binary / reduce / where | Supported | Standard WGSL |
| F32 matmul (generic) | Supported | Generic WGSL GEMM; no coop-matrix |
| F32 conv/pool (GPU) | Supported where shaders are pure WGSL | Same as native generic path |
| Integer ops (I32/U32/…) | Supported where WGSL types allow | Exact match required in tests |
| F16 storage | Often storage-only without extension | Do not claim full native arithmetic |
| F16 arithmetic without `shader-f16` | `GPUEmulated` | GPU F32 hub |
| F16 arithmetic with `shader-f16` | Native-only / not portable claim | Feature is adapter-specific |
| F64 arithmetic | `UnsupportedBySpecification` | Base WGSL lacks f64 |
| BF16 arithmetic | `GPUEmulated` | GPU F32 hub |
| Quantized matmul needing f16 shaders | `UnsupportedBySpecification` or feature-gated error | Honest typed error preferred |
| Quantized dequant (pure WGSL) | `GPUEmulated` / Native if pure | See wasm quantized tests |
| NCCL / multi-GPU | `CudaSpecific` | N/A |

## Runtime reporting rules

1. Query adapter features/limits; never assume desktop native features in portable builds.
2. If an op cannot run on portable profile: typed error or capability query — not panic-only.
3. Native-only smoke tests on RTX-class GPUs do **not** count as portable `Verified`.
4. Portable `Verified` requires one of: `portable_tests` entry, `candle-wasm-*` test, or browser harness log.

## Environment on this host

| Item | Status |
| --- | --- |
| Native wgpu (Vulkan backend) | Available (RTX 3060) — **not** portable evidence alone |
| Browser WebGPU harness | May be unavailable in headless CI; capture limit honestly |
| `candle-wasm-tests` | Structural/unit evidence for portable subset |
| WGSL sources | `candle-wgpu-kernels/src/shaders/**` |

If browser e2e cannot run, acceptance rests on: portable status classification, WGSL validation, wasm tests, and no native-only feature leakage into portable claims.

## Comparison rule vs native WebGPU

For any hot path that is **>2× slower** than native WebGPU on the **same** GPU when both are measurable, profile and document (pipeline churn, extra copies, missing subgroup path, etc.). No single “% of CUDA” SLO applies to all browsers.
