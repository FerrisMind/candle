# CUDA ↔ Vulkan / WebGPU Backend Parity

Date: 2026-07-17  
Baseline CUDA commit: `b51db4c11751ee1e765398b1847e73f151a5e412`  
Work branch tip (perf): see `git rev-parse HEAD` on `feat/vulkan-webgpu-cuda-parity`  
Machine-readable inventory: [`backend-parity-manifest.json`](./backend-parity-manifest.json)  
Related snapshots: [`../cuda-wgpu-vulkan-parity-matrix.md`](../cuda-wgpu-vulkan-parity-matrix.md), [`../wgpu-vulkan-parity-audit.md`](../wgpu-vulkan-parity-audit.md)

## Perf snapshot (RTX 3060, release, batch20 median_ms)

Harness: `cargo run -p candle-core --example backend_parity_microbench --features cuda,vulkan,wgpu --release -- --suite`  
Evidence: SCRATCH `bench-suite-immediates-ok.log` / tip `git rev-parse HEAD`.

| op | CUDA | Vulkan | WGPU | Vulkan×CUDA | WGPU×CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul 256³ | ~0.019 | ~0.022 | ~0.024 | **~1.1 ✓** | **~1.2** |
| matmul 1024³ | ~0.32 | ~0.22 | ~0.23 | **0.7 ✓** | **0.72 ✓** |
| matmul 64×4096×4096 | ~0.29 | ~0.26 | ~0.34 | **0.90 ✓** | **~1.20** |
| relu 1024² | ~0.034 | ~0.038 | ~0.036 | **~1.1 ✓** | **~1.05 ✓** |
| mul 1024² | ~0.043 | ~0.051 | ~0.049 | **~1.2** | **~1.13** |
| sum_last 1024² | ~0.10 | ~0.025 | ~0.039 | **0.25 ✓** | **0.39 ✓** |

✓ = ≤1.10× CUDA.

**WGPU elementwise:** (1) cache preprocessed unary/binary WGSL; (2) atomic
dyn-uniform + stack deferred payloads; (3) unary `Features::IMMEDIATES`;
(4) `CONTIG` binary define skips broadcast indexing for same-shape contiguous
ops. Prior residual ~3× was **rebuild/preprocess of WGSL every dispatch**.
Multi-run: relu ~1.05×, mul ~1.08–1.15× (Vulkan mul often ~1.09×). Smoke 84/84.

**WGPU tall GEMM (~1.20–1.25×):** dual-MMA 128×64 virtual-Bᵀ still best on RTX 3060.
Rejected (all regressed vs dual, evidence in SCRATCH `bench-suite-*.log`):
coop64-skinny, BK=64 multi-MMA panel, N-coalesced BT (± pad), 128×32/256-thread
tall, materialize Bᵀ (~6×), Vulkan-style 128-thread 64×64 BK=64 coalesced VBT
(~2×, correct but slower), dual K-panel=32 (~2×). Vulkan stays ~0.9× CUDA on
the same GPU — remaining gap is WGSL coopLoad / occupancy tradeoffs.

## Scope

This audit inventories the **CUDA tensor-operation surface** defined by:

| Layer | Location |
| --- | --- |
| Trait surface | `candle-core/src/backend.rs` (`BackendStorage`, `BackendDevice`) |
| CUDA storage | `candle-core/src/cuda_backend/mod.rs` |
| CUDA device | `candle-core/src/cuda_backend/device.rs` |
| CUDA kernels | `candle-kernels/src/*.cu` (+ `mmq_gguf/`, `moe/`) |
| Vulkan | `candle-core/src/vulkan_backend.rs` + `candle-vulkan-kernels` |
| WebGPU | `candle-core/src/wgpu_backend.rs` + `candle-wgpu-kernels` |
| Quantized | `candle-core/src/quantized/` (CPU recovery removed; GPU dequant+dense on specialized-kernel miss) |

**CUDA does not implement** (trait defaults / explicit bail) and is **out of CUDA-parity scope**:

- `cumsum_last_dim` — default “not implemented”
- `clamp` — default “not implemented” (host may compose min/max)
- `upsample_nearest1d` — explicit CUDA bail

Vulkan and WebGPU *do* implement those three; they are extras, not CUDA gaps.

## Status legend

| Status | Meaning |
| --- | --- |
| `native` | GPU path for core CUDA-relevant dtypes with certified smoke/model coverage |
| `partial` | GPU path exists with dtype / layout / rank / perf restrictions |
| `missing` | No usable GPU path for CUDA-core cases (immediate unsupported or only CPU recovery) |
| `unsupported_spec` | Not part of CUDA’s supported surface |

## Counts (full manifest: 47 rows)

Curated statuses in `backend-parity-manifest.json` (`vulkan_status` / `wgpu_status`):

| Backend | native | partial | missing | notes |
| --- | ---: | ---: | ---: | --- |
| **Vulkan** | **11** | **36** | **0** | 3 CUDA-N/A ops still listed as `partial` because GPU implements them |
| **WebGPU** | **11** | **36** | **0** | same shape as Vulkan at method level |

**Native (11, both backends):**  
`try_clone`, `to_cpu_storage`, `cmp`, `binary_impl`, `where_cond`, `device::zeros_impl`, `device::alloc_uninit`, `device::storage_from_cpu_storage`, `device::set_seed`, `device::synchronize`, `argsort_last_dim`

**CUDA-required storage ops with real CUDA impl:** all `BackendStorage` methods except `cumsum_last_dim`, `clamp`, `upsample_nearest1d` (plus accessors `dtype`/`device`).

Notes:

- Both backends **implement every CUDA `BackendStorage` method** with a non-immediate stub for the methods CUDA implements; remaining work is **depth** (dtypes, layouts, performance), not blank methods.
- `missing` is zero at the trait-method level; residual holes appear as **dtype/layout/perf** inside `partial` rows (e.g. f8e4m3, full F16 GEMM, native Q8K MMQ).
- Quantized paths no longer use **CPU result recovery**: missing specialized kernels fall through to **GPU dequant + dense matmul/index_select**. Fallback counters remain for diagnostics but production paths do not increment them.

### Status breakdown by family

| Family | Vulkan | WGPU |
| --- | --- | --- |
| Device alloc / H2D / D2H / sync / zeros | native | native |
| `try_clone`, `to_cpu_storage`, `set_seed`, `synchronize` | native | native |
| Binary / cmp / where (core dtypes) | native | native |
| Argsort / sort_last_dim (widened dtypes) | native | native |
| Reduce (f32 + half decomp + int) | partial | partial |
| Unary / affine / powf / elu | partial | partial |
| `to_dtype` / cast matrix | partial | partial |
| Matmul / GEMM | partial | partial |
| Conv / pool / upsample | partial | partial |
| Indexing / scatter / gather | partial | partial |
| Quantized load / dequant / qmatmul / MoE | partial | partial |
| Softmax / RMSNorm / RoPE (nn-level) | partial | partial |
| RNG | partial | partial |
| Layout / copy / const_set | partial | partial |

## Top 20 remaining gaps (by impact)

Ordered for model correctness first, then throughput, then surface completeness.

| # | Gap | Severity | Backends | Why it matters | Suggested direction |
| ---: | --- | --- | --- | --- | --- |
| 1 | **Silent / residual quantized CPU fallback** on dequant, qmatmul, q-index_select | Critical | Both | Breaks “GPU-resident” inference; only place `record_*_cpu_fallback` fires today | Close remaining quant dtypes; CI on fallback count == 0 |
| 2 | **Quantized matmul perf vs CUDA MMQ/MMVQ** | Critical | Both | Decoder tokens/sec; Q8K is dequant+dense | Port/profile MMVQ/MMQ routes; keep reuse-first where faster |
| 3 | **Dense matmul F16 / BF16 specialization** (BF16 via F32 GEMM) | High | Both | Half models pay 2–3× bandwidth | Cooperative matrix / half GEMM when profiling requires |
| 4 | **Conv2d + matmul host overhead (Vulkan ConvMixer)** | High | Vulkan | ~6–8× slower than CUDA in release traces historically | Descriptor/batch residual + better conv/GEMM routing |
| 5 | **Always-CPU `quantize*`** (download → CPU pack → upload) | High | Both | Weight conversion / online quant not GPU | Device-side quantize (Q8_0/Q4_0 first) |
| 6 | **F8E4M3 (and microfloat) surface** present on CUDA Map1 | High | Both | CUDA maps many ops through f8e4m3; GPU backends omit | Defer unless models need it; document as out-of-scope or add cast+compute |
| 7 | **Scatter/index_add dtype matrix** beyond f32+u32 | High | Both | Training / sparse updates | Extend set_rows family dtypes |
| 8 | **Gather / index_select / scatter strided + non-contiguous** | High | Both | CUDA requires contiguous; GPU often same — strided views force materialize cost | GPU materialize only; never CPU |
| 9 | **Conv1d/2d half + groups/dilation edge matrix** | Medium | Both | Audio/vision graphs | Widen smoke; profile Whisper/ConvMixer |
| 10 | **Pool / upsample beyond f32** | Medium | Both | Vision pipelines | f16 path or documented f32-only policy |
| 11 | **F64 hot path** (binary/unary/reduce via f32 cast) | Medium | Both | Numerics drift vs CUDA f64 kernels | Keep emulated; only promote if models need true f64 |
| 12 | **Integer affine / unary breadth** vs CUDA Map1 | Medium | Both | Rare in transformers | Error clearly; or implement if API users hit it |
| 13 | **RoPE custom ops** (`RotaryEmb*`) CUDA-only in some crates | Medium | Both | nn path closed via shaders/decomp; custom op trait holes | `wgpu_fwd` / `vulkan_fwd` or route all models through nn |
| 14 | **MoE expert kernels** incomplete vs CUDA `moe/` | Medium | Both | Sparse experts; public model matrix weak | Fuse with `quantized_indexed_moe`; add real MoE case |
| 15 | **InplaceOp1/2/3** no wgpu/vulkan dispatch | Medium | Both | `storage.rs` hard-errors | Extend trait + backends if inplace APIs are required |
| 16 | **Flash-attention** CUDA-only crates | Medium | Both | Long-context training/inference | SDPA composition already partially certified; flash is optional |
| 17 | **RNG f64 / half** | Low | Both | CUDA f32/f64 only for uniform/normal; GPU f32 | Match CUDA: document unsupported half RNG |
| 18 | **Dedicated top-k kernel** (argsort composition only) | Low | Both | MoE routing | Profile; add only if composition shows up |
| 19 | **Rank>4 / rank0 edge completeness** | Low | Both | Many paths compact materialize | Exhaustive property tests |
| 20 | **Fallback / adapter telemetry** | Low | Both | llvmpipe / wrong adapter risk | Keep `CANDLE_REQUIRE_*_TEST_DEVICE` and name checks |

## CUDA kernel surface (reference)

| Kernel file | Primary ops |
| --- | --- |
| `affine.cu` | scale-and-shift |
| `unary.cu` | unary math, `ucopy_*`, `uelu_*`, `upowf_*` |
| `binary.cu` | arithmetic, min/max, compares → u8 |
| `ternary.cu` | `where` |
| `cast.cu` | `to_dtype` |
| `fill.cu` | `const_set`, `copy2d`, fill |
| `reduce.cu` | sum/min/max/arg*, softmax lineage |
| `conv.cu` | im2col, pool, upsample, transpose conv |
| `indexing.cu` | gather, scatter, index_select/add |
| `sort.cu` | argsort |
| `quantized.cu` + `mmq_gguf/` + `mmvq_gguf.cu` | dequant, MMQ/MMVQ |
| `moe/*.cu` | expert GEMM |

CUDA matmul for dense floats uses **cuBLAS** `gemm_strided_batched` (bf16/f16/f32/f64), not a `.cu` file.

### CUDA `BackendStorage` methods (implemented)

`try_clone`, `const_set`, `to_dtype`, `affine`, `powf`, `elu`, `reduce_op`, `cmp`, `unary_impl`, `binary_impl`, `to_cpu_storage`, `where_cond`, `conv1d`, `conv_transpose1d`, `conv2d`, `conv_transpose2d`, `avg_pool2d`, `max_pool2d`, `upsample_nearest2d`, `upsample_bilinear2d`, `index_select`, `gather`, `scatter_set`, `scatter_add_set`, `index_add`, `matmul`, `copy2d`, `copy_strided_src`

### CUDA Map1 / Map2 dtypes (typical)

`u8`, `u32`, `i16`, `i32`, `i64`, `bf16`, `f16`, `f32`, `f64`, `f8e4m3` (with op-specific exclusions: e.g. matmul floats only; conv rejects integer kernels; rand only f32/f64).

## Fallback policy (current truth)

```
Dense BackendStorage (Vulkan/WGPU):
  native GPU path → Ok
  unsupported dtype/layout → Err(UnsupportedDTypeForOp | Msg)
  (generally no silent to_cpu_storage for compute)

Quantized (quantized/mod.rs):
  try native → on Err: record_*_cpu_fallback + CPU matmul/dequant path
```

Use `CANDLE_DEBUG_GPU_FALLBACK=1` and `*_cpu_fallback_count()` in native-required tests.

## How to re-audit

```bash
# Static method-surface check (no GPU needed)
python scripts/backend_parity_audit.py

# JSON machine summary
python scripts/backend_parity_audit.py --json

# Include curated status histogram from manifest
python scripts/backend_parity_audit.py --manifest docs/backend-parity-manifest.json
```

Exit code **1** if any CUDA-required op is immediately-unsupported / missing on **both** Vulkan and WebGPU.

## Definition of done (CUDA parity)

1. Every CUDA-implemented `BackendStorage` / `BackendDevice` op has a **GPU-resident** path for CUDA’s **core model dtypes** (`f32`, `f16`, `bf16`, `u8`, `u32`, `i64` as applicable).
2. Native-required smokes and `gpu_model_matrix_*` report **fallback count 0** on target hardware.
3. Quantized GGUF public formats: dequant + matmul + row gather without CPU recovery.
4. Performance: no multi-× regressions on dense decoder and ConvMixer vs CUDA without an explicit documented exception.
5. Manifest + this doc updated when a row moves `partial` → `native`.
