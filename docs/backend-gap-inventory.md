# Candle Vulkan / WGPU Backend Gap Inventory

Inventory of CPU-fallback paths, incomplete ops, layout restrictions, and CUDA parity gaps.
Generated from static analysis of `candle-core` and `candle-nn` (2026-07-17).

**Scope files**
- `candle-core/src/vulkan_backend.rs`
- `candle-core/src/wgpu_backend.rs`
- `candle-core/src/storage.rs`
- `candle-core/src/quantized/mod.rs`
- `candle-core/src/custom_op.rs`
- `candle-nn/src/{ops,rotary_emb,moe}.rs`

Debug counters: set `CANDLE_DEBUG_GPU_FALLBACK` to log each recorded fallback.

---

## 1. `record_*_cpu_fallback` call sites

Fallback counters live in `storage.rs` and are **only wired for quantized paths** today.
Regular dense ops do **not** auto-fallback to CPU on Vulkan/WGPU (they return errors).

### Recording infrastructure

| API | Location |
|-----|----------|
| `record_wgpu_cpu_fallback` | `candle-core/src/storage.rs:14-18` |
| `record_vulkan_cpu_fallback` | `candle-core/src/storage.rs:21-25` |
| `wgpu_cpu_fallback_count` / `reset_*` | `storage.rs:28-41` |
| Predicate | `quantized/mod.rs:77-86` via `should_quantized_backend_fallback` |
| Message classifier | `quantized/mod.rs:63-75` (`is_backend_not_implemented_msg`) |

**Fallback is taken when error is:**
- `Error::UnsupportedDTypeForOp(..)`
- `Error::Msg` matching:
  - `"{backend} backend op … not implemented"`
  - `"no {backend} implementation for …"`
  - `"{backend} … shader … not generated"`
  - `"{backend} … supports up to rank-4 tensors"`
  - dimension / tmp / workgroup overflow strings containing backend name

### Call sites (all in `quantized/mod.rs`)

| Backend | Path | Lines | Trigger |
|---------|------|-------|---------|
| **WGPU** | `QWgpuStorage::dequantize` (raw F32/F16/BF16) | 312-316 | `quantized_raw_float_dequantize_f32` fails → CPU dequant → re-upload |
| **WGPU** | `QWgpuStorage::dequantize` (block quant) | 344-348 | `quantized_index_select_f32` fails → CPU dequant → re-upload |
| **WGPU** | `QWgpuStorage::fwd` (qmatmul) | 492-495 | `quantized_matmul` fails → `fwd_cpu_fallback` |
| **WGPU** | `QWgpuStorage::index_select_f32` | 515-518 | GPU get-rows fails → `index_select_f32_cpu_fallback` |
| **Vulkan** | `QVulkanStorage::dequantize` | 632-635 | `quantized_dequantize_f32` fails → CPU dequant → re-upload |
| **Vulkan** | `QVulkanStorage::fwd` | 786-789 | `quantized_matmul` fails → `fwd_cpu_fallback` |
| **Vulkan** | `QVulkanStorage::index_select_f32` | 809-812 | GPU get-rows fails → CPU index_select |
| **Vulkan** | MoE indirect | 1673 | `should_quantized_backend_fallback` swallows error and falls through to **dense GPU** dequant+matmul path (not counted via `record_*`) |

### What `fwd_cpu_fallback` actually does (hidden CPU compute)

WGPU (`quantized/mod.rs:391-427`) / Vulkan (`:679+` mirror):
1. `storage.to_cpu_storage()` (download activations)
2. `to_cpu_quantized()?.matmul_t(...)` (CPU quantized matmul)
3. `storage_from_cpu_storage` (upload result)

`index_select_f32_cpu_fallback` similarly downloads ids + full dequant to CPU, then re-uploads.

**Special case — Q8K:** no GPU matmul; always `q8k_fwd_via_dense_gpu` = dequantize on GPU then dense matmul (`mod.rs:353-377` wgpu, `:641-665` vulkan). Not recorded as CPU fallback unless dequant itself fails.

---

## 2. `UnsupportedDTypeForOp` / “not implemented” by op family

Helper:
```rust
// vulkan_backend.rs:816-818, wgpu_backend.rs:767-769
fn unsupported(op) -> Error::Msg("{backend} backend op {op} not implemented")
```

### 2.1 Upload / download / storage

| Op | Vulkan | WGPU |
|----|--------|------|
| Upload F6E2M3 / F6E3M2 / F4 | 1229-1234 | 826-831 |
| Download unsupported dtypes | 1252 | 849 |
| `to_dtype` missing pair | 2882 (`to_dtype {src}->{dst} not implemented`) | 2714 `unsupported("to_dtype")` |
| `const_set` / fill | 2904 | 4557 |

### 2.2 Unary / binary / affine

| Family | Vulkan | WGPU | Notes |
|--------|--------|------|-------|
| Unary dtypes | 2755: only **F32/F16** native; else Unsupported | shader path limited; BF16 often emulated | BF16 unary via f32 materialize on Vulkan |
| Unary ops missing SPIR-V | 2781 `unsupported("unary")` — e.g. cos/sin/sqr/sqrt only F32 | similar | |
| Binary float | 2809: F32/F16; ints via opcode shader; **BF16 not native** | 3295 | BF16 binary via f32 (`vulkan:3304`, `wgpu:5077`) |
| Binary int ops | 2831 `unsupported("binary int")` | 2545-2674 | |
| Binary min/max | 3915 `unsupported("binary min/max")` | — | |
| Affine / powf / elu / clamp | 7593, 7609, 7623/7645, 7746 | 9186, 9217, 4045 | elu mostly F32 (+ F16 upcast) |
| Unary BackendStorage | 7797 | 9478/9535/9549 generic `"op"` | |

### 2.3 Compare / where / cast

| Op | Vulkan lines | WGPU lines |
|----|--------------|------------|
| cmp | 3599-3698 | 4111-4114 |
| where_cond | 3717-3807 | 4219-4228 |
| bf16 cmp/where | 3341, 3378-3381 | 5114, 5151-5154 |
| emulated cast | — | 4671, 4834, 5424 (f64) |

### 2.4 Reduce / sort / cumsum / softmax

| Op | Vulkan | WGPU | Restriction |
|----|--------|------|-------------|
| reduce dtype | 7665 | 9262 | F32/F16/BF16 + ints; not F64 native on Vulkan reduce |
| int reduce | 4273, 4312, 4351 | 1628 | |
| argsort | 4565-4587; hardware: 4602, 4620 | 5896-5918 | last-dim only API |
| cumsum | 4768 (F32 only in kernel) | 6117 | last-dim; F16/BF16 via upcast |
| softmax | 4879 | 6258 | F32 kernel; F16/BF16 upcast |

### 2.5 Norm / rope / sigmoid

| Op | Vulkan | WGPU |
|----|--------|------|
| rms_norm | 5102 | 6617-6620 |
| rope / positions | 4929-4932 | 6491-6500 |
| sigmoid | 5357 | 6896 |

### 2.6 Indexing

| Op | Vulkan | WGPU |
|----|--------|------|
| index ids dtype | 5380 | 6922 |
| index_select | 5404 | 6946 |
| gather | 5511-5515 | 7073-7082 |
| scatter_set | 5614-5636 | 7203-7222 |
| scatter_add | 5720-5746 | 7318-7342 |

### 2.7 Matmul / conv / pool / upsample

| Op | Vulkan | WGPU |
|----|--------|------|
| matmul dtype | 5855, 6101 | 7459, 7642 |
| matmul rank/batch | 5862-5882 `unsupported` | 7466-7493 `unsupported` |
| im2col / conv1d / conv2d | 6150-6333 | 7761, 7871 |
| conv_transpose1d/2d | 6380, 6492 | 7959-8090 |
| upsample | 6599, 6632 | 8219, 8252 |
| pool2d | 6678 | 8298 |

### 2.8 Quantized

| Op | Vulkan | WGPU |
|----|--------|------|
| quant dtype stem | 841, 858 bail | 1419 bail (IQ etc.) |
| dequantize | 2923-2938 `unsupported` (missing Q8_1, IQ*, …) | via index_select path |
| quantized matmul dtype | 6841 | 8659; needs **SHADER_F16** 8657 |
| quantized matmul rank | 6846 | 8672 |
| quantized matvec | — | 8837 |
| q index_select | — | 8433 (needs shader-f16), 8437 |
| indexed MoE | 7246-7306 | (uses higher-level path) |
| quantize_q8_1 | 1113 | — |

**Vulkan quantized stem support** (`vulkan_quantized_stem` 828-842): Q4_0/1, Q5_0/1, Q8_0/1, Q2K–Q6K. **No Q8K, no IQ\*, no TQ\*.**

**WGPU** (`wgpu_quantized_dtype` 1406-1420): same + Q8K enum mapping, but Q8K matmul still dense-dequant path.

### 2.9 Copy / strided

| Op | Vulkan | WGPU |
|----|--------|------|
| copy / copy_strided | 3994, 8277-8282 | 10050-10056, 10073 |
| copy2d | 8320 | 10113 |
| emulated u8 copy | — | 5355 |

---

## 3. Rank / layout restrictions (vs CUDA)

CUDA generally accepts arbitrary rank (within kernel launch limits) and strided layouts via offset/stride parameters. Vulkan/WGPU often **materialize** or **reject**.

### Rank ≤ 4 (hard in several kernels)

| Location | Behavior |
|----------|----------|
| `vulkan_backend.rs:2633-2634` `dims4_ggml` | bail: `"vulkan backend supports up to rank-4 tensors for this op"` — used by cumsum, softmax params, sum_rows, etc. |
| Rank>4 handling elsewhere | `materialize_rank_gt4_compact` (3230+), matmul flatten (5858-5873), binary/where compact (3615+, 3750+) — **extra copies**, not full native rank |
| WGPU `dims4` family | similar rank-4 packing; matmul rank>4 flatten at 7462+ |

CUDA: no such rank-4 GGML param packing limit on core ops.

### Contiguous-only (or materialize-first)

Many paths require `is_contiguous() && start_offset()==0`, else copy:

| Op area | Vulkan | WGPU |
|---------|--------|------|
| Softmax | materialize if non-contig | 6179-6183 |
| RMS/LayerNorm | 5117 area / contig checks | 6622-6623, 6765-6767 |
| Rope positions | — | 6502 |
| Scatter set/add | last-dim needs contiguous dst | 7192, 7309 |
| Quantized matmul activations | 6864-6874 | 8686+ |
| Argsort / cumsum | materialize non-contig | 5923, 6122 |
| Affine (some) | — | 3931, 3965 max binding |

CUDA kernels typically take strides and run in-place on views.

### Last-dim-only native kernels

Public Tensor API often still works for other dims via **permute + materialize + last-dim kernel**:

| Op | Last-dim kernel | Non-last handling |
|----|-----------------|-------------------|
| reduce extrema / sum | `run_*_last_dim` | `run_reduce_non_last_dim` permutes (Vulkan 4499+, WGPU 5835+) |
| argsort | `argsort_last_dim` only | Tensor API is last-dim (`sort.rs`) — same as CUDA API surface |
| cumsum | `cumsum_last_dim` | API last-dim only (both backends) |
| gather/scatter specialized | `run_*_last_dim_f32` | other dims may use slower/generic paths |

### Other hardware gates CUDA does not have

| Gate | Location | Effect |
|------|----------|--------|
| Argsort non-power-of-two needs robust buffer access | Vulkan 4601-4604 | hard error / fallback N/A for dense |
| Large argsort needs `vulkan_memory_model` | 4619-4620 | |
| WGPU quantized matmul needs `SHADER_F16` | 8650-8657, 8834 | CPU fallback on quant path |
| Integer-dot / subgroup features for MMVQ | Vulkan q8_1 rhs path | quality/perf path selection |

---

## 4. CUDA custom ops missing `vulkan_fwd` / `wgpu_fwd`

### Trait defaults (`custom_op.rs`)

| Trait | CUDA | Metal | WGPU | Vulkan |
|-------|------|-------|------|--------|
| `CustomOp1/2/3` | default error | default error | default error | default error |
| **`InplaceOp1/2/3`** | `cuda_fwd` | `metal_fwd` | **NONE** | **NONE** |

`storage.rs:462-522` hard-errors on WGPU/Vulkan for all inplace ops:
```text
"no wgpu implementation for {name}"
"no vulkan implementation for {name}"
```
Traits themselves (`custom_op.rs:327-428`) only declare `cpu_fwd` + `cuda_fwd` + `metal_fwd` — **no `wgpu_fwd`/`vulkan_fwd` methods at all**.

### candle-nn custom ops matrix

| Op | File | cuda | metal | wgpu | vulkan | Gap |
|----|------|------|-------|------|--------|-----|
| Sigmoid | ops.rs | ✓ | ✓ | ✓ | ✓ | — |
| SoftmaxLastDim | ops.rs | ✓ | ✓ | ✓ | ✓ | — |
| RmsNorm | ops.rs | ✓ | ✓ | ✓ | ✓ | — |
| LayerNorm | ops.rs | ✓ | ✓ | ✓ | ✓ | — |
| Sdpa | ops.rs | metal+ | metal | ✓ flash_attn | ✓ flash_attn | CUDA uses separate flash-attn crates |
| RotaryEmbI (interleaved) | rotary_emb.rs | ✓ | — | **✗** | **✗** | **CUDA-only GPU** |
| RotaryEmb (contiguous) | rotary_emb.rs | ✓ | — | **✗** | **✗** | **CUDA-only GPU** |
| RotaryEmbThd | rotary_emb.rs | ✓ | — | **✗** | **✗** | **CUDA-only GPU** |
| RotaryEmbGgml | rotary_emb.rs | — | — | ✓ | ✓ | CUDA N/A (ggml path) |
| MoE gemm | moe.rs | ✓ (feature cuda) | — | **✗** | **✗** | CUDA-only; non-CUDA uses `moe_gemm_fallback` |
| ArgSort | sort.rs (core) | ✓ | ✓ | ✓ | ✓ | — |
| QTensor CustomOp1 | quantized/mod.rs | ✓ | ✓ | ✓ | ✓ | fallbacks above |

### External CUDA-only

| Crate | Notes |
|-------|-------|
| `candle-flash-attn` | `cuda_fwd` only — no wgpu/vulkan |
| `candle-flash-attn-v3` | same |
| `candle-examples/.../llama_multiprocess` | cuda custom op |

**candle-transformers:** no direct `cuda_fwd` implementations found; depends on candle-nn/core.

---

## 5. Production paths: download → CPU compute → upload

### Explicit / always-on CPU (not just fallback)

| Path | Location | Severity |
|------|----------|----------|
| **Quantize on GPU storage** | `QWgpuStorage::quantize` / `quantize_imatrix` (`mod.rs:271-283`), `QVulkanStorage` (`:599-610`) | Always `to_cpu_storage` then `quantize_from_cpu` (CPU block encoding) then re-upload U8 |
| **Quantize from CPU helper** | `quantize_from_cpu` 247-268 | Always builds on `Device::Cpu.qzeros` |
| **Q8K matmul** | `q8k_fwd_via_dense_gpu` | Full dequant on GPU (or CPU if dequant fails) then dense matmul — not true quantized GEMM |
| **Device transfer between same backend instances** | `VulkanDevice::transfer_to_device` 1727-1729; `WgpuDevice::transfer_to_device` 905-907 | CPU bounce buffer |

### Conditional CPU fallbacks (recorded)

See §1 — qmatmul / dequant / get-rows.

### Not production (tests only)

`vulkan_backend.rs` ~8868–9254+: extensive `to_cpu_storage()` in `#[test] #[ignore]` conv probes — **not** runtime inference paths.

### Dense ops: no silent CPU fallback

Unlike quantized paths, `Storage::unary_impl` / binary / matmul for WGPU/Vulkan (`storage.rs:539-546`) **propagate errors** — they do not download and compute on CPU. Failures surface as `UnsupportedDTypeForOp` or `"… not implemented"`.

---

## 6. InplaceOp* support for wgpu/vulkan

**Answer: not supported.**

1. Traits lack methods (`custom_op.rs:327-428`).
2. Dispatch rejects storage variants (`storage.rs:467-472`, `489-492`, `515-520`).
3. Any `tensor.inplace_op1/2/3` on WGPU/Vulkan returns an error immediately.

**Fix shape:** add `wgpu_fwd`/`vulkan_fwd` to `InplaceOp1/2/3` with default errors; wire `storage.rs`; implement for any in-tree inplace ops that need GPU (e.g. test `Elu` in `custom_op_tests.rs`).

---

## 7. Top 15 prioritized fixes (CUDA parity blockers)

| # | Gap | Why it blocks | Suggested approach |
|---|-----|---------------|-------------------|
| **1** | **Quantized matmul CPU fallback** for unsupported dtypes / rank / missing shaders | Dominates GGUF LLM latency; silent CPU matmul under load | Expand SPIR-V/WGSL MMQ/MMVQ coverage (Q8K native, all Qk); fail-fast option via env to catch regressions; add CI metric on `*_cpu_fallback_count` |
| **2** | **WGPU quantized path requires `SHADER_F16`** (`wgpu_backend.rs:8650+`) | Many WebGPU/DX12/Vulkan-via-wgpu adapters lack native f16 → full CPU fallback | F32-only quantized shaders or f16 polyfill in WGSL; feature-detect and choose shader variant |
| **3** | **Always-CPU quantize** (`quantize` → `to_cpu_storage`) | Blocks on-device QAT / dynamic quant; PCIe thrash | Port block quantize kernels (start with Q8_0/Q4_0) to Vulkan/WGPU compute |
| **4** | **Q8K = dense dequant + matmul** | Common intermediate / training dtype; 2× bandwidth | Native Q8K matvec/matmul kernels (CUDA already has dequant + better paths) |
| **5** | **RotaryEmb / RotaryEmbI / RotaryEmbThd lack wgpu/vulkan_fwd** | Llama/Mistral/etc. rope custom ops fall back to default error or force CPU tensor path | Implement `wgpu_fwd`/`vulkan_fwd` calling existing `ggml_rope` or new rope kernels; mirror CUDA launch geometry |
| **6** | **MoE CUDA-only** (`moe.rs`) | Mixture-of-experts models cannot use fused GPU path on Vulkan/WGPU | Port `moe_gemm` to Vulkan compute (indexed MoE already partially exists at `quantized_indexed_moe_f32`); wire dense MoE similarly |
| **7** | **BF16/F16 matmul via F32 upcast** (Vulkan 5831-5848) | 2× memory + cast overhead vs CUDA tensor cores / native half GEMM | Native f16/bf16 matmul shaders (Vulkan cooperative matrix if available; WGPU f16 where `SHADER_F16`) |
| **8** | **Unary/binary BF16 not native** (materialize_to_f32 paths) | Transformer residual streams often BF16 | Generate bf16 SPIR-V/WGSL for add/mul/unary core set; keep f32 fallback only for rare ops |
| **9** | **InplaceOp* no wgpu/vulkan** | User extensions + any in-place fused ops hard-fail | Extend traits + storage dispatch; start with no-op defaults, then implement high-value ops |
| **10** | **Rank-4 GGML packing (`dims4_ggml`)** | Rank≥5 tensors error or force compact copies | Generalize param structs to N-D (prefix product for batch) like CUDA reduce/matmul |
| **11** | **Argsort hardware gates** (robust buffer / memory model) | Sort used in top-k sampling paths | CPU-free fallback shader without robust access (bounds clamps); or document device requirements |
| **12** | **Conv / pool dtype + layout** (mostly F32; contig materialize) | Vision models (Whisper encoder, SAM, YOLO wasm) | Match CUDA: f16 conv where possible; fused im2col+gemm staying on device (Vulkan already GPU but heavy materialize) |
| **13** | **Flash-attn quality gap** | Sdpa uses simplified `flash_attn` on Vulkan/WGPU vs CUDA flash-attn crates | Expand head dims, softcap, mask variants; parity tests vs `candle-flash-attn` |
| **14** | **Missing integer / F64 coverage on hot ops** | Indexing, reduce, binary on I64/F64 inconsistently supported | Close `UnsupportedDTypeForOp` holes for I64 index path and F64 reduce (or document intentional limits) |
| **15** | **No dense-op CPU fallback telemetry** | Silent error vs quantized recording asymmetry confuses debugging | Optional `CANDLE_GPU_DENSE_FALLBACK=cpu` for parity debugging only; keep default fail-fast for production |

---

## 8. Quick reference: where to look first

```
record_*_cpu_fallback  → quantized/mod.rs only
unsupported()          → vulkan ~16 sites, wgpu ~15 sites
Inplace gap            → custom_op.rs + storage.rs:462-522
Rope CUDA-only         → candle-nn/src/rotary_emb.rs RotaryEmb{,I,Thd}
MoE CUDA-only          → candle-nn/src/moe.rs
Quant always CPU       → Q*Storage::quantize* → to_cpu_storage
Rank-4 limit           → dims4_ggml / dims4
SHADER_F16 wall        → wgpu quantized_matmul / index_select
```

### Suggested validation workflow

1. Run GGUF inference with `CANDLE_DEBUG_GPU_FALLBACK=1`; assert `vulkan_cpu_fallback_count()==0` / wgpu counterpart for target models.
2. Extend `gpu_parity_matrix_tests` / `cuda-wgpu-vulkan-parity-matrix.md` with rows for rope, MoE, Q8K, BF16 matmul.
3. Prefer fixing GPU kernels over expanding CPU fallback surface.
