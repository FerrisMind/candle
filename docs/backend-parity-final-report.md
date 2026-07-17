# Backend Parity Final Report — Vulkan & WebGPU (three profiles)

**Date:** 2026-07-17  
**Repo:** `D:\Users\ПК\Desktop\candle`  
**Branch:** `feat/vulkan-webgpu-cuda-parity`  
**Baseline commit:** `039a0edc6f60a654d7fef6dc23a755fddadbe930`  
**Author of this pass:** implementer agent (FerrisMind / Grok Build goal)

## 1. Environment (baseline freeze)

Captured in scratch `baseline-env.txt`:

| Item | Value |
|------|--------|
| OS | Windows 10/11 NT 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3060 |
| Driver | 32.0.16.1074 |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| cargo | 1.97.0 |
| CUDA toolkit | 12.6 (nvcc V12.6.20) |
| Vulkan | `vulkaninfo` present |
| Features exercised | `cuda`, `vulkan`, `wgpu` |

Dirty tree at start (preserved, not clobbered): user edits to `wgpu_backend.rs`, `mul_mat_coop.wgsl`, `candle-wgpu-kernels` lib, `.gitignore`; untracked `docs/backend-parity-spec.md`, `scripts/export-candle-refs.py`.

## 2. Profiles (not merged)

| Profile | Meaning | Evidence surface |
|---------|---------|------------------|
| **Native Vulkan** | Direct ash/SPIR-V backend | `candle-core` + `candle-vulkan-kernels` |
| **Native WebGPU** | `wgpu` on desktop (Vulkan/DX12/Metal) | `candle-core` + `candle-wgpu-kernels` |
| **Portable WebGPU** | Browser/WASM WGSL capability subset | Classified separately in manifest; **not** claimed from native `wgpu` desktop runs |

Machine-readable source of truth: [`docs/backend-parity-manifest.json`](backend-parity-manifest.json) (`schema_version: 2`).

Status vocabulary (only): `Native`, `Optimized`, `GPUEmulated`, `UnsupportedBySpecification`, `UnsupportedByHardware`, `CudaSpecific`, `Missing`, `Verified`.

Validator: `python scripts/backend_parity_audit.py` (enforces CUDA surface presence + allowed statuses + three profile fields).

## 3. Inventory / classification summary

54 ops after upgrade (BackendStorage CUDA surface + device + extended model ops including `layer_norm`/`sdpa` + meta rows).

Approximate status mix (Native Vulkan / Native WebGPU / Portable):

- **Verified** — core smoke-certified ops on **native** profiles only (unary/binary, reduce, cast matrix, upload, layer_norm, sdpa, …)
- **GPUEmulated** — on-device f32 materialize / composition (BF16 paths, Q8K dense dequant matmul, rank>4 compact, rope_i tensor path, half argsort keys)
- **Optimized** — `matmul` with release microbench SLO evidence (batch20)
- **UnsupportedByHardware** — `dtype::f8e4m3` on native Vulkan/WebGPU (no portable FP8 ALU)
- **UnsupportedBySpecification** — FP8 on portable WGSL
- **CudaSpecific** — NCCL, external `candle-flash-attn` crates
- **Missing** — none remaining after classification
- **Portable profile** uses **Native/GPUEmulated** (not `Verified`) unless a browser/WASM suite exists — native `wgpu` smokes are **not** portable Verified

**100% coverage** here means every CUDA backend-independent capability is listed and classified — not that portable WGSL natively implements non-portable features.

## 4. Fallback / silent cast policy

### Shipped compute paths

- Dense Vulkan/WGPU: **fail-fast** (no silent CPU compute).
- Quantized qmatmul / dequant / get-rows: **GPU-only** recovery via `dense_qmatmul_via_gpu` / dequant+dense index (no host matmul).
- `record_*_cpu_fallback` remain in `storage.rs` but have **no active call sites** in quantized hot paths.
- Weight `Q*Storage::quantize` host encode matches **CUDA baseline** (CUDA also downloads F32 and quantizes on CPU in `quantized/cuda.rs`).

### Intentional host APIs (allowed)

- `to_cpu_storage` / `Tensor::to_device(Cpu)`
- `data()` / `to_cpu_quantized` export
- Cross-backend `transfer_to_device` bounce

Audit log: scratch `fallback-audit.log`.

## 5. Correctness evidence

| Suite | Command | Result |
|-------|---------|--------|
| Vulkan smoke | `cargo test -p candle-core --features vulkan --test backend_smoke_tests` | **48 passed** |
| WGPU smoke | `cargo test -p candle-core --features wgpu --test backend_smoke_tests` | **40 passed** (after cfg stub fix) |
| Metamorphic Vulkan | `cargo test -p candle-core --features vulkan --test gpu_metamorphic_tests` | **1 passed** |
| Shader validation | `gpu_shader_validation_tests` (vulkan) | **1 passed** (+ ignored utility) |
| candle-nn ops Vulkan | `cargo test -p candle-nn --features vulkan --test ops` | softmax/rms/rope/sdpa exercised; rope/layer_norm/sdpa un-ignored to match skip pattern |
| candle-nn wgpu | `cargo test -p candle-nn --features wgpu` | **ok** (exit 0) |
| Parity audit | `python scripts/backend_parity_audit.py` | **OK** schema v2 |

Logs under implementer scratch: `correctness-tests-*.log`, `parity-manifest-check.log`, `launch-*.log`.

### Differential / numeric policy

- Integer/bool/index: exact match in smoke matrices.
- Float: existing backend smoke tolerances; CUDA↔CPU is the primary oracle in matrix tests — Vulkan/WGPU not auto-loosened.
- Edge families covered in smoke: rank5, strided const_set, BF16 unary/binary/cmp/where/matmul/reduce, int reduce, quantized Q8_1/Q8K, argsort multi-dtype.

## 6. Performance SLO (same GPU: RTX 3060)

### 6.1 End-to-end (primary release SLO)

Command:

```text
CANDLE_GPU_MODEL_CASE_FILTER=vision_convmixer_case
cargo test -p candle-transformers --release --features {cuda,vulkan,wgpu} \
  --test gpu_model_matrix gpu_model_matrix_{cuda,vulkan,wgpu} -- --ignored --exact --nocapture
```

| Backend | Wall time | Fallback | vs CUDA |
|---------|-----------|----------|---------|
| CUDA (warm) | **3.29s** | 0 | 1.00× |
| Vulkan | **3.03s** | 0 | **0.92× PASS** (≤1.15) |
| Native WebGPU | **5.71s** | 0 | **1.74×** (exceeds 1.30) |

- **Vulkan e2e SLO (≤15% slower):** **PASS**
- **Native WebGPU e2e SLO (≤30% slower):** **MISS** — documented objective limit below

### 6.2 Hot-path microbench (`batch20`)

```text
cargo run -p candle-core --release --features "cuda,vulkan,wgpu" --example backend_parity_microbench -- --suite
```

| Metric | Value | SLO | Verdict |
|--------|-------|-----|---------|
| Vulkan geo-mean V/C | **0.791×** | ≤ 1.20 | **PASS** |
| Vulkan max critical | **1.46×** | ≤ 1.50 | **PASS** |
| Native WebGPU max | **1.66×** | ≤ 2.00 | **PASS** |

### Documented objective limits

1. **WebGPU e2e 1.74× CUDA** on ConvMixer: non-contiguous binary ops (BatchNorm broadcast chains) require a map-based GPU fence after each op on this Windows/wgpu stack; without it logits explode (~1e18). Contiguous elementwise keeps the fast path. Cheaper dependency tracking is future work.
2. **Per-op `sync` microbench mode** on tiny kernels is launch/flush dominated — not e2e primary.
3. **Portable WebGPU** has no single CUDA-relative %; browser matrix separate.
4. Stretch goals (Vulkan 10%, WebGPU 20%/10%) non-blocking.

Logs: `{SCRATCH}/e2e-vision-*.log`, `perf-slo.md`, `perf-slo-e2e.md`, `launch-microbench.log`.

## 7. Portable WebGPU capability matrix (checked, not browser-timed)

| Family | Portable status | Notes |
|--------|-----------------|-------|
| Storage/upload/sync/copy | Verified/Native | Standard WebGPU buffers |
| f32 unary/binary/reduce/matmul/conv | Verified/GPUEmulated | WGSL compute |
| f16 | Native if `shader-f16`; else GPUEmulated f32 | Must feature-detect |
| bf16 | GPUEmulated | No native BF16 in portable WGSL |
| f64 compute | GPUEmulated or limit | lo/hi words where implemented; not native f64 ALU |
| f8e4m3 | UnsupportedBySpecification | No portable FP8 |
| Cooperative matrix / subgroup fast paths | Not portable | Native-only; generic path required |
| Quantized GGUF | GPUEmulated/Verified subset | Prefer f32 dequant composition without SHADER_F16 hard require |
| NCCL / flash-attn crates | CudaSpecific | N/A |

WASM/browser timing was **not** run in this environment (honest escape hatch). Structural portable classification is in the manifest `portable_webgpu_status` + `portable_notes`.

## 8. Code changes in this completion pass

1. **`docs/backend-parity-manifest.json`** — schema v2, three profiles, allowed status vocabulary, meta rows (f8e4m3, edges, NCCL, flash-attn).
2. **`scripts/backend_parity_audit.py`** — validates schema v2 statuses, Verified⇒tests, Optimized⇒bench, CUDA ops ⊆ manifest.
3. **`candle-core/tests/backend_smoke_tests.rs`** — Vulkan q8_1 routing helpers `cfg(feature="vulkan")` with pure-wgpu stub so `wgpu`-only builds compile; dead_code allow on helper.
4. **`candle-nn/tests/ops.rs`** — un-ignore Vulkan rope/layer_norm/sdpa (use device_or_skip like softmax).
5. **`docs/backend-parity-final-report.md`** — this report.
6. User dirty wgpu matmul comment edits preserved.

## 9. Licenses / sources

- Candle: Apache-2.0 / MIT dual (repo root).
- Kernel lineage notes: llama.cpp Vulkan/WebGPU shader ideas referenced historically; candle-owned SPIR-V/WGSL under repo licenses.
- Refs under `candle_refs` (ash, wgpu, cudarc, half, …) — dependency licenses unchanged; no new third-party code bulk-imported in this pass.
- Specs: Vulkan 1.3, WebGPU/WGSL (portable limits for f64/f8/coop).

## 10. Independent review

See `docs/backend-parity-review.md` (subagent review of this pass).

**Initial review:** 2 Critical (portable `Verified` without portable tests; `layer_norm`/`sdpa` missing from manifest).

**Remediation in same pass:**

1. All portable `Verified` → `Native` with notes that browser/WASM suite was not run.
2. Manifest rows added for `layer_norm` and `sdpa` (Vulkan tests un-ignored and green).
3. `conv2d`/`quantized_matmul` downgraded from `Optimized` without matching microbench; only `matmul` remains `Optimized`.
4. `argsort_last_dim` half keys classified `GPUEmulated`.

Re-run audit after remediation: **OK**. Residual major notes in review.md (stale docs wording) are non-blocking.

## 11. Residual work (non-blocking / documented)

- Browser WASM portable timing matrix per browser/OS/adapter.
- Optional: promote more `GPUEmulated` half paths to native half kernels where profiled.
- Optional: GPU weight quantize beyond CUDA’s own host encode (not required for CUDA parity).
- Full vision ConvMixer e2e re-bench on release GPU CI.
- Property suite (`gpu_properties_vulkan`) is `#[ignore]` heavy — run with `--ignored` in long CI.

## 12. Completion checklist mapping

| Criterion | Status |
|-----------|--------|
| Parity inventory three profiles, allowed statuses | **Met** (54 ops, schema v2) |
| Required ops implemented or proven limited | **Met** (f8/NCCL/flash classified) |
| No hidden CPU compute on dense/quant hot path | **Met** (runtime fallback audit counters = 0) |
| Correctness + CUDA differential matrix | **Met** (`gpu_parity_matrix_{vulkan,wgpu}` PASS) |
| Perf SLO or documented limits | **Met** (Vulkan e2e PASS; WebGPU e2e limit documented with fence evidence) |
| CI hygiene + report + review | **Met** (clippy vulkan/wgpu `-D warnings` PASS after fence fix; smokes; review C1/C2 remediated) |

### Critical code fix this pass

`candle-core/src/wgpu_backend.rs`:
- Always retain storage Arcs for in-flight batches (no `skip_retain` warm path).
- One compute pass per pending dispatch (write→read order).
- Defer size-class pool recycle until after GPU drain.
- Non-contig binary: eager dyn-uniform write + minimal map fence (correctness for BN/ConvMixer).
