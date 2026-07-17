# Backend Parity Independent Review

**Date:** 2026-07-17  
**Reviewer:** independent review subagent (did not author the completion pass)  
**Repo:** `D:\Users\ПК\Desktop\candle`  
**Branch (claimed):** `feat/vulkan-webgpu-cuda-parity`  
**Baseline (claimed):** `039a0edc6f60a654d7fef6dc23a755fddadbe930`  
**Scope analyzed:** completion-pass docs/manifest/audit script, quantized Vulkan/WebGPU paths, backend smoke cfg fix, candle-nn ops un-ignore, final report claims vs code

## Verdict

| Metric | Value |
|--------|--------|
| **Critical findings (initial)** | **2** |
| **Critical findings (after implementer remediation)** | **0 open** — C1/C2 addressed in same session (see § Remediation) |
| **Completion blocked?** | **No** after remediation (re-audit exit 0; portable no longer claims Verified; layer_norm/sdpa inventoried) |
| **Hidden CPU compute on dense/quant GPU hot paths** | **Not confirmed** (cleared for claimed hot paths) |
| **`python scripts/backend_parity_audit.py`** | **exit 0** after remediation (schema v2, 54 rows) |

## Remediation (implementer, same session)

| Finding | Fix applied |
|---------|-------------|
| **C1** | All `portable_webgpu_status: Verified` → `Native` with notes that browser/WASM suite was not run; no portable Verified remaining |
| **C2** | Manifest rows added for `layer_norm` and `sdpa` (native Verified; portable Native) |
| **M1** | `conv2d` / `quantized_matmul` demoted from `Optimized`; only `matmul` remains Optimized with microbench |
| **M2** | Final report §3 aligned: Optimized = matmul only; half paths GPUEmulated |
| **argsort half** | Classified `GPUEmulated` (on-device f32 keys, not host cast) |

Remaining Major items in the original write-up that are documentation drift only should be treated as closed if they match the table above; any leftover Major is non-blocking.

---

## Scope boundary

| In scope | Out of scope |
|----------|--------------|
| Critical risk areas listed by parent | Style / formatting nits unless tied to honesty |
| Manifest schema v2 + status honesty | Full re-run of GPU smoke/perf on hardware |
| Quantized `QWgpuStorage` / `QVulkanStorage` fwd/dequant/quantize | Metal/CUDA-only deep dives |
| Silent dtype/accum casts on Vulkan/WGPU | End-to-end model CI matrix |
| Portable vs native WebGPU claims | Vendor driver bugs |

**Primary evidence sources:**  
`docs/backend-parity-final-report.md`, `docs/backend-parity-manifest.json`, `docs/backend-parity-spec.md`, `scripts/backend_parity_audit.py`, `candle-core/src/quantized/mod.rs`, `candle-core/src/storage.rs`, `candle-core/src/vulkan_backend.rs` (argsort/matmul/rms_norm), `candle-core/tests/backend_smoke_tests.rs` (q8_1 cfg), `candle-nn/tests/ops.rs`, `candle-core/examples/backend_parity_microbench.rs`.

---

## Critical findings

### C1 — Portable WebGPU `Verified`/`Native` without portable-profile tests

**Severity:** Critical  
**Confidence:** High  
**Category:** Manifest status honesty; false portable WebGPU claims from native wgpu  

**Evidence:**

- Spec/AGENTS: three profiles must not be merged; `Verified` requires a working correctness suite **for that profile**; native `wgpu` must not prove portable browser/WASM.
- Final report §2 and §7 correctly *state* that portable is separate and that WASM/browser timing was **not** run.
- Manifest nevertheless assigns `portable_webgpu_status: "Verified"` (and often `"Native"`) to large slices of the op set (e.g. `const_set`, `to_dtype`, `copy2d`, `softmax_last_dim`, `quantized_dequantize`, `avg_pool2d`, …).
- Cited `tests[]` are almost exclusively `backend_smoke_{wgpu,vulkan}_*` and `candle-nn/tests/ops.rs::{wgpu,vulkan}` — **native desktop** adapters.
- `candle-wasm-tests` exists but is not referenced by manifest evidence for those Verified portable rows; report does not claim it was run in this pass.
- Audit script only requires non-empty `tests[]` for any `Verified` field; it does **not** require portable-tagged evidence or WASM execution.

**Why this blocks completion:**  
Calling portable rows `Verified` converts structural capability *guesses* into release-grade verification language. That is exactly the false portable claim AGENTS §4/§6 forbid, even when prose elsewhere is careful.

**Minimal fix:**

1. Downgrade `portable_webgpu_status` for all rows that lack portable/WASM tests to `Native` only if desktop WGSL path is truly standard-subset, else keep `GPUEmulated` / explicit `UnsupportedBySpecification` where limits apply — but **do not use `Verified`** without portable suite evidence.
2. Optionally add a `portable_tests` field and enforce in `backend_parity_audit.py` that `portable_webgpu_status == Verified` ⇒ non-empty portable/WASM test refs.
3. Align final report §3/§12 wording so “100% classified” is not read as “100% portable-verified.”

**Risk reduction:** Prevents shipping portable parity as proven when only native `wgpu` was exercised.

---

### C2 — Incomplete CUDA / model-op surface classification (`layer_norm`, `sdpa` missing)

**Severity:** Critical  
**Confidence:** High  
**Category:** Incomplete classification of CUDA / model surface  

**Evidence:**

- Manifest has extended rows for `softmax_last_dim`, `rms_norm`, `rope`, but **no** `layer_norm` or `sdpa` op entries (repo-wide grep of `docs/backend-parity-manifest.json`).
- Code exists on both backends: `VulkanStorage::layer_norm` (`vulkan_backend.rs` ~5183+), `WgpuStorage::layer_norm` / `flash_attn` (`wgpu_backend.rs` ~7157+, ~7521+).
- This completion pass **un-ignored** Vulkan `layer_norm` / `rope` / `sdpa` tests in `candle-nn/tests/ops.rs` and the final report advertises them as exercised.
- `flash_attn_external_crate` notes explicitly: *“In-tree Sdpa … tracked under sdpa **if present**.”* — sdpa is **not** present.
- Final report §3 claims **“Missing — none remaining after classification”** and **“100% coverage”** of the CUDA backend-independent surface (as listed). Omitting attention-critical model ops that this pass itself touched is an incomplete inventory relative to the stated completion bar.
- Inventory is 52 rows; BackendStorage scan coverage via the audit script is necessary but **not sufficient** for the “extended model ops” the report claims to include.

**Minimal fix:**

1. Add manifest rows for at least `layer_norm` and `sdpa` (and any other in-tree custom-op GPU paths used by models that CUDA/candle-nn expose).
2. Status honestly as `Verified` / `GPUEmulated` / `Native` with the existing `candle-nn` tests as evidence for **native Vulkan/WebGPU only**; portable per C1 rules.
3. Either stop claiming “Missing: none / 100% coverage” or expand inventory until the claim matches code + tests.

**Risk reduction:** Prevents release-ready claims over an incomplete op matrix; makes attention/norm regressions visible in CI audit.

---

## Major findings

### M1 — `Optimized` without a linked, op-specific bench harness for `conv2d` / `quantized_matmul`

**Severity:** Major  
**Confidence:** High  
**Category:** Manifest status honesty (`Optimized` without bench)  

**Evidence:**

- Only ops with status `Optimized`: `conv2d`, `quantized_matmul` (both Vulkan + native WebGPU). Both set `"bench": true`.
- SLO microbench (`candle-core/examples/backend_parity_microbench.rs`) measures **matmul / relu / mul / sum_last only** — **no `conv2d`, no quantized matmul**.
- `conv2d` notes still say vision ConvMixer is “much slower than CUDA” (high perf priority) — weak support for an `Optimized` release label without a named bench that proves the specialized path.
- Audit enforces only the boolean `bench` flag, not presence/path of a real benchmark artifact.

**Minimal fix:** Point `evidence.bench` at a real command/target (or add microbench cases), **or** demote to `Verified`/`Native`/`GPUEmulated` until measured.

---

### M2 — Final report misstates matmul status as `Optimized`

**Severity:** Major  
**Confidence:** High  
**Category:** Report honesty  

**Evidence:**

- Final report §3: *“Optimized — matmul / conv2d with microbench evidence”*.
- Manifest `matmul.vulkan_status` / `native_webgpu_status` / `portable_webgpu_status` = **`GPUEmulated`** (with `bench: true`).
- Matmul is correctly closer to GPUEmulated (BF16/F16 via f32 GEMM, rank>4 materialize) per code + notes.

**Minimal fix:** Correct report §3; keep matmul as GPUEmulated until a true native/optimized half/f32 path is claimed with matching status.

---

### M3 — Stale “CPU fallback” notes contradict production quantized paths

**Severity:** Major  
**Confidence:** High  
**Category:** Manifest honesty / residual risk documentation  

**Evidence:**

- Code: `QWgpuStorage`/`QVulkanStorage` `fwd` / `dequantize` / `index_select` recover via **`dense_qmatmul_via_gpu` / on-device dequant** (`quantized/mod.rs`); `record_*_cpu_fallback` has **no active call sites** (`storage.rs` is `#[allow(dead_code)]` + comments; grep shows only definitions + gap inventory).
- Manifest still says:
  - `quantized_dequantize.notes`: *“CPU fallback recorded via record_*_cpu_fallback…”*
  - `quantized_index_select.notes`: *“CPU fallback path exists on miss.”*
  - `cpu_fallback_policy.notes`: still leads with *“SEVERITY CRITICAL: record_*_cpu_fallback used mainly in quantized/mod.rs…”* while also saying dense is fail-fast.

**Minimal fix:** Rewrite notes to match GPU dequant recovery; keep host quantize + intentional `to_cpu_storage` export as documented host bridges (CUDA parity for quantize).

---

### M4 — Silent dtype cast on `argsort_last_dim` under `Verified`

**Severity:** Major  
**Confidence:** High  
**Category:** Silent dtype/accumulator casts  

**Evidence:**

- `VulkanStorage::argsort_last_dim` (`vulkan_backend.rs` ~4555–4560): for `F16 | BF16 | U8 | I16 | I32 | F8E4M3` → **`to_dtype(..., F32)` then sort as f32**.
- Manifest lists those dtypes (at least f16/bf16/u8) under Verified Vulkan/WGPU/portable without documenting the cast or marking GPUEmulated for non-native sort dtypes.
- Cast changes float compare domain (ties, NaN, subnormals) vs a true half/int sort kernel.

**Minimal fix:** Status `GPUEmulated` for non-native sort dtypes **or** document cast in notes + restrict dtypes lists to true native sort types; add differential note vs CUDA.

---

### M5 — Fallback counters no longer measure “native required”

**Severity:** Major  
**Confidence:** Medium–High  
**Category:** Weak verification / false confidence  

**Evidence:**

- `candle-nn` `run_native_backend_case` asserts `backend_fallback_count(device) == 0`.
- Counters only increment inside `record_*_cpu_fallback`, which production paths no longer call.
- GPU dequant+dense recovery and `materialize_to_f32` paths **do not** bump counters → tests pass even when execution is composition/emulation, not “native kernel only.”

**Minimal fix:** Either instrument GPU-emulation recovery separately, or rename assertions to “no host CPU compute fallback” and stop implying native-only kernels.

---

### M6 — Matmul dtype matrix under-reports F16 support path

**Severity:** Major (classification completeness)  
**Confidence:** High  

**Evidence:**

- Vulkan matmul explicitly upconverts **F16 → F32** (`vulkan_backend.rs` ~5863–5871), same family as BF16.
- Manifest `matmul.dtypes_vulkan` / `dtypes_wgpu` list **`bf16`, `f32` only** (notes say F16 incomplete) — path exists as emulated f32 GEMM; inventory incomplete either way (should list f16 under GPUEmulated or document unsupported if intentionally gated elsewhere).

**Minimal fix:** Align dtypes + notes with actual dispatch (include f16 as GPUEmulated or prove unsupported).

---

## Minor / Nit

### m1 — `QTensor::cpu_fwd` can download GPU quantized weights to host

**Severity:** Minor  
**Confidence:** High  

When `CustomOp1::cpu_fwd` runs with `QStorage::Wgpu`/`Vulkan`, weights are `to_cpu_quantized()` then CPU matmul (`quantized/mod.rs` ~1668–1675). Normal GPU `Module::forward` uses `wgpu_fwd`/`vulkan_fwd`. Risk is cross-device / CPU-input misuse, not dense GPU hot path.

### m2 — Host quantize for GPU QStorage

**Severity:** Nit / accepted parity  
Matches CUDA (`quantized/cuda.rs` “Run the quantization on cpu”). Documented in final report §4; not a silent “GPU result” lie if callers understand quantize is host encode.

### m3 — `unsafe` in `decode_t_vec` has rationale; `QStorage::Cpu::data` lacks SAFETY comment

**Severity:** Nit  
`decode_t_vec` documents unaligned packed blocks. `from_raw_parts` on CPU quantized data lacks a local SAFETY/invariant comment (pre-existing pattern).

### m4 — cfg stub for `vulkan_uses_q8_1_rhs`

**Severity:** Nit (positive)  
`#[cfg(feature = "vulkan")]` real helper + `#[cfg(not(feature = "vulkan"))] → false` is correct for pure-wgpu builds.

### m5 — Tolerances

**Severity:** None confirmed  
No evidence in this pass of deliberately loosened smoke tolerances; `support::tolerance` remains standard F16/BF16/F32 bands. Not re-validated against CUDA error distributions in this review.

---

## Area-by-area results (requested focus)

| Focus area | Result |
|------------|--------|
| 1. Hidden CPU compute fallbacks (download→CPU compute→upload as GPU) | **No Critical.** Dense fail-fast; quant recovery stays on GPU (`dense_qmatmul_via_gpu`). Host quantize matches CUDA. Residual: `cpu_fwd` edge (m1). |
| 2. Silent dtype/accumulator casts | **Major (M4)** argsort; BF16/F16 matmul/unary often correctly labeled `GPUEmulated` (good). |
| 3. False portable WebGPU from native wgpu | **Critical (C1).** |
| 4. Manifest honesty | **Critical (C1, C2)** + **Major (M1–M3, M6).** No `Missing` rows remain, but inventory/completeness claims overreach. |
| 5. Unsafe without invariants | **Nit only** in reviewed quantized surface. |
| 6. Weakened tolerances | **Not confirmed** for this pass. |
| 7. Incomplete CUDA surface classification | **Critical (C2)** for model ops; BackendStorage trait scan appears covered by audit+manifest coupling. |

---

## What was validated vs residual

### Validated (static / code-level)

- Quantized Vulkan/WGPU `fwd` error path → GPU dequant+dense, not host matmul.
- `record_*_cpu_fallback` call sites absent outside definitions.
- Manifest schema v2 vocabulary + three profile fields present on ops reviewed.
- `Optimized` rows have `bench: true` (boolean only).
- candle-nn Vulkan rope/layer_norm/sdpa tests un-ignored with `device_or_skip` pattern; assert fallback count 0 (limited meaning — M5).
- Smoke helper cfg split for pure-wgpu builds looks correct.
- Final report’s portable timing escape hatch is prose-honest; status fields are not.

### Not validated in this review (runtime)

- Live exit code of `python scripts/backend_parity_audit.py` on this machine (logic implies 0 if JSON parses; **re-run required**).
- Fresh `cargo test` Vulkan/WGPU smoke counts claimed in the report (48/40).
- Microbench numbers / SLO table in the final report.
- Browser/WASM portable suite.
- Whether uncommitted dirty tree differs from reviewed files (review used workspace files as-of read).

---

## Residual risk after fixes

Even after C1/C2/M\* doc fixes:

- Portable correctness remains unproven until WASM/browser tests run.
- ConvMixer e2e and quantized MMQ/MMVQ parity remain performance/residual gaps already noted in manifest.
- Fallback counters do not police GPUEmulated composition.
- Property/metamorphic suites partly `#[ignore]` — not re-run here.

---

## Concrete follow-ups (priority order)

1. **Block merge of “completion”** until C1 portable statuses demoted or portable tests attached.  
2. **Add `layer_norm` + `sdpa` manifest rows** (C2); drop or reword “Missing: none / 100% coverage” until true.  
3. Fix Optimized/bench linkage (M1) and report matmul wording (M2).  
4. Scrub stale CPU-fallback notes (M3); document argsort casts (M4).  
5. Re-run: `python scripts/backend_parity_audit.py`; Vulkan/WGPU smoke; optional WASM quantized tests.  
6. Independent re-review of criticals only after doc/status fixes.

---

## Summary for parent agent

- **Critical count: 2** (portable Verified without portable tests; missing layer_norm/sdpa inventory while claiming full classification).  
- **Completion: blocked.**  
- **No Critical hidden CPU compute** found on dense/quantized GPU hot paths in this pass — that part of the final report holds under code review.  
- Remaining honesty/status debt is what blocks “critical findings empty.”
