# Verification status matrix (commit under test)

**Pinned target:** `14c3fe5ab0fd08b0212561abd459d159bf45ae65`  
**Working tree:** verification tooling + evidence may be newer; logs include the commit hash in `environment.json` at run time.

## Separate flags (do not merge)

| Flag | Value | Evidence |
| --- | --- | --- |
| **Implemented** | **Yes** (CUDA-required BackendStorage surface on Vulkan and Native WebGPU) | `logs/parity_audit_strict.log` exit 0; static presence all yes |
| **Tested locally** | **Yes** (parity matrix + smokes + fallback audit) | `logs/gpu_parity_matrix_tests.log` 2 passed; `fallback/fallback_runtime_audit.log` PASS |
| **Reproducibly verified** | **Partial** | Commands in `logs/COMMANDS.txt`; full stdout/stderr captured; requires ASCII CUDA path on this host |
| **SLO verified** | **No** | `bench/slo_report.json` lists **many** critical-kernel limit violations (gather/scatter/conv/sync latency); matmul **batch** geomean OK (Vulkan ~0.77×, WGPU ~1.14× CUDA) but that is **not** full release SLO |
| **Portable verified** | **No** | `portable/README.md` — browser/WASM e2e not run; no portable `Verified` promotion |
| **Production-ready** | **No** | Blocked by: SLO violations list, incomplete numerical dtype matrix (F32 primary), no LLM/e2e, portable not verified |

## What passed

1. Strict static audit (per-backend fail; Verified needs `path::function`).
2. CUDA differential parity matrix under required device envs + RTX 3060 name check.
3. Strict no-CPU-fallback runtime audit (dense+quant) counters remain 0.
4. Numerical CPU reference harness (F32 critical ops, U8 exact, special abs) for Vulkan and WGPU.
5. Microbench raw CSV + automated SLO ratio table.

## Explicit non-claims

- Not claiming production readiness.
- Not claiming full CUDA parity for every dtype/layout without residual.
- Not claiming portable WebGPU browser readiness.
- Not claiming release SLO for all kernels (see violations in `bench/slo_report.json`).

## Independent review

See `independent_review.md`.
