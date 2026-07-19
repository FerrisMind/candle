# Independent verification review

**Role:** reviewer separate from implementation narrative  
**Evidence root:** `docs/backend-parity-evidence/`  
**Date:** 2026-07-19  

## 1. Audit script vs stated hard rules

| Rule | Met? | Notes |
| --- | --- | --- |
| Fail if CUDA op missing on Vulkan **or** WGPU alone | **Yes** | `run_audit` appends separate failures |
| Presence ≠ implementation | **Yes** | `usable()` requires GPU-ish work, rejects immediate unsupported |
| Dtype/layout-dependent unsupported | **Partial** | Emitted as **warnings**, not hard fail — acceptable if classified in manifest |
| Verified → test **function** refs | **Yes** | Bare file alone fails; `path::function` required |
| Generic file over-share limit | **Yes** | `MAX_GENERIC_FILE_SHARE` |
| Self-tests | **Yes** | `--self-test` + `scripts/tests/test_backend_parity_audit.py` (8 tests OK) |

## 2. Manifest test references

Spot-check: Verified ops point at `gpu_parity_matrix_tests.rs::parity_*` or specific smoke helpers. Strict audit log exit 0.

**Caveat:** One function (e.g. `parity_indexing_matrix`) can still cover several ops — that is intentional clustering, not a bare-file free-for-all. Acceptable if the function actually exercises those ops (scatter/index path does).

## 3. Raw logs vs STATUS.md

| STATUS claim | Supported by logs? |
| --- | --- |
| Implemented Yes | Yes — audit + matrix |
| Tested locally Yes | Yes — matrix 2/2 pass; fallback audit PASS |
| Reproducibly verified Partial | Yes — COMMANDS.txt + full logs; ASCII path note honest |
| SLO verified **No** | Yes — `slo_report.json` has large `violations[]` |
| Portable verified **No** | Yes — portable/README |
| Production-ready **No** | Consistent with gates |

## 4. SLO methodology

- Strengths: same GPU, release build, warm-up, sync, raw CSV retained, automated ratios.
- Weaknesses: **no p95/variance columns in CSV** (helpers removed); many ops **sync-only** (latency-dominated); **no** LLM prefill/decode/TTFT/VRAM/e2e transformer.
- Conclusion: **SLO verified = No** is correct. Matmul batch geomean looking healthy does **not** clear release SLO while gather/scatter/conv show 10–20×.

## 5. Fallback evidence

- Runtime: `fallback_runtime_audit` + STRICT env on matrix — counters 0.
- Static: heuristic noisy; classification doc separates bridges.
- Gap: not every compute path is dynamically instrumented (only explicit `record_*` hooks). Honest as **partial** dynamic coverage.

## 6. Portable

- No browser/WASM GPU e2e. Status **No** is mandatory.

## 7. Agreement with STATUS flags

| Flag | Agree? |
| --- | --- |
| Implemented | Agree Yes |
| Tested locally | Agree Yes |
| Reproducibly verified | Agree Partial |
| SLO verified | Agree **No** |
| Portable verified | Agree **No** |
| Production-ready | Agree **No** |

## 8. Open gaps / findings

1. **Critical for production claim:** SLO violations on gather/scatter/conv/small-sync kernels — not waived with profiled objective limits in evidence pack.
2. **Major:** Numerical harness is **F32-primary** (22 cases); full F16/BF16/F64 × layout matrix not fully measured here.
3. **Major:** No e2e transformer / LLM metrics.
4. **Minor:** Static fallback audit false positives; refine allowlist.
5. **Minor:** Working tree may include post-14c3fe5a verification tooling; pin evidence `environment.json` `actual_commit` when committing.

## Verdict

Evidence pack is **usable and mostly honest**. Do **not** declare Production-ready or full SLO verified. Implement+local test bars are supported by artifacts.
