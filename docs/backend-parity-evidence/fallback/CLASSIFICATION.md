# Host transfer vs compute fallback classification

## Intentional host bridges (allowed)

| Path | Purpose |
| --- | --- |
| `to_cpu_storage` / `storage_from_cpu_*` | User-visible H2D/D2H API |
| `transfer_to_device` | Cross-device bounce via host |
| `read_buffer` / `map_async` in D2H implementation of `to_cpu_storage` | Required for download |
| Host quantize encode matching CUDA pack | Encode-only; GPU dequant/qmatmul after |

## Forbidden if used as compute

| Pattern | Status in this stage |
| --- | --- |
| Host recompute after GPU error, return as GPU storage | Not observed on runtime audit |
| `record_*_cpu_fallback` increments during parity matrix | 0 with `CANDLE_STRICT_NO_CPU_FALLBACK=1` |
| Full tensor `to_vec` inside compute trait methods | Static audit: mostly upload/download helpers and shape `.to_vec()` (dims) |

## Runtime instrumentation

- `CANDLE_STRICT_NO_CPU_FALLBACK=1` panics if `record_vulkan_cpu_fallback` / `record_wgpu_cpu_fallback` / host-compute recorders fire.
- Ran on: `gpu_parity_matrix_tests`, `fallback_runtime_audit`.

## Static audit

See `static_audit.json` (heuristic; many hits are false positives on dim `.to_vec()` and bridge helpers).
