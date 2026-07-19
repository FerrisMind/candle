# Portable WebGPU evidence

## Result of this verification stage

**Browser / WASM e2e harness: not executed.**

Reasons:

- No automated browser WebGPU driver was run in this environment.
- `candle-wasm-tests` currently exercise CPU quantized paths, not portable WebGPU GPU ops.

## Policy applied

Per verification rules, **no op may use `portable_webgpu_status: Verified`** without a real browser/WASM WebGPU run for that op.

Manifest statuses for portable profile remain:

- `Native` — WGSL path exists and is not claimed browser-verified;
- `GPUEmulated` — GPU composition / f32 hub;
- `UnsupportedBySpecification` — e.g. F64 arithmetic in base WGSL;
- `CudaSpecific` — NCCL etc.

## How to produce Verified later

1. Build WASM with `wgpu` portable features.
2. Run WebGPU adapter tests in Chrome/Edge with WebGPU enabled.
3. Attach adapter name, browser version, OS, and pass/fail per op to this directory.
4. Only then promote individual ops to `Verified` with `portable_tests` pointing at those artifacts.
