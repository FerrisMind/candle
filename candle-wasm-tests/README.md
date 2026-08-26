# candle-wasm-tests

WASM browser tests for Candle (quantized kernels + device resolve smoke).

## Run

```bash
# CPU / auto resolve (no wgpu feature)
RUST_LOG=wasm_bindgen_test_runner wasm-pack test --chrome --headless

# With optional portable WebGPU resolve (requires candle-core wgpu wasm create path)
wasm-pack test --chrome --headless -- --features wgpu

# If your local Chrome version does not match wasm-pack's auto-downloaded driver,
# pin the matching chromedriver explicitly:
wasm-pack test --chrome --headless --chromedriver "<path-to-chromedriver.exe>"

# Firefox with an explicit geckodriver:
wasm-pack test --firefox --headless --geckodriver "<path-to-geckodriver.exe>"
```

Or headed:

```bash
wasm-pack test --chrome
```

If you get an "invalid session id" failure in headless mode, check that logs and
it may well be that your ChromeDriver is not at the same version as your
browser.

Requires: `wasm-pack`, Chrome, and `wasm32-unknown-unknown` Rust target.

## Device resolve tests (`tests/device_select_tests.rs`)

| Test | Expectation |
|------|-------------|
| `resolve_cpu` | `ResolvedKind::Cpu`, never fails |
| `resolve_auto_never_panics` | `Ok(Cpu \| Wgpu)` — no panic; auto without GPU → explicit CPU |
| `resolve_wgpu` | `Ok` or `Err` — must not panic |
| `cpu_f32_matmul_golden` | Minimal CPU F32 golden when browser GPU differential is unavailable |

## Differential tolerances (slice 1)

From design spec §8 — for CPU ↔ portable WebGPU comparisons when evidence exists:

| Dtype / path | rtol | atol |
|--------------|------|------|
| F32 ops | `1e-5` | `1e-6` |
| Paths that fall back when F64 unavailable | `1e-3` | `1e-4` (document if used) |

These tolerances are for **future** differential harnesses. Passing resolve smoke
tests here does **not** claim `portable_webgpu_status: Verified` (that requires
browser-matrix GPU evidence under `docs/backend-parity-evidence/portable/`).

## Features

- default: resolve tests against `candle-wasm-device-select` without wgpu
- `wgpu`: enables `candle-wasm-device-select/wgpu` + `candle/wgpu` for Auto/Wgpu GPU attempts
