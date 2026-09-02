## Running BERT with Candle and WASM

Here, we provide an example of how to run Bert using a Candle-compiled WASM binary and runtime, with a **CPU / WebGPU / auto** device switch.

### Feature flags

| Feature | Effect |
|---|---|
| *(default)* | CPU path only. `DeviceMode::Wgpu` fails; `auto` resolves to CPU. |
| `wgpu` | Enables portable WebGPU via `candle-wasm-device-select` + `candle` wgpu backend (`BROWSER_WEBGPU` on wasm32). |

Build the WASM library:

```bash
# CPU-only
./build-lib.sh

# With portable WebGPU (Chrome / browsers that expose navigator.gpu)
./build-lib.sh wgpu
# or: ./build-lib.sh --features wgpu
```

This bundles the library under `./build`. Import it inside the WebWorker:

```js
import init, { Model } from "./build/m.js";
```

### Device switch (JS worker protocol)

- UI sends `{ command: "setDevice", deviceMode: "cpu"|"wgpu"|"auto", …model URLs }` to reload weights on the resolved device.
- Inference uses `{ command: "run", … }` **without** `deviceMode`.
- Cache key is `${modelID}:${resolvedDevice}` (not `deviceMode`), so `auto→cpu` and explicit `cpu` share one instance.
- Explicit `wgpu` init/runtime failure → `deviceError` with `previousDeviceMode` / `requestedDeviceMode`; UI rolls back the radio.
- `auto` without WebGPU → `ready` with `resolvedDevice: "cpu"` (not an error).

Default UI mode is **auto**.

### Limitations / portable WebGPU notes

- This pilot targets **portable WebGPU** (browser `BROWSER_WEBGPU`), not native Vulkan/DX12 `wgpu` and not the native Vulkan Candle backend. Do not treat a native-desktop wgpu success as proof of browser portability.
- There is **no silent CPU-as-GPU fallback**: failed explicit `wgpu` is an error; only `auto` may choose CPU when GPU init fails.
- F64 / some advanced shader features are capability-gated in candle-core; absence of `SHADER_F64` must not panic device creation.
- Device ownership lives in the **worker**; `run` must not change device.
- Slice-1 differential tolerances (when comparing CPU vs wgpu embeddings): rtol `1e-5`, atol `1e-6` for F32 — not claimed Verified in this README alone.
- Manual smoke: Chrome with WebGPU enabled (auto/wgpu → `resolvedDevice: "wgpu"` + adapter name); Chrome with WebGPU disabled or unavailable (auto → cpu; explicit wgpu → `deviceError` + UI rollback).

### Preview

All model assets are fetched from the Hub. Serve locally:

```bash
python -m http.server
```

Then open `http://localhost:8000/lib-example.html`.
