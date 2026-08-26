## Running Whisper Examples

Here, we provide two examples of how to run Whisper using a Candle-compiled WASM binary and runtimes.

### Pure Rust UI

To build and test the UI made in Rust you will need [Trunk](https://trunkrs.dev/#install)
From the `candle-wasm-examples/whisper` directory run:

Download assets:

```bash
# mel filters
wget -c https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors
# Model and tokenizer tiny.en
wget -c https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors -P whisper-tiny.en 
wget -c https://huggingface.co/openai/whisper-tiny.en/raw/main/tokenizer.json -P whisper-tiny.en
wget -c https://huggingface.co/openai/whisper-tiny.en/raw/main/config.json -P whisper-tiny.en
# model and tokenizer tiny multilanguage
wget -c https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors -P whisper-tiny
wget -c https://huggingface.co/openai/whisper-tiny/raw/main/tokenizer.json -P whisper-tiny
wget -c https://huggingface.co/openai/whisper-tiny/raw/main/config.json -P whisper-tiny

#quantized 
wget -c https://huggingface.co/lmz/candle-whisper/resolve/main/model-tiny-en-q80.gguf -P quantized
wget -c https://huggingface.co/lmz/candle-whisper/raw/main/tokenizer-tiny-en.json -P quantized
wget -c https://huggingface.co/lmz/candle-whisper/raw/main/config-tiny-en.json -P quantized



# Audio samples
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_gb0.wav -P audios
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_a13.wav -P audios
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_gb1.wav -P audios
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_hp0.wav -P audios
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav -P audios
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_mm0.wav -P audios

```

Run hot reload server:

```bash
trunk serve --release --public-url / --port 8080
```

### Vanilla JS and WebWorkers

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
# Portable WebGPU (default; larger wasm). `wgpu`/`auto` resolve to WebGPU when available.
./build-lib.sh

# CPU-only (smaller wasm; `wgpu` fails and `auto` resolves to `cpu`)
./build-lib.sh cpu
```

This will bundle the library under `./build` and we can import it inside our WebWorker like a normal JS module:

```js
import init, { Decoder } from "./build/m.js";
```

The full example can be found under `./lib-example.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/lib-example.html` in your browser (secure context / localhost required for WebGPU).

#### Device switch (`cpu` | `wgpu` | `auto`)

The JS demo exposes a device radio (default **auto**):

| Mode | Behavior |
| --- | --- |
| `cpu` | Load on CPU |
| `wgpu` | Require portable WebGPU; on failure the UI rolls back to the previous mode |
| `auto` | Try WebGPU, else explicit CPU (`resolvedDevice: "cpu"` — not a silent GPU claim) |

Protocol (UI → worker):

- `{ command: "setDevice", deviceMode, /* model URLs */ }` — loads/reloads weights on the chosen device
- `{ command: "run", audioURL }` — inference only; **no** `deviceMode`

Worker reports `resolvedDevice` and `adapterName` on `ready` / `complete`. Cache key is `modelID:resolvedDevice` (so `auto→cpu` and `cpu` share one instance).

#### Limitations (portable WebGPU)

- Browser **portable WebGPU** only (`BROWSER_WEBGPU`); do not treat native wgpu backends as portable proof.
- F64 shader ops are capability-gated; F32 path when the adapter lacks F64.
- `wgpu::Device` lives on the **worker** thread; the UI never owns the GPU device.
- The `wgpu` feature increases wasm size by roughly 2–4 MB; it is the **default** build.
- Build **without** `wgpu` (i.e. `./build-lib.sh cpu`): Auto behaves as CPU; explicit `wgpu` fails with `deviceError`.

#### Manual smoke (Chrome)

1. Build with `./build-lib.sh` (wgpu default), serve over `http://localhost…`, open `lib-example.html`.
2. With WebGPU available: leave **Auto**, confirm status shows `resolved: wgpu` and an adapter name; transcribe a sample.
3. Switch to **CPU**, confirm reload and `resolved: cpu`; run again.
4. Switch to **WebGPU**; on success keep wgpu; if init fails, radio rolls back and status shows the error.
5. Disable WebGPU (or use a browser without it): **Auto** should become `resolved: cpu` without error; explicit **WebGPU** should `deviceError` + rollback.
