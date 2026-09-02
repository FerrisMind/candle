// Candle Whisper WASM worker — UI ↔ worker protocol (spec §6).
//
// Assumed wasm exports from sibling Task 3 (adjust import/call if names differ):
//   Decoder.load_with_device(
//     weights, tokenizer, mel_filters, config,
//     quantized, is_multilingual, timestamps, task, language,
//     deviceMode /* "cpu" | "wgpu" | "auto" */
//   ) → Promise<Decoder>
//   decoder.resolved_device() → "cpu" | "wgpu"
//   decoder.adapter_name() → string | undefined/null
// If Task 3 ships a free function instead, use:
//   import init, { Decoder, load_with_device } from "./build/m.js";
//   const decoder = await load_with_device(...);

import init, { Decoder } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "whisper-candle-cache-v2";
  try {
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      const data = await cachedResponse.arrayBuffer();
      return new Uint8Array(data);
    }
  } catch (_) {
    /* Cache API unavailable (opaque origin / quota) — fall through to network */
  }

  const res = await fetch(url, { cache: "force-cache", mode: "cors" });
  if (!res.ok) {
    throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  }

  // Read the FULL body before touching the Cache API. Cloning a large streaming
  // response (151 MB model.safetensors) and consuming one branch via cache.put()
  // can truncate the other branch in Firefox, yielding "incomplete metadata,
  // file not fully covered". Verify against Content-Length to fail loud instead.
  let buf = await res.arrayBuffer();
  let declared = res.headers.get("content-length");
  if (declared != null && Number(declared) !== buf.byteLength) {
    // One retry bypassing the HTTP cache (stale/partial cached body).
    const retry = await fetch(url, { cache: "reload", mode: "cors" });
    if (!retry.ok) {
      throw new Error(
        `fetch ${url} failed on retry: ${retry.status} ${retry.statusText}`
      );
    }
    buf = await retry.arrayBuffer();
    declared = retry.headers.get("content-length");
  }
  if (declared != null && Number(declared) !== buf.byteLength) {
    throw new Error(
      `fetch ${url} truncated: got ${buf.byteLength} bytes, expected ${declared}`
    );
  }

  try {
    const cache = await caches.open(cacheName);
    await cache.put(url, new Response(buf, { headers: res.headers }));
  } catch (_) {
    /* caching is best-effort */
  }
  return new Uint8Array(buf);
}

function errMessage(e) {
  if (e == null) return "Unknown error";
  if (typeof e === "string") return e;
  if (e.message) return e.message;
  return String(e);
}

class Whisper {
  /** @type {Record<string, { decoder: any, resolvedDevice: string, adapterName?: string, modelID: string }>} */
  static instance = {};
  static activeKey = null;
  static activeDeviceMode = "auto";
  static wasmReady = false;

  static cacheKey(modelID, resolvedDevice) {
    return `${modelID}:${resolvedDevice}`;
  }

  static async ensureWasm() {
    if (!this.wasmReady) {
      await init();
      this.wasmReady = true;
    }
  }

  /**
   * Load or reuse a decoder for the requested deviceMode.
   * Cache key is modelID:resolvedDevice (not deviceMode), so auto→cpu and cpu share.
   */
  static async setDevice(params) {
    const {
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL,
      configURL,
      quantized,
      is_multilingual,
      timestamps,
      task,
      language,
      deviceMode,
    } = params;

    const previousDeviceMode = this.activeDeviceMode ?? "auto";
    const isReload = this.activeKey != null;

    self.postMessage({
      status: isReload ? "reloading" : "loading",
      message: isReload
        ? "Reloading model on new device"
        : "Loading Model",
    });

    await this.ensureWasm();

    // Fast path when the resolved backend is known a priori.
    if (deviceMode === "cpu" || deviceMode === "wgpu") {
      const key = this.cacheKey(modelID, deviceMode);
      const cached = this.instance[key];
      if (cached) {
        this.activeDeviceMode = deviceMode;
        this.activeKey = key;
        self.postMessage({
          status: "ready",
          message: "Model Already Loaded",
          resolvedDevice: cached.resolvedDevice,
          adapterName: cached.adapterName,
        });
        return cached;
      }
    }

    try {
      const [
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8,
        configArrayU8,
      ] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
        fetchArrayBuffer(mel_filtersURL),
        fetchArrayBuffer(configURL),
      ]);

      // Assumed async factory — see file header if export name differs.
      const decoder = await Decoder.load_with_device(
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8,
        configArrayU8,
        quantized,
        is_multilingual,
        timestamps,
        task,
        language,
        deviceMode
      );

      const resolvedDevice = decoder.resolved_device();
      const adapterName = decoder.adapter_name
        ? decoder.adapter_name() ?? undefined
        : undefined;
      const key = this.cacheKey(modelID, resolvedDevice);

      // Truthfulness guard: an explicit `wgpu` selection must never silently
      // resolve to (or reuse) a CPU backend. If the built wasm lacks the wgpu
      // feature, or WebGPU init degrades to CPU, surface a deviceError instead
      // of claiming success and running inference on CPU while the UI shows
      // `wgpu`. `auto` may legitimately resolve to `cpu`; only `wgpu` is strict.
      if (deviceMode === "wgpu" && resolvedDevice !== "wgpu") {
        if (decoder.free) {
          try {
            decoder.free();
          } catch (_) {
            /* ignore */
          }
        }
        const err = new Error(
          `WebGPU requested but resolved to ${resolvedDevice}; WebGPU unavailable in this build`
        );
        self.postMessage({
          status: "deviceError",
          error: err.message,
          previousDeviceMode,
          requestedDeviceMode: deviceMode,
          phase: "init",
        });
        throw err;
      }

      // Reuse existing instance for this resolved device (e.g. auto→cpu after cpu).
      if (this.instance[key]) {
        if (decoder.free) {
          try {
            decoder.free();
          } catch (_) {
            /* ignore */
          }
        }
        this.activeDeviceMode = deviceMode;
        this.activeKey = key;
        const cached = this.instance[key];
        self.postMessage({
          status: "ready",
          message: "Model Already Loaded",
          resolvedDevice: cached.resolvedDevice,
          adapterName: cached.adapterName,
        });
        return cached;
      }

      this.instance[key] = {
        decoder,
        resolvedDevice,
        adapterName,
        modelID,
      };
      this.activeDeviceMode = deviceMode;
      this.activeKey = key;

      self.postMessage({
        status: "ready",
        message: "Model ready",
        resolvedDevice,
        adapterName,
      });
      return this.instance[key];
    } catch (e) {
      self.postMessage({
        status: "deviceError",
        error: errMessage(e),
        previousDeviceMode,
        requestedDeviceMode: deviceMode,
        phase: "init",
      });
      throw e;
    }
  }

  static getActive() {
    if (!this.activeKey || !this.instance[this.activeKey]) {
      throw new Error("No model loaded; call setDevice first");
    }
    return this.instance[this.activeKey];
  }

  /**
   * Transient inference: fully unload the active model + device so the (wgpu) device
   * is destroyed and VRAM/RAM are released after each decode. `free()` is the
   * wasm-bindgen-generated destructor on the `Decoder`, which drops the inner model
   * (weights/tensors) and device, firing `Drop for WgpuInner` → `device.destroy()`.
   * Safe to call when nothing is loaded (no-op).
   */
  static disposeActive() {
    const key = this.activeKey;
    if (key && this.instance[key]) {
      const entry = this.instance[key];
      if (entry.decoder && entry.decoder.free) {
        try {
          entry.decoder.free();
        } catch (_) {
          /* ignore */
        }
      }
      delete this.instance[key];
    }
    this.activeKey = null;
    this.activeDeviceMode = "auto";
  }
}

self.addEventListener("message", async (event) => {
  const data = event.data;
  const { command } = data;

  if (command === "setDevice") {
    try {
      const modelID = data.modelID;
      const quantized =
        data.quantized ?? (modelID?.includes("quantized") ?? false);
      const is_multilingual =
        data.is_multilingual ?? (modelID?.includes("multilingual") ?? false);
      await Whisper.setDevice({
        weightsURL: data.weightsURL,
        modelID,
        tokenizerURL: data.tokenizerURL,
        mel_filtersURL: data.mel_filtersURL,
        configURL: data.configURL,
        quantized,
        is_multilingual,
        timestamps: data.timestamps ?? true,
        task: data.task ?? null,
        language: data.language ?? null,
        deviceMode: data.deviceMode ?? "auto",
      });
    } catch (_) {
      // deviceError already posted
    }
    return;
  }

  if (command === "run") {
    // Device is fixed by setDevice — do not read deviceMode from run.
    try {
      self.postMessage({ status: "decoding", message: "Starting Decoder" });
      const entry = Whisper.getActive();

      self.postMessage({ status: "decoding", message: "Loading Audio" });
      let audioArrayU8;
      try {
        audioArrayU8 = await fetchArrayBuffer(data.audioURL);
      } catch (e) {
        self.postMessage({
          status: "error",
          error: errMessage(e),
          phase: "audioDecode",
        });
        return;
      }

      self.postMessage({
        status: "decoding",
        message: "Running Decoder...",
        resolvedDevice: entry.resolvedDevice,
        adapterName: entry.adapterName,
      });

      let segments;
      try {
        try {
          // `decode` is an async wasm-bindgen export → returns a JS Promise. Await it so the
          // wasm autoregressive loop yields to the JS event loop between tokens (wgpu buffer
          // recycle, prevents the 12GB VRAM balloon). Normalize: a sync CPU build or older
          // wasm may still return the JSON string directly.
          let raw = entry.decoder.decode(audioArrayU8);
          if (raw && typeof raw.then === "function") {
            raw = await raw;
          }
          segments = raw;
        } catch (e) {
          self.postMessage({
            status: "error",
            error: errMessage(e),
            phase: "inference",
          });
          return;
        }

        self.postMessage({
          status: "complete",
          message: "complete",
          output: JSON.parse(segments),
          resolvedDevice: entry.resolvedDevice,
          adapterName: entry.adapterName,
        });
      } finally {
        // Transient inference: unload the model + device after every decode so VRAM
        // (wgpu device.destroy()) and RAM are released. `complete`/`error` above are
        // posted BEFORE dispose so the UI still gets its result.
        Whisper.disposeActive();
      }
    } catch (e) {
      self.postMessage({
        status: "error",
        error: errMessage(e),
        phase: "inference",
      });
    }
    return;
  }

  self.postMessage({
    status: "error",
    error: `Unknown command: ${command ?? "(missing)"}`,
    phase: "modelLoad",
  });
});
