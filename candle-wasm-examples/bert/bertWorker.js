// Candle Bert WASM worker — UI ↔ worker protocol (spec §6).
// Mirrors whisperWorker.js: setDevice / run, cache key modelID:resolvedDevice.
//
// Wasm exports:
//   Model.load_with_device(weights, tokenizer, config, deviceMode) → Promise<Model>
//   model.resolved_device() → "cpu" | "wgpu"
//   model.adapter_name() → string | undefined/null
//   model.get_embeddings({ sentences, normalize_embeddings })

import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "bert-candle-cache-v2";
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
  // response (up to ~440 MB model.safetensors) and consuming one branch via
  // cache.put() can truncate the other branch in Firefox, yielding "incomplete
  // metadata, file not fully covered". Verify against Content-Length to fail loud
  // instead.
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

class Bert {
  /** @type {Record<string, { model: any, resolvedDevice: string, adapterName?: string, modelID: string }>} */
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
   * Load or reuse a model for the requested deviceMode.
   * Cache key is modelID:resolvedDevice (not deviceMode), so auto→cpu and cpu share.
   */
  static async setDevice(params) {
    const { weightsURL, tokenizerURL, configURL, modelID, deviceMode } =
      params;

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
      const [weightsArrayU8, tokenizerArrayU8, configArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(configURL),
        ]);

      const model = await Model.load_with_device(
        weightsArrayU8,
        tokenizerArrayU8,
        configArrayU8,
        deviceMode
      );

      const resolvedDevice = model.resolved_device();
      const adapterName = model.adapter_name
        ? model.adapter_name() ?? undefined
        : undefined;
      const key = this.cacheKey(modelID, resolvedDevice);

      // Reuse existing instance for this resolved device (e.g. auto→cpu after cpu).
      if (this.instance[key]) {
        if (model.free) {
          try {
            model.free();
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
        model,
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
   * is destroyed and VRAM/RAM are released after each run. `free()` is the
   * wasm-bindgen-generated destructor on the `Model`, which drops the inner model
   * (weights/tensors) and device, firing `Drop for WgpuInner` → `device.destroy()`.
   * Safe to call when nothing is loaded (no-op).
   */
  static disposeActive() {
    const key = this.activeKey;
    if (key && this.instance[key]) {
      const entry = this.instance[key];
      if (entry.model && entry.model.free) {
        try {
          entry.model.free();
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
      await Bert.setDevice({
        weightsURL: data.weightsURL,
        tokenizerURL: data.tokenizerURL,
        configURL: data.configURL,
        modelID: data.modelID,
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
      const entry = Bert.getActive();
      self.postMessage({
        status: "embedding",
        message: "Calculating Embeddings",
        resolvedDevice: entry.resolvedDevice,
        adapterName: entry.adapterName,
      });

      let output;
      try {
        // `get_embeddings` is an async wasm-bindgen export → returns a JS Promise.
        // Await it so the wgpu async GPU→CPU readback can yield to the event loop.
        // A sync CPU build or older wasm may still return the object directly.
        let raw = entry.model.get_embeddings({
          sentences: data.sentences,
          normalize_embeddings: data.normalize ?? true,
        });
        output = raw && typeof raw.then === "function" ? await raw : raw;
      } catch (e) {
        const msg = errMessage(e);
        const isDeviceRuntime =
          /device lost|uncaptured|webgpu|wgpu/i.test(msg) &&
          entry.resolvedDevice === "wgpu";
        if (isDeviceRuntime) {
          self.postMessage({
            status: "deviceError",
            error: msg,
            previousDeviceMode: Bert.activeDeviceMode,
            requestedDeviceMode: Bert.activeDeviceMode,
            phase: "runtime",
          });
        } else {
          self.postMessage({
            status: "error",
            error: msg,
            phase: "inference",
          });
        }
        return;
      }

      self.postMessage({
        status: "complete",
        message: "complete",
        output: output.data,
        resolvedDevice: entry.resolvedDevice,
        adapterName: entry.adapterName,
      });
    } catch (e) {
      self.postMessage({
        status: "error",
        error: errMessage(e),
        phase: "modelLoad",
      });
    } finally {
      // Transient inference: unload the model + device after every run so VRAM
      // (wgpu device.destroy()) and RAM are released. `complete`/`error` above are
      // posted BEFORE dispose so the UI still gets its result.
      Bert.disposeActive();
    }
    return;
  }

  self.postMessage({
    status: "error",
    error: `Unknown command: ${command ?? "(missing)"}`,
    phase: "modelLoad",
  });
});
