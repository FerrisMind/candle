#!/usr/bin/env bash
# Build the JS-lib WASM module (`m`) for whisperWorker.js / lib-example.html.
#
# Default: portable WebGPU (enables the crate `wgpu` feature), so the shipped
# artifact resolves `wgpu`/`auto` to WebGPU when the browser exposes an adapter.
#   ./build-lib.sh
#
# CPU-only (smaller wasm; `wgpu` selection fails and `auto` resolves to `cpu`):
#   ./build-lib.sh cpu
#
# Explicit wgpu (same as default) or a FEATURES override:
#   ./build-lib.sh wgpu
#   FEATURES=wgpu ./build-lib.sh
#
set -euo pipefail

FEATURES_ARGS=()
ARG="${1:-}"
if [[ "${ARG}" == "cpu" || "${ARG}" == "--features=cpu" ]]; then
  # CPU-only default: build without the wgpu feature.
  :
elif [[ "${ARG}" == "wgpu" || "${ARG}" == "--features=wgpu" || -z "${ARG}" ]]; then
  # Default produces a wgpu-capable wasm.
  FEATURES_ARGS=(--features wgpu)
elif [[ "${ARG}" == "--features" && -n "${2:-}" ]]; then
  FEATURES_ARGS=(--features "${2}")
elif [[ -n "${FEATURES:-}" ]]; then
  FEATURES_ARGS=(--features "${FEATURES}")
fi

cargo build --target wasm32-unknown-unknown --release "${FEATURES_ARGS[@]}"
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
