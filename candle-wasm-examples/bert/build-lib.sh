#!/usr/bin/env bash
# Build the JS-lib WASM module (`m`) for bertWorker.js / lib-example.html.
#
# CPU-only (default):
#   ./build-lib.sh
#
# Portable WebGPU (opt-in):
#   ./build-lib.sh wgpu
#   ./build-lib.sh --features wgpu
#   FEATURES=wgpu ./build-lib.sh
#
set -euo pipefail

FEATURES_ARGS=()
if [[ "${1:-}" == "wgpu" || "${1:-}" == "--features=wgpu" ]]; then
  FEATURES_ARGS=(--features wgpu)
elif [[ "${1:-}" == "--features" && "${2:-}" == "wgpu" ]]; then
  FEATURES_ARGS=(--features wgpu)
elif [[ -n "${FEATURES:-}" ]]; then
  FEATURES_ARGS=(--features "${FEATURES}")
fi

cargo build --target wasm32-unknown-unknown --release "${FEATURES_ARGS[@]}"
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
