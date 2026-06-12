# WGPU/Vulkan Parity Audit

Date: 2026-06-11

## Environment Audit

- `nvidia-smi` detected `NVIDIA GeForce RTX 3060` with 12 GiB VRAM on driver `595.71.05`.
- `nvcc --version` is available and reports CUDA `12.0`.
- `vulkaninfo --summary` detected:
  - `GPU0`: `NVIDIA GeForce RTX 3060`, Vulkan API `1.4.329`, driver `595.71.05`
  - `GPU1`: `llvmpipe (LLVM 20.1.2, 256 bits)`, CPU Vulkan device
- Risk identified during audit: GPU tests can accidentally bind to the CPU Vulkan path (`llvmpipe`) unless adapter/device selection is explicit and observable.

## Repository Audit

- Core backend surface is defined in `candle-core/src/backend.rs` via `BackendDevice` and `BackendStorage`.
- Active backend implementations found:
  - CPU: `candle-core/src/cpu_backend`
  - CUDA: `candle-core/src/cuda_backend`
  - Metal: `candle-core/src/metal_backend`
  - WGPU: `candle-core/src/wgpu_backend.rs`
  - Vulkan: `candle-core/src/vulkan_backend.rs`
- GPU feature flags in `candle-core/Cargo.toml`:
  - `cuda`, `cudnn`, `nccl`, `metal`, `wgpu`, `vulkan`
- Kernel crates:
  - CUDA: `candle-kernels`
  - WGPU: `candle-wgpu-kernels`
  - Vulkan: `candle-vulkan-kernels`
- Existing parity-oriented tests already present:
  - synthetic backend smoke: `candle-core/tests/backend_smoke_tests.rs`
  - metamorphic/property suites: `candle-core/tests/gpu_metamorphic_tests.rs`, `gpu_property_tests.rs`
  - model matrix: `candle-transformers/tests/gpu_model_matrix.rs`
- Existing parity blockers still present:
  - CPU fallback machinery in `candle-core/src/storage.rs`
  - ignored/skip-driven WGPU tests in `candle-core/tests/*` and `candle-nn/tests/*`
  - many dtype/rank/not-implemented branches in `wgpu_backend.rs` and `vulkan_backend.rs`

## Reference Audit

- Useful local references found under `/home/mod479711/Downloads/candle_refs`:
  - `wgpu--ae4c7cc9`
  - `gpu-allocator--8041f70d`
  - `ug--29129f3f/ug-llama`
  - `llama.cpp`
- Current in-repo shader/reference corpus already contains strong ggml/llama.cpp lineage:
  - `candle-vulkan-kernels/src/shaders/*`
  - `candle-vulkan-kernels/build.rs` using `vulkan-shaders-gen.cpp`
  - `candle-wgpu-kernels/src/shaders/*`
  - multiple source comments referencing `ggml-vulkan.cpp`, `ggml-cuda.cu`, and `llama.cpp`

## First Concrete Fix

- Fixed `WgpuDevice::new(ordinal)` to stop ignoring `ordinal`.
- `wgpu` adapter selection now:
  - enumerates adapters explicitly
  - prefers non-CPU adapters by default
  - supports explicit selection with `CANDLE_WGPU_ADAPTER_NAME`
- Vulkan device selection now:
  - prefers non-CPU physical devices by default
  - supports explicit selection with `CANDLE_VULKAN_DEVICE_NAME`
- Added backend identity accessors:
  - `WgpuDevice::{adapter_name, adapter_backend, adapter_driver, adapter_driver_info, adapter_pci_bus_id}`
  - `VulkanDevice::{physical_device_name, physical_device_type}`
- Added smoke tests that can assert the real selected device/backend:
  - `backend_smoke_wgpu_reports_adapter_identity`
  - `backend_smoke_vulkan_reports_physical_device_identity`

## Verification Performed

- `cargo test -p candle-nn --features wgpu,vulkan softmax_vulkan -- --exact --nocapture`
  - passed
- `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 cargo test -p candle-nn --features wgpu,vulkan softmax_wgpu -- --ignored --exact --nocapture`
  - passed
- `CANDLE_EXPECTED_GPU_NAME="NVIDIA GeForce RTX 3060" cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_reports_physical_device_identity -- --exact --nocapture`
  - passed
- `CANDLE_EXPECTED_GPU_NAME="NVIDIA GeForce RTX 3060" CANDLE_EXPECTED_WGPU_BACKEND="Vulkan" CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_reports_adapter_identity -- --ignored --exact --nocapture`
  - passed
- `cargo clippy -p candle-core --features wgpu,vulkan --tests`
  - passed with warnings
  - notable pre-existing warning: `clippy::collapsible_if` in `candle-core/src/vulkan_backend.rs`

## Progress Update: 2026-06-11 Later Pass

- `candle-vulkan-kernels` build contract tightened on the Rust side only:
  - `build.rs` now verifies `glslc` is callable
  - `build.rs` now fails if SPIR-V generation produces an empty `spv.rs`
  - no changes were made to the copied `vulkan-shaders-gen.cpp`
- Added a Candle-side Vulkan shader for `powf_f32` outside the copied llama.cpp shader generator path.
- Verified `powf` is now native on Vulkan:
  - `backend_smoke_vulkan_powf_native_only` passes on `NVIDIA GeForce RTX 3060`
  - `backend_smoke_vulkan_unary_binary` no longer logs a `powf_f32` CPU fallback
- Measured current Vulkan smoke-family status on RTX 3060 with `CANDLE_DEBUG_GPU_FALLBACK=1`:
  - clean/no fallback log observed for:
    - `backend_smoke_vulkan_upload_and_dtype`
    - `backend_smoke_vulkan_reductions`
    - `backend_smoke_vulkan_shape_layout`
    - `backend_smoke_vulkan_matmul_conv_pool`
    - `backend_smoke_vulkan_rank5_native_policy`
    - `backend_smoke_vulkan_unary_binary`
    - `backend_smoke_vulkan_quantized_family`
- Found and resolved a Vulkan quantized parity bug in `indexed_moe`:
  - failing case was `Q8_0` under `backend_smoke_vulkan_quantized_family`
  - diagnostic probe in `candle-core/src/vulkan_backend.rs` showed Vulkan output matched the native f32 rhs path more closely than the q8_1 rhs MMVQ reference for this path
  - Vulkan `quantized_indexed_moe_f32` now keeps `Q8_0` on the direct GPU rhs path instead of the q8_1 rhs MMVQ variant
  - test reference logic was narrowed only for Vulkan `indexed_moe` so it mirrors the current backend rule rather than the broader qmatmul rule
- Additional targeted verification now passing:
  - `backend_smoke_vulkan_q8_1_qmatmul_regression`
  - `backend_smoke_vulkan_quantized_paths_only`
  - `cargo clippy -p candle-vulkan-kernels --all-targets`
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`

## Progress Update: Native `cmp` / `where_cond` Closure Slice

- `candle-core/src/wgpu_backend.rs`:
  - `cmp` now has a native packed-`u8` GPU path for `f32`/`f16` inputs.
  - `where_cond` now has a native GPU path for `u8` masks selecting between `f32`/`f16` tensors.
  - the new path covers contiguous and strided rank<=4 layouts exercised by backend smoke, including a narrowed compare view and transposed `where_cond`.
- `candle-core/src/vulkan_backend.rs`:
  - `cmp` now runs natively for `f32`/`f16` via a single shared compare shader and op code in params, reusing the ggml-style binary header path.
  - `where_cond` now runs natively for `u8` masks selecting between `f32`/`f16` tensors via one shared ternary shader.
- `candle-vulkan-kernels/src/candle-shaders` additions were kept minimal and Candle-owned only:
  - `cmp.comp`
  - `where_u8.comp`
- Llama/ggml reuse note:
  - the implementation follows the same “single shader + parameterized mode” direction seen in ggml/llama.cpp patterns such as `multi_add.comp`, instead of creating one shader per comparison operator.
- Targeted verification on the RTX 3060:
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_cmp_where_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_cmp_where_native_only -- --ignored --exact --nocapture`
    - passed

## Progress Update: Scatter / Indexing / Upsample Native Closure Slices

- `scatter_add_set` / `index_add`:
  - `candle-core/src/wgpu_backend.rs` now has a native last-dim `scatter_add` path for `f32 + u32 ids` using the existing `set_rows` shader family in additive mode.
  - `candle-core/src/vulkan_backend.rs` now has a matching native last-dim `scatter_add` path using one Candle-owned shader, `set_rows_add.comp`, while keeping the copied `vulkan-shaders-gen.cpp` untouched.
  - `candle-core/src/tensor.rs` now decomposes non-last-dim `scatter_add_set` and `index_add` into `permute -> contiguous -> last-dim GPU op -> permute back`, keeping the exercised float slice on GPU instead of routing through CPU.
- `scatter_set` / `gather` / `index_select`:
  - `gather` already used tensor-level decomposition for non-last-dim cases; `scatter_set` now mirrors that pattern for the exercised `f32`/`f16` + `u32 ids` slice.
  - the resulting non-last-dim `gather` / `scatter_set` / `index_select` smoke slice is now certified with `native_required` on both backends.
- `upsample_nearest1d`:
  - both backends already had a native implementation via GPU matmul with interpolation weights; this is now explicitly certified on RTX 3060 with native-only smoke tests instead of being treated as merely inferred from code.
- Llama/ggml reuse note:
  - no copied llama.cpp generator sources were modified.
  - the new scatter-add work reuses the existing row-wise `set_rows` family instead of introducing separate per-shape kernels.
- Targeted verification on the RTX 3060:
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_scatter_add_index_add_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_scatter_add_index_add_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_shape_indexing_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_shape_indexing_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_upsample_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_upsample_native_only -- --ignored --exact --nocapture`
    - passed

## Progress Update: Softmax / RMSNorm / Rope Native Certification Slice

- `candle-nn/tests/ops.rs` backend-specific `softmax`, `rms_norm`, and `rope` tests now assert `fallback_count == 0` instead of only checking numerics.
- `rope_i` root cause:
  - both `wgpu` and Vulkan were still falling back through the `rotary-emb-int` custom op.
  - a naive fallback to `rope_i_slow()` was not acceptable on GPU because it builds a rank-5 graph and immediately hit remaining backend rank limitations.
- Candle-side fix:
  - `candle-nn/src/rotary_emb.rs` now routes `rope_i()` on `wgpu`/Vulkan through a rank-4 tensor decomposition:
    - `index_select` even channels
    - `index_select` odd channels
    - broadcast `cos` / `sin`
    - elementwise combine
    - last-dim `scatter_set` to re-interleave
  - this reuses already-closed GPU indexing and scatter paths instead of adding new shaders.
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-nn --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan softmax_vulkan -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan softmax_wgpu -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan rms_norm_vulkan -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan rms_norm_wgpu -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan rope_vulkan -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan rope_wgpu -- --ignored --exact --nocapture`
    - passed

## Progress Update: LayerNorm / SDPA / Mini-Graph Native Certification Slice

- `candle-nn/tests/ops.rs`:
  - `layer_norm_{wgpu,vulkan}` and `sdpa_{wgpu,vulkan}` now use the same `run_native_backend_case` helper as `softmax`, `rms_norm`, and `rope`, so these tests now fail on any backend CPU fallback instead of only checking numerics.
- `candle-nn/tests/ops.rs::mini_graph_{wgpu,vulkan}` was already checking fallback counts through `run_graph_case`; this pass re-certified both backends on RTX 3060.
- Targeted verification on the RTX 3060:
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan layer_norm_vulkan -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan layer_norm_wgpu -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan sdpa_vulkan -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan sdpa_wgpu -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan mini_graph_vulkan -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-nn --features wgpu,vulkan mini_graph_wgpu -- --ignored --exact --nocapture`
    - passed
- Current meaning of this slice:
  - a synthetic attention block, gated MLP block, and mixer block now execute end-to-end on both `wgpu` and Vulkan without backend CPU fallback on the target RTX 3060.
  - this strengthens op-to-graph evidence for `softmax`, `rms_norm`, `rope`, `matmul`, `index_select`, `scatter_set`, and residual MLP compositions beyond the earlier single-op smoke tests.

## Progress Update: Conv-Transpose Native Closure Slice

- `candle-core/src/conv.rs`:
  - `wgpu` `conv_transpose1d` now has a Candle-side tensor decomposition that reuses existing GPU ops instead of adding a new shader:
    - cast to `f32` when the source dtype is `f16`
    - `matmul`
    - `reshape` / `permute`
    - `index_add` with precomputed `u32` output indices and a validity mask
    - cast back to `f16` when needed
  - `wgpu` `conv_transpose2d` now uses the same reuse-first pattern over flattened spatial positions.
  - the copied `candle-vulkan-kernels/src/shaders/vulkan-shaders-gen.cpp` was not modified.
- Llama/ggml reuse note:
  - the 1D path follows the same `mul_mat + col2im` decomposition idea documented in `llama.cpp/tests/test-col2im-1d.cpp`, adapted to Candle tensor layouts and implemented entirely on the Candle side.
  - instead of adding a dedicated `wgpu` deconvolution shader, this pass reuses already-closed `matmul` and `index_add` GPU primitives.
- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_wgpu_conv_transpose_native_only`
  - added `backend_smoke_vulkan_conv_transpose_native_only`
  - widened `smoke_f32_conv_transpose` to compare additional `groups`, `dilation`, `padding`, and `output_padding` cases against CPU references.
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_conv_transpose_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_conv_transpose_native_only -- --exact --nocapture`
    - passed

## Progress Update: Argsort Native Certification Slice

- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_wgpu_argsort_native_only`
  - added `backend_smoke_vulkan_argsort_native_only`
- Scope certified in this pass:
  - ascending and descending `arg_sort_last_dim`
  - `sort_last_dim`
  - the existing smoke already includes a larger `1x300` case on both backends and a larger `1x1100` case on Vulkan
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_argsort_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_argsort_native_only -- --exact --nocapture`
    - passed

## Progress Update: Vulkan `Q8_1` Quantized Closure Slice

- `candle-core/src/quantized/mod.rs`:
  - at that stage, the explicit `Q8K` fallback gate was still in place; this pass only closed `Q8_1`
- Why this change is justified:
  - honest fallback accounting showed the previous Vulkan `Q8_1` closure claim was false:
    - `matmul_q8_1_f32` was not generated
    - `get_rows_q8_1_f32` was not generated
  - the minimal reuse-first fix is a Candle-side GPU repack from `Q8_1` weights to `Q8_0`, then reuse of the already working Vulkan `Q8_0` kernels
  - this keeps the copied `llama.cpp` shader generator untouched and avoids inventing a new Vulkan quantized shader family
- `candle-vulkan-kernels/src/candle-shaders/repack_q8_1_to_q8_0.comp`:
  - added a Candle-owned compute shader that reads `block_q8_1`, keeps `d`, copies `qs`, ignores the auxiliary `s`, and writes `block_q8_0`
- `candle-core/src/vulkan_backend.rs`:
  - added a local Vulkan helper that runs the GPU repack and returns a temporary `U8` storage for the existing `Q8_0` kernels
  - `quantized_matmul` now intercepts `Q8_1` weights and reuses the existing Vulkan `Q8_0` matmul path
  - `quantized_index_select_f32` now intercepts `Q8_1` weights and reuses the existing Vulkan `Q8_0` get-rows path
- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_vulkan_q8_1_quantized_native_only`
  - this test certifies, without backend CPU fallback:
    - `Q8_1` quantized matmul
    - `Q8_1` quantized row gather via the public embedding/index-select path
    - `Q8_1` quantized indexed MoE through the existing GPU composition path
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_q8_1_quantized_native_only -- --exact --nocapture`
    - passed, with no fallback log for the certified `Q8_1` slice
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_q8_1_quantized_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_quantized_family -- --exact --nocapture`
    - passed
  - at that stage, `Q8K` was still covered only by a fallback-policy test; this is superseded by the later `Q8K` closure update below

## Progress Update: Vulkan Quantized Dequantize Native Slice

- `candle-core/src/vulkan_backend.rs`:
  - added a native Vulkan dequantize helper for the currently shader-backed quantized dtypes
  - legacy quantized dtypes now use the existing generated `dequant_*` shader family
  - `Q8_1` dequantize reuses the existing Candle-side GPU `Q8_1 -> Q8_0` repack, then dequantizes via the existing Vulkan `Q8_0` path
  - K-quants `Q2K..Q6K` dequantize on GPU to `f16`, then convert natively to `f32`
- `candle-core/src/quantized/mod.rs`:
  - `QVulkanStorage::dequantize` now tries the native Vulkan path first
  - unsupported dequantize cases now record honest Vulkan CPU fallback before using the previous CPU path
- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_vulkan_quantized_dequantize_native_only`
  - this test certifies, without backend CPU fallback, Vulkan `QTensor::dequantize` for:
    - `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
    - `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`
  - at that stage, `Q8K` remained outside this native slice; this is superseded by the later `Q8K` closure update below
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_quantized_dequantize_native_only -- --exact --nocapture`
    - passed

## Progress Update: WGPU `Q8_1` Direct Quantized Closure Slice

- `candle-core/src/quantized/mod.rs`:
  - removed the stale Candle-side `wgpu` special case that downloaded `Q8_1` weights to CPU, repacked them as `Q8_0`, and re-uploaded them before `qmatmul` and quantized row-gather
- `candle-wgpu-kernels/src/shaders/*`:
  - widened the existing shader family instead of adding a new shader file
  - `get_rows.wgsl` now dequantizes `Q8_1` directly
  - fixed the existing `Q8_1` decode in `mul_mat.wgsl`, `mul_mat_vec.wgsl`, and `mul_mat_decls.tmpl` so it matches Candle/ggml semantics:
    - `Q8_1` stores `d` plus an auxiliary `s`
    - it does not store a per-value additive offset
- Why this change is justified:
  - the previous repack path was Candle-side only and hid a real bug in the existing `wgpu` `Q8_1` decode path
  - this pass fixes Candle-side interpretation instead of introducing another parallel shader family
- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_wgpu_q8_1_quantized_native_only`
  - this test certifies, without backend CPU fallback:
    - `Q8_1` quantized matmul
    - `Q8_1` quantized row gather via the public embedding/index-select path
    - `Q8_1` quantized indexed MoE through the existing `get_rows + matmul` GPU composition path
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_q8_1_quantized_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_quantized_paths -- --ignored --exact --nocapture`
    - passed

## Progress Update: WGPU Quantized Dequantize Native Slice

- `candle-core/src/quantized/mod.rs`:
  - `QWgpuStorage::dequantize` now tries a native `wgpu` path first instead of downloading quantized weights to CPU unconditionally
  - the native path reuses the existing quantized `get_rows` shader family through `quantized_index_select_f32`
  - implementation detail:
    - build device-side row ids `[0..num_blocks)`
    - view the quantized tensor as `(num_blocks, block_size)`
    - dequantize every row through the existing indexed row-gather path
  - unsupported cases still record honest `wgpu` CPU fallback before using the previous CPU path
- Reuse-first note:
  - this pass does not add a new `wgpu` dequant shader
  - it deliberately follows the existing quantized row-gather composition instead of creating another parallel kernel family
- `candle-core/tests/backend_smoke_tests.rs`:
  - added `backend_smoke_wgpu_quantized_dequantize_native_only`
  - this test certifies, without backend CPU fallback, `wgpu` `QTensor::dequantize` for:
    - `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
    - `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`
  - at that stage, `Q8K` remained outside this native slice; this is superseded by the later `Q8K` closure update below
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_quantized_dequantize_native_only -- --ignored --exact --nocapture`
    - passed

## Progress Update: `Q8K` Quantized Closure Slice

- `candle-core/src/quantized/mod.rs`:
  - removed the explicit `Q8K` CPU-fallback routing on both `wgpu` and Vulkan public quantized paths
  - `Q8K` now closes reuse-first through:
    - native GPU `dequantize`
    - existing dense GPU `matmul`
    - existing dense GPU `index_select`
  - this keeps the copied `llama.cpp` generator code untouched and avoids adding a full specialized `Q8K` quantized matmul family before parity is closed
- `candle-wgpu-kernels/src/lib.rs` and `candle-wgpu-kernels/src/shaders/get_rows.wgsl`:
  - added Rust-side `Q8_K` dtype wiring
  - extended the existing quantized get-rows/dequant family with a `Q8_K` decode branch
  - no new standalone `wgpu` shader file was introduced
- `candle-vulkan-kernels/src/candle-shaders/dequant_q8_k_f32.comp`:
  - added one Candle-owned Vulkan shader for native `Q8K -> f32` dequantize
  - the copied `vulkan-shaders-gen.cpp` and generated llama.cpp shader sources were left unchanged
- `candle-core/src/vulkan_backend.rs`:
  - added native Vulkan `Q8K` dequantize wiring
  - fixed the Vulkan `Q8K` path to allocate an `f32` destination directly for this shader, avoiding a mismatched `f16` intermediate
- `candle-core/tests/backend_smoke_tests.rs`:
  - replaced the old `Q8K` fallback-policy tests with native-only tests:
    - `backend_smoke_wgpu_q8k_quantized_native_only`
    - `backend_smoke_vulkan_q8k_quantized_native_only`
  - widened `backend_smoke_{wgpu,vulkan}_quantized_dequantize_native_only` to include `Q8K`
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `cargo clippy -p candle-wgpu-kernels --all-targets`
    - passed
  - `cargo clippy -p candle-vulkan-kernels --all-targets`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_q8k_quantized_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_quantized_dequantize_native_only -- --ignored --exact --nocapture`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_q8k_quantized_native_only -- --exact --nocapture`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_quantized_dequantize_native_only -- --exact --nocapture`
    - passed

## Progress Update: Vulkan `q8_1 rhs` Route Gating And General Quantized Smoke Re-Certification

- `candle-core/src/vulkan_backend.rs`:
  - Vulkan quantized `q8_1 rhs` routing now verifies that the exact SPIR-V module selected by the current runtime policy is actually present before taking that path.
  - if the chosen `matmul` / `mul_mat_vec` / `mul_mat_vec_id` `q8_1 rhs` shader variant is absent in the current build, Candle stays on the existing native GPU `f32` rhs path instead of dropping into a CPU-backed failure path.
  - the copied `llama.cpp` generator sources remain untouched; this pass only changes Candle-side route selection.
- `candle-core/tests/backend_smoke_tests.rs`:
  - Vulkan `q8_1 rhs` expectation logic now mirrors real shader availability, not only dtype/device heuristics.
  - the general `Q8K` quantized smoke reference now matches the implemented GPU route, `dequantize(weight) + dense matmul`, instead of CPU quantized `QMatMul`.
- Targeted verification on the RTX 3060:
  - `cargo clippy -p candle-core --features wgpu,vulkan --tests`
    - passed
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_wgpu_quantized_paths -- --ignored --exact --nocapture`
    - passed after promotion to `native_required`, so this suite now certifies `fallback_count == 0`
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_quantized_family -- --exact --nocapture`
    - passed after promotion to `native_required`, so this suite now certifies `fallback_count == 0`
  - `CANDLE_DEBUG_GPU_FALLBACK=1 CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-core --features wgpu,vulkan backend_smoke_vulkan_quantized_paths_only -- --exact --nocapture`
    - passed after promotion to `native_required`, so the focused Vulkan quantized-path slice also certifies `fallback_count == 0`

## Real-Model Certification Slice: Vulkan on RTX 3060

- Full command executed:
  - `CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-transformers --features vulkan gpu_model_matrix_vulkan -- --ignored --exact --nocapture`
- Result:
  - `gpu_model_matrix_vulkan ... ok`
  - total runtime in debug test profile: `283.30s`
- Per-case runtime / fallback evidence from the test log:
  - `dense_causal_decoder_case`: `4.63s`, fallback count `0`
  - `quantized_causal_gguf_case`: `11.75s`, fallback count `0`
  - `encoder_only_text_case`: `1.60s`, fallback count `0`
  - `audio_seq2seq_case`: `17.39s`, fallback count `0`
  - `vision_convmixer_case`: `247.56s`, fallback count `0`

## Real-Model Certification Slice: WGPU on RTX 3060

- Full command executed:
  - `CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-transformers --features wgpu gpu_model_matrix_wgpu -- --ignored --exact --nocapture`
- Result:
  - `gpu_model_matrix_wgpu ... ok`
  - total runtime in debug test profile: `86.35s`
- Per-case runtime / fallback evidence from the test log:
  - `dense_causal_decoder_case`: `3.14s`, fallback count `0`
  - `quantized_causal_gguf_case`: `13.22s`, fallback count `0`
  - `encoder_only_text_case`: `2.06s`, fallback count `0`
  - `audio_seq2seq_case`: `12.86s`, fallback count `0`
  - `vision_convmixer_case`: `54.59s`, fallback count `0`

## Initial Operation-to-Model Coverage Slice

This is an initial, honest slice rather than a final closed matrix. The real-model runs below are now proven by passing `gpu_model_matrix_vulkan`, `gpu_model_matrix_wgpu`, and `gpu_model_matrix_cuda` commands, but some per-operation coverage claims are still architectural inferences from the executed model graph and have not yet been backed by fine-grained op tracing.

| Operation family | Real-model case | Model artifact / revision | DType / quantization | Input / prompt | Comparison target | Vulkan evidence | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense decoder prefill/decode | `dense_causal_decoder_case` in `candle-transformers/tests/gpu_model_matrix.rs` | `karpathy/tinyllamas@main`, `stories15M.bin` | dense f32 activations | token ids `[1, 13, 42, 7, 19, 5]`, decode token `[11]` | CPU logits | passed, fallback count `0` | proven real-model case |
| Embedding + attention + rope + KV-cache + logits | `dense_causal_decoder_case` | same as above | dense path | same as above | CPU logits | same run as above | graph-level inference from executed Llama2-c path; op-level trace still pending |
| Quantized causal decode | `quantized_causal_gguf_case` | `unsloth/Qwen3-0.6B-GGUF@main`, `Qwen3-0.6B-Q4_K_M.gguf` | GGUF quantized, Q4_K_M weights | token ids `[1, 2, 3, 4]`, decode token `[5]` | CPU logits | passed, fallback count `0` | proven real-model case |
| Quantized matmul / dequant / quantized model-critical path | `quantized_causal_gguf_case` | same as above | quantized GGUF path | same as above | CPU logits | same run as above | graph-level inference from executed quantized Qwen3 path; op-level trace still pending |
| Encoder-only text transformer | `encoder_only_text_case` | `sentence-transformers/all-MiniLM-L6-v2@refs/pr/21`, `config.json`, `model.safetensors` | bert::DTYPE path | ids `[101, 2023, 2003, 1037, 3231, 102, 0, 0]`, mask `[1, 1, 1, 1, 1, 1, 0, 0]` | CPU hidden states + pooled output | passed, fallback count `0` | proven real-model case |
| Attention + mask handling + reductions/mean-pool | `encoder_only_text_case` | same as above | dense path | same as above | CPU hidden states + pooled output | same run as above | graph-level inference from executed MiniLM path; op-level trace still pending |
| Audio encoder-decoder seq2seq | `audio_seq2seq_case` | `openai/whisper-tiny.en@refs/pr/15`, `config.json`, `model.safetensors` | whisper::DTYPE path | deterministic mel spectrogram, decoder ids `[1]` | CPU encoder output + decode logits | passed, fallback count `0` | proven real-model case |
| Seq2seq attention / logits / decoder cross-attention | `audio_seq2seq_case` | same as above | dense path | same as above | CPU encoder output + decode logits | same run as above | graph-level inference from executed Whisper path; op-level trace still pending |
| Conv-heavy vision path | `vision_convmixer_case` | `lmz/candle-convmixer@main`, `convmixer_1024_20_ks9_p14.safetensors` | f32 | deterministic image tensor `(1, 3, 224, 224)` | CPU logits | passed, fallback count `0` | proven real-model case |
| Conv / layout / activation / pooling-adjacent vision graph | `vision_convmixer_case` | same as above | dense path | same as above | CPU logits | same run as above | graph-level inference from executed ConvMixer path; op-level trace still pending |

## Immediate Gaps After The Current Real-Model Pass

- `gpu_model_matrix_vulkan` and `gpu_model_matrix_wgpu` prove that a representative dense decoder, quantized GGUF decoder, encoder-only transformer, seq2seq audio model, and conv-heavy vision model all run on the RTX 3060 Vulkan and `wgpu` backends without observed CPU fallback counts.
- This is still not the final operation-to-model matrix required by `goal.md` because:
  - per-operation trace attribution is not yet recorded;
  - benchmark/profiling status for these model cases is still missing;
  - the matrix is not yet expanded into the full CUDA parity row set.

## CUDA Comparison Slice On The Same RTX 3060

- Extended `candle-transformers/tests/gpu_model_matrix.rs` and `tests/support/mod.rs` so the same public real-model matrix can now run on `cuda` in addition to `wgpu` and `vulkan`.
- CUDA command executed:
  - `CANDLE_REQUIRE_CUDA_TEST_DEVICE=1 cargo test -p candle-transformers --features cuda gpu_model_matrix_cuda -- --ignored --exact --nocapture`
- Result:
  - `gpu_model_matrix_cuda ... ok`
  - total runtime in debug test profile: `47.82s`
- Per-case runtime / fallback evidence from the CUDA test log:
  - `dense_causal_decoder_case`: `2.27s`, fallback count `0`
  - `quantized_causal_gguf_case`: `8.05s`, fallback count `0`
  - `encoder_only_text_case`: `1.53s`, fallback count `0`
  - `audio_seq2seq_case`: `12.03s`, fallback count `0`
  - `vision_convmixer_case`: `23.64s`, fallback count `0`

## First CUDA-vs-Vulkan Real-Model Comparison Table

| Model case | Vulkan status | CUDA status | Same input contract | Same comparison target | Notes |
| --- | --- | --- | --- | --- | --- |
| `dense_causal_decoder_case` | pass, fallback `0`, `4.63s` | pass, fallback `0`, `2.27s` | yes | CPU logits | first dense decoder parity slice on same RTX 3060 |
| `quantized_causal_gguf_case` | pass, fallback `0`, `11.75s` | pass, fallback `0`, `8.05s` | yes | CPU logits | first quantized GGUF parity slice on same RTX 3060 |
| `encoder_only_text_case` | pass, fallback `0`, `1.60s` | pass, fallback `0`, `1.53s` | yes | CPU hidden states + pooled output | encoder-only text path aligned across backends |
| `audio_seq2seq_case` | pass, fallback `0`, `17.39s` | pass, fallback `0`, `12.03s` | yes | CPU encoder output + decode logits | seq2seq audio path aligned across backends |
| `vision_convmixer_case` | pass, fallback `0`, `247.56s` | pass, fallback `0`, `23.64s` | yes | CPU logits | strong conv-heavy coverage slice; Vulkan debug runtime is notably slower and needs profiling |

## First CUDA-vs-WGPU Real-Model Comparison Table

| Model case | WGPU status | CUDA status | Same input contract | Same comparison target | Notes |
| --- | --- | --- | --- | --- | --- |
| `dense_causal_decoder_case` | pass, fallback `0`, `3.14s` | pass, fallback `0`, `2.27s` | yes | CPU logits | dense decoder parity slice on the same RTX 3060 |
| `quantized_causal_gguf_case` | pass, fallback `0`, `13.22s` | pass, fallback `0`, `8.05s` | yes | CPU logits | quantized GGUF parity slice on the same RTX 3060 |
| `encoder_only_text_case` | pass, fallback `0`, `2.06s` | pass, fallback `0`, `1.53s` | yes | CPU hidden states + pooled output | encoder-only text path aligned across backends |
| `audio_seq2seq_case` | pass, fallback `0`, `12.86s` | pass, fallback `0`, `12.03s` | yes | CPU encoder output + decode logits | seq2seq audio path aligned across backends |
| `vision_convmixer_case` | pass, fallback `0`, `54.59s` | pass, fallback `0`, `23.64s` | yes | CPU logits | conv-heavy coverage slice; `wgpu` is materially closer to CUDA than current Vulkan debug runtime |

## Immediate Next Steps

1. Build the first explicit CUDA parity matrix from `candle-core/src/cuda_backend`, `candle-kernels/src/*.cu`, `wgpu_backend.rs`, and `vulkan_backend.rs`.
2. Convert current fallback-allowed backend smoke/property/model suites into a matrix of:
   - native-covered
   - fallback-covered
   - missing
3. Remove CPU fallback row-by-row starting from already exercised paths:
   - shape/layout ops
   - gather/index_select/scatter
   - reductions/cumsum/argmax
   - matmul/quantized matmul
4. Run `candle-transformers/tests/gpu_model_matrix.rs` selectively once required model artifacts are staged locally.

## Focused Profiling Slice: `vision_convmixer_case`

This section is the first profiler-backed performance slice on the exact target machine. It is still far from the full profiling closure required by `goal.md`, but it moves the work from broad timing suspicion to a concrete, code-level Vulkan optimization target.

### Release Single-Case Runtime

- Selective release commands used:
  - `CANDLE_GPU_MODEL_CASE_FILTER=vision_convmixer_case CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 CANDLE_EXPECTED_GPU_NAME='NVIDIA GeForce RTX 3060' cargo test -p candle-transformers --features vulkan gpu_model_matrix_vulkan --release -- --ignored --exact --nocapture`
  - `CANDLE_GPU_MODEL_CASE_FILTER=vision_convmixer_case CANDLE_REQUIRE_CUDA_TEST_DEVICE=1 cargo test -p candle-transformers --features cuda gpu_model_matrix_cuda --release -- --ignored --exact --nocapture`
- Results before the allocator-churn cleanup:
  - Vulkan: `19.28s`, fallback count `0`
  - CUDA: `2.36s`, fallback count `0`
- After the first Vulkan host-allocation cleanup in `candle-core/src/vulkan_backend.rs`:
  - Vulkan: `17.58s`, fallback count `0`
- After descriptor-set batch allocation in `candle-core/src/vulkan_backend.rs`:
  - Vulkan: `15.34s`, fallback count `0`
- Negative experiment that was measured and reverted:
  - raising the dense rank-2 matvec row chunk from `8` to `16` regressed to `18.45s`
  - raising it to `32` regressed to `22.29s`
- Interpretation:
  - The ConvMixer gap is not a debug-only illusion; it remains large in release on the same RTX 3060.
  - The first host-side cleanup and then descriptor-set batching both improved the observed Vulkan release runtime, but Vulkan is still much slower than CUDA on this model.
  - Simply increasing dense row chunk size was not a safe win here; the larger shader specialization cost more than the reduced dispatch count saved.

### External Runtime Baseline

- Direct release test binaries were also timed with `/usr/bin/time -v` before the Vulkan cleanup:
  - Vulkan command: `target/release/deps/gpu_model_matrix-e36ae9784094cd26 --ignored --exact gpu_model_matrix_vulkan --nocapture`
    - wall clock: `16.12s`
    - user time: `15.70s`
    - system time: `0.58s`
    - max RSS: `353780 kB`
  - CUDA command: `target/release/deps/gpu_model_matrix-8f6dbe0042707428 --ignored --exact gpu_model_matrix_cuda --nocapture`
    - wall clock: `3.10s`
    - user time: `2.71s`
    - system time: `0.61s`
    - max RSS: `411348 kB`
- Direct release test binary timing after descriptor-set batching:
  - Vulkan command: `target/release/deps/gpu_model_matrix-6f5f91390761be95 --ignored --exact gpu_model_matrix_vulkan --nocapture`
    - wall clock: `16.08s`
    - user time: `15.91s`
    - system time: `0.49s`
    - max RSS: `358024 kB`
- Interpretation:
  - The Vulkan ConvMixer path is consuming far more wall time while using roughly one CPU core on average, which points away from simple multi-core saturation and toward per-dispatch orchestration / synchronization overhead on the host side.

### Profiler Availability

- `perf` is installed, but direct `perf stat` / `perf record` runs are blocked on this machine by `perf_event_paranoid=4`.
- Available userspace profiler used instead:
  - `heaptrack`
- Generated profiling artifacts:
  - `profiling/vision_convmixer_vulkan.heaptrack.zst`
  - `profiling/vision_convmixer_vulkan_after_smallvec.heaptrack.zst`
  - `profiling/vision_convmixer_vulkan_after_descriptor_batch.heaptrack.zst`

### Heaptrack Findings

- Initial Vulkan ConvMixer `heaptrack` run:
  - total allocations: `5,642,954`
  - leaked allocations reported by tool: `12,838`
  - temporary allocations: `1,444,396`
  - the run completed the model case, then the Rust test thread overflowed its stack during teardown under heaptrack instrumentation; the trace file was still produced and analyzable
- `heaptrack_print` on the initial trace showed:
  - `1,353,446` allocation calls rooted at `candle_core::vulkan_backend::VulkanDevice::run_compute_specialized_with_options`
  - the dominant call chain under that path was:
    - `VulkanStorage::conv2d`
    - `VulkanStorage::run_matmul_f32`
    - `VulkanStorage::run_rank2_matmul_f32_via_batched_matvec`
  - `676,647` temporary allocations were attributed to `ash::device::Device::allocate_descriptor_sets`
- First optimization pass applied on the Candle side only:
  - replaced several per-dispatch `Vec` allocations in `run_compute_with_shader` with `SmallVec`
  - files touched:
    - `Cargo.toml`
    - `candle-core/Cargo.toml`
    - `candle-core/src/vulkan_backend.rs`
- Vulkan `heaptrack` after the `SmallVec` pass:
  - total allocations: `3,611,920`
  - leaked allocations reported by tool: `12,856`
  - temporary allocations: `1,588,276`
  - `heaptrack_print` now shows the top allocation hotspot as:
    - `ash::device::Device::allocate_descriptor_sets` with `676,910` allocation calls
  - dominant remaining stack:
    - `VulkanDevice::run_compute_specialized_with_options`
    - `VulkanStorage::run_rank2_matmul_f32_via_batched_matvec`
    - `VulkanStorage::run_matmul_f32`
    - `VulkanStorage::conv2d`
- Second optimization pass applied on the Candle side only:
  - descriptor sets are now allocated in small batches per descriptor-set layout inside an active Vulkan batch, instead of one `allocate_descriptor_sets` call per dispatch
  - file touched:
    - `candle-core/src/vulkan_backend.rs`
- Vulkan `heaptrack` after descriptor-set batching:
  - total allocations: `3,104,162`
  - leaked allocations reported by tool: `12,651`
  - temporary allocations: `974,979`
  - `ash::device::Device::allocate_descriptor_sets` dropped from `676,910` calls to `105,396` calls
  - `heaptrack_print` now shows the broader caller as the hottest allocation root:
    - `candle_core::vulkan_backend::VulkanDevice::run_compute_specialized_with_options`
  - dominant remaining stack is still:
    - `VulkanDevice::run_compute_specialized_with_options`
    - `VulkanStorage::run_rank2_matmul_f32_via_batched_matvec`
    - `VulkanStorage::run_matmul_f32`
    - `VulkanStorage::conv2d`

### Code-Level Interpretation

- The current Vulkan ConvMixer bottleneck is no longer just “Vulkan is slow”.
- The first profiler-validated hotspot was:
  - per-dispatch descriptor-set allocation in `candle-core/src/vulkan_backend.rs:2105-2128`
- The first profiler-validated host-side improvement was:
  - bulk allocating unique descriptor sets per layout from the active batch pool instead of calling `allocate_descriptor_sets` for every dispatch
- The high-frequency caller generating this churn is:
  - `candle-core/src/vulkan_backend.rs:2371-2438`
  - `run_rank2_matmul_f32_via_batched_matvec` loops over `row_idx` and repeatedly calls `run_compute_specialized_with_options`
- The matmul path selection that routes ConvMixer into this implementation is visible at:
  - `candle-core/src/vulkan_backend.rs:3895-3898`

### Next Optimization Target

The highest-value next Vulkan performance step is now concrete:

1. Continue reducing residual per-dispatch host overhead inside `run_compute_with_shader` now that descriptor-set allocation pressure is lower.
2. Reconsider the dense ConvMixer route through `run_rank2_matmul_f32_via_batched_matvec`; a different dense matmul strategy may be needed because larger row chunks regressed.
3. Re-measure `vision_convmixer_case` release runtime and heap allocation profile after the next Vulkan-side change.

## Progress Update: Tiled GEMM Route For Dense f32 Matmul (Vulkan)

- Root cause of the ConvMixer gap confirmed: rank-2 dense f32 matmul with `m > 1` always routed through `run_rank2_matmul_f32_via_batched_matvec`, which loops `m / 8` host-side dispatches of the `mul_mat_vec` shader. ConvMixer conv2d-as-matmul shapes (`m` in the hundreds/thousands) paid massive per-dispatch host overhead.
- llama.cpp reference behavior (`ggml-vulkan.cpp:9157`, `mul_mat_vec_max_cols = 8` at `ggml-vulkan.cpp:302`): the matvec family is only used when output rows `<= 8`; larger GEMMs go to the tiled `mul_mm` pipeline. Candle now mirrors this routing rule via `vulkan_dense_gemm_prefers_tiled` in `candle-core/src/vulkan_backend.rs`.
- Two real bugs were exposed and fixed in the previously dormant tiled `matmul_f32_f32` path:
  1. wrong spec constants: `BLOCK_SIZE` was hardcoded to `64` with `WARP = subgroup_size`; the `mul_mm.comp` warp grid requires `BLOCK_SIZE / WARP == (BM/WM) * (BN/WN)`, i.e. 4 warps for the 64x64/32x32 tile, so `BLOCK_SIZE` must be `4 * WARP`. The old values produced zero/garbage output tiles for `m > 16` shapes (verified by a standalone shape sweep: relative error ~1.0 on `64x64x64` and larger).
  2. wrong precision variant: the un-suffixed `matmul_f32_f32` SPIR-V variant stages tiles as f16 (ggml naming: `_fp32` suffix = full-f32 staging). The f16 staging lost ~1e-3 precision vs the CPU/CUDA reference and failed `backend_smoke_vulkan_conv1d_multi_channel_regression`. The route now uses `matmul_f32_f32_fp32`.
- New regression coverage so this class of bug cannot return silently:
  - `backend_smoke_vulkan_matmul_shape_sweep_native_only`
  - `backend_smoke_wgpu_matmul_shape_sweep_native_only`
  - the sweep crosses the matvec/tiled routing threshold, tile-size boundaries (32/64), K-loop boundaries, odd shapes for OOB guards, transposed rhs, and batched cases, with a K-scaled f32 tolerance.
- Measured effect on RTX 3060 (release, `vision_convmixer_case`):
  - Vulkan: `15.34s -> 2.60s` (CUDA same machine: `2.42s`)
  - heaptrack total allocations: `3,104,162 -> 2,065,598`
  - heaptrack temporary allocations: `974,979 -> 234,592`
  - artifact: `profiling/vision_convmixer_vulkan_after_tiled_gemm.zst`

## Progress Update: GGUF Raw-Float (F32/F16/BF16) Dequantize Closure

- Honest fallback accounting in the full release model matrix exposed 113 Vulkan CPU fallbacks in `quantized_causal_gguf_case`: every GGUF tensor stored as raw `F32` (norm weights/biases in Qwen3-0.6B) hit `vulkan backend op quantized dequantize not implemented`.
- Vulkan closure (`candle-core/src/vulkan_backend.rs::quantized_dequantize_f32`):
  - `GgmlDType::F32`: device-side buffer copy into an `f32` storage (no math needed)
  - `GgmlDType::F16` / `GgmlDType::BF16`: device-side copy into typed storage + existing native `to_dtype` cast
- WGPU closure (`candle-core/src/wgpu_backend.rs::quantized_raw_float_dequantize_f32` + routing in `candle-core/src/quantized/mod.rs`):
  - `F32`: `copy_buffer_to_buffer`
  - `F16`: WGSL `unpack2x16float` bit-decode (no `SHADER_F16` feature required)
  - `BF16`: WGSL exponent-shift bit-decode (`bitcast<f32>(bits << 16)`), first native BF16-consuming path on the `wgpu` backend
- `candle-core/src/dummy_wgpu_backend.rs` gained the matching stub so `vulkan`-only builds keep compiling.
- The vulkan dequantize match is now exhaustive over `GgmlDType`; the dead `unsupported` arm was removed.
- Test closure:
  - `backend_smoke_{wgpu,vulkan}_quantized_dequantize_native_only` now include `F32`, `F16`, `BF16`
- Real-model effect on RTX 3060 (release):
  - `quantized_causal_gguf_case` Vulkan fallbacks: `113 -> 0`
  - `gpu_model_matrix_vulkan` now passes end-to-end as native-required: dense `0.85s`, quantized `3.12s`, encoder `0.31s`, audio `1.04s`, vision `2.60s`, all fallback `0`
  - `gpu_model_matrix_wgpu` passes: dense `0.40s`, quantized `3.66s`, encoder `0.41s`, audio `1.63s`, vision `5.99s`, all fallback `0`
  - `gpu_model_matrix_cuda` passes on the same RTX 3060: dense `0.30s`, quantized `2.71s`, encoder `0.22s`, audio `0.83s`, vision `2.42s`

## Progress Update: Honest Fallback Accounting For Backend-Internal CPU Recovery

- Found a systemic accounting hole: both `wgpu_backend.rs` and `vulkan_backend.rs` contain dozens of `Err(err) if should_cpu_fallback(&err) => <cpu recovery>` sites that silently ran work on the CPU **without recording the fallback counter**. Only errors that escaped to `storage.rs` were counted, so `native_required` tests could pass while ops silently used the CPU.
- Fix: the backend-local `should_cpu_fallback` predicates now record into the same global counters used by `storage.rs` (`record_wgpu_cpu_fallback` / `record_vulkan_cpu_fallback`). Counting stays exact: internal recovery records once, propagated errors record once at the storage layer.
- What the honest accounting immediately exposed (all reproduced on RTX 3060, then fixed natively):
  1. `index_add` on both backends *always* silently fell back: the last-dim scatter-add kernels required ids rank == dst rank, but `index_add` passes one rank-1 id row. Fixed by mapping the broadcast case onto a zero ids-row-stride in the existing `set_rows_add` shader params on both backends — no new shaders.
  2. Vulkan `to_dtype`/unary on views with a storage offset (56 hits in the Qwen3 GGUF case): the ggml unary header only carries 16-bit misalign offsets, so the dispatch helper now materializes offset/strided views contiguously **on the GPU** before dispatch instead of recovering on CPU.
  3. Vulkan `affine` on strided inputs (4 hits in the Whisper case): ggml's `scale.comp` indexes linearly, so strided inputs previously either fell back to CPU or — when routed through the generic header — produced wrong results. Covered by the same GPU materialization in the shared dispatch helper. This also fixed a *silent wrong-result* hazard for strided `powf`/`clamp` which use the same linear-indexed shaders.
  4. WGPU `to_dtype U32->F32` (token-id casts in every decoder case): extended `cpy.wgsl` with `SRC_U32`/`SRC_I32`/`DST_U32` variants and widened the `copy_shader` matrix to the full f32/f16/u32/i32 cross product. The unsupported-cast error message now names the dtype pair.
- Re-certified after the changes on RTX 3060 (release, honest accounting):
  - `gpu_model_matrix_vulkan`: all five cases fallback `0`
  - `gpu_model_matrix_wgpu`: all five cases fallback `0`
  - 46-test smoke suite, 24-test nn ops suite, metamorphic/property suites: pass
- Remaining known dtype-conversion gaps (still recorded honestly as fallbacks, next work item): Vulkan F32/F16<->U8/U32/I64 integer casts and BF16<->integer compositions; WGPU U8/I64 casts (need packed-byte and split-word WGSL emulation).

## Progress Update: Full DType Conversion Matrix Closure (F32/F16/BF16/U8/U32/I64)

- Verified state before this slice (release probe on RTX 3060): Vulkan had 23 of 36 ordered cast pairs falling back to CPU, `wgpu` had 18.
- Vulkan closure:
  - new Candle-owned `convert.comp` (strided-aware via the ggml `generic_unary_head.glsl`), compiled into 17 SPIR-V variants covering every missing integer pair; float -> integer follows Rust `as` semantics (saturating, NaN -> 0), integer -> integer truncates.
  - `copy_spirv` now routes the full {F32,F16,U8,U32,I64} matrix; BF16 -> integer decomposes through the native bf16 -> f32 path entirely on GPU.
  - device creation now properly enables `shaderInt64`, Vulkan 1.1 16-bit storage, and Vulkan 1.2 8-bit storage / `shaderInt8` / `shaderFloat16` when the physical device supports them (the ggml shader corpus was already relying on several of these).
- WGPU closure:
  - `cpy.wgsl` gained `SRC_U32`/`SRC_I32`/`DST_U32` variants and `copy_shader` covers the full f32/f16/u32/i32 native-WGSL matrix.
  - new `run_emulated_cast` builder handles the dtypes WGSL cannot express: `U8` as four packed bytes per `u32` word (one thread packs each output word) and `I64` as lo/hi `u32` pairs with two's-complement splitting; same Rust `as` conversion semantics.
- Result (verified by release probe and new native-required tests on RTX 3060): **both backends now pass all 36 ordered conversion pairs natively with CPU-matching numerics**.
- New permanent coverage: `backend_smoke_{wgpu,vulkan}_dtype_conversion_matrix_native_only` runs the full 36-pair matrix under `native_required` with numeric comparison against the CPU reference.
- Known intentional divergence (tracked, not final): the storage layer remaps BF16 *destinations* to F16 on wgpu/Vulkan (`storage.rs:471-474`); BF16 sources convert natively. Removing this remap requires BF16-native op coverage and is a separate parity row.
- Found during this slice (not yet fixed): `Tensor::to_device` Vulkan -> CPU errors with "not implemented yet"; transfer parity is a separate row to close.

## Progress Update: GPU Test Concurrency Hardening

- The default parallel test runner raced concurrent wgpu/Vulkan device create/teardown and the process-global fallback counters, producing spurious `native_required` failures and an intermittent SIGSEGV in driver teardown.
- `candle-core/tests/support/mod.rs`: all backend cases now serialize on a global `FALLBACK_COUNTER_LOCK`; direct-device smoke tests acquire the same guard.
- `candle-nn/tests/ops.rs`: added `gpu_test_guard()`; every backend test fn takes it before device creation and holds it through device drop.
- Verified: full 46-test `backend_smoke_tests` suite and 24-test `candle-nn` ops suite now pass repeatedly under the default parallel runner on RTX 3060.

## Progress Update: `Tensor::to_device` Transfer Parity (wgpu/Vulkan)

- Closed the previously recorded gap: `Tensor::to_device` had no match arms for wgpu/Vulkan storage and bailed with "not implemented yet" for Vulkan -> CPU, CPU -> Vulkan, wgpu -> CPU, CPU -> wgpu, same-backend deep copies, and every cross-backend pair (e.g. CUDA -> Vulkan).
- `candle-core/src/tensor.rs::to_device` now routes:
  - `Cpu -> Wgpu/Vulkan` via `storage_from_cpu_storage` (device upload)
  - `Wgpu/Vulkan -> Cpu` via `to_cpu_storage` (device download)
  - `Wgpu -> Wgpu`, `Vulkan -> Vulkan` via the backend `transfer_to_device`
  - any remaining cross-backend pair (Cuda <-> Vulkan, Metal <-> wgpu, ...) by staging through host memory with `storage_from_cpu_storage_owned`
- BF16 dtype honesty: wgpu/Vulkan storage normalizes BF16 uploads to F16, so the resulting tensor dtype is now derived from the destination storage instead of copied from the source tensor; a CPU BF16 tensor moved to Vulkan reports `DType::F16` instead of lying about its storage.
- `dummy_wgpu_backend.rs`/`dummy_vulkan_backend.rs` gained `transfer_to_device` stubs so non-GPU builds keep compiling; all four feature combos (`none`, `wgpu`, `vulkan`, `wgpu,vulkan`) checked.
- New native-required coverage on RTX 3060 (both passing, fallback count 0):
  - `backend_smoke_vulkan_to_device_transfers_native_only`
  - `backend_smoke_wgpu_to_device_transfers_native_only`
  - covers CPU->GPU, GPU->CPU, same-device copy, strided-view roundtrip, u32/f16 dtype integrity, and the BF16->F16 normalization contract.
- Full regression state after the change: 30 Vulkan + 20 wgpu backend smoke tests pass on RTX 3060.

## Progress Update: Integer DType Closure For `cmp` / `where_cond` (u8/u32/i64)

- CUDA reference (`candle-kernels/src/binary.cu` BINARY_OP_OUT rows, `candle-kernels/src/ternary.cu` WHERE_OP rows) supports `u8`/`u32`/`i64` for compare and select; wgpu/Vulkan previously silently fell back to CPU for every non-float dtype.
- Vulkan closure (`candle-vulkan-kernels/src/candle-shaders/cmp.comp` + `build.rs`):
  - `cmp.comp` now compares in the source type (`A_TYPE`) instead of routing through `float`, so `u32 > 2^24` and `i64 > 2^53` keep exact integer semantics; new SPIR-V variants `cmp_u8`, `cmp_u32`, `cmp_i64` (uses `shaderInt64`).
  - `where_u8.comp` gained `where_u8_u8`, `where_u8_u32`, `where_u8_i64` variants; dispatch in `run_where_u8_cond` routes all five value dtypes.
- WGPU closure (`candle-core/src/wgpu_backend.rs`):
  - `custom_cmp_wgsl` emits dtype-specific load snippets: U8 unpacks bytes from u32 words, I64 compares lo/hi word pairs with two's-complement aware predicates (`i64_lt` = signed-high compare or equal-high/unsigned-low compare); U32 compares natively.
  - `custom_where_u8_wgsl` emits per-dtype main bodies: U8 packs four result bytes per output word, I64 selects lo/hi pairs; both stay on GPU with no float roundtrip.
- Test closure on RTX 3060 (native-required, fallback 0):
  - `backend_smoke_vulkan_int_cmp_where_native_only`
  - `backend_smoke_wgpu_int_cmp_where_native_only`
  - cases include mantissa-overflow values (`16_777_217u32`), sign boundaries, > 2^33 i64 magnitudes, strided/transposed views, and CPU-reference comparison for where_cond over all three integer dtypes.
- Full regression state: 31 Vulkan + 21 wgpu backend smoke tests pass on RTX 3060.

## Progress Update: Integer Strided Copy Closure (`contiguous` on u8/u32/i64 views)

- `copy_strided_src` for non-contiguous layouts previously routed every non-float dtype through a CPU download/upload roundtrip on both backends. This is the path behind `Tensor::contiguous()` on transposed/sliced integer tensors (token-id manipulation, masks, KV-cache bookkeeping).
- Vulkan closure: new same-dtype `convert_u8_u8` / `convert_u32_u32` / `convert_i64_i64` SPIR-V variants compiled from the existing Candle-owned `convert.comp` (strided-aware ggml unary header); `copy_spirv` routes the identity pairs and `copy_strided_src` accepts `U8/U32/I64` on the shader path.
- WGPU closure: `run_emulated_strided_copy_into` generates a WGSL kernel for the dtypes WGSL cannot address as scalars — U8 gathers four strided bytes per packed output word (with read-merge of trailing bytes in the final word so neighbors are never clobbered), I64 copies lo/hi word pairs; U32/I32 reuse the existing `cpy.wgsl` matrix.
- Test closure folded into `backend_smoke_{wgpu,vulkan}_int_cmp_where_native_only`: `t().contiguous()` round-trips for u8 (>127 values), u32 (mantissa-overflow 16_777_217), and i64 (negative sub-2^33) all asserted against exact expected values under native-required accounting.
- Regression state on RTX 3060: 31 Vulkan + 21 wgpu smoke tests, `gpu_property_tests` (both backends), `gpu_metamorphic_tests` (both backends) all pass.
