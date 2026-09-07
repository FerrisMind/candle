# PERF-NOTES — GPU backend performance status (fresh, post-main-merge)

Hardware used for numbers below: RTX 3060 12GB (sm_86), driver 596.49, CUDA 12.8,
Windows/WDDM. Full `candle-core` bench_main suite, `--sample-size 30`, same process
runs cuda+vulkan+wgpu backends; medians from criterion. Rev: merge 36e84faa + bf5d99ec.

## cuda vs vulkan vs wgpu (µs, medians; vk/cu = vulkan over cuda)

### GEMM
| bench | cuda | vulkan | wgpu | vk/cu | wg/cu |
|---|---|---|---|---|---|
| matmul_square_1024 | 293.5 | 213.9 | 4542 | **0.73x** | 15.5x |
| matmul_square_512 | 46.9 | 39.3 | 706 | **0.84x** | 15.0x |
| matmul_gemv | 55.3 | 57.2 | 95.0 | 1.03x | 1.72x |
| matmul_linear_large | 8733 | 25698 | 163217 | 2.94x | **18.7x** |
| matmul_attn_4d (large/small) | 2246/2282 | 3282/3306 | ~16000 | 1.45x | **7.1x** |
| matmul_batch_1000 | 737 | 1200 | 5126 | 1.63x | **7.0x** |

### quantized matmul (Q8_0..Q2K, f32/f16)
- **vulkan k-quants 4x faster than CUDA** (0.24–0.27x of cuda across Q8_0..Q2K, q2k 0.46x).
- wgpu k-quants ≈ cuda parity (0.65–1.31x).
- **wgpu qmatmul_f32/f16 hole: 34–54x slower than cuda** (~1.0 ms vs 16–29 µs) — unchanged
  after merge; the wgpu QMatMul F32/F16 path bypasses the fast quant kernels.

### wgpu holes vs cuda (unchanged by the merge, ordered by severity)
| hole | wg/cu |
|---|---|
| conv_transpose2d f32/f16/bf16 | 38.6–40.2x |
| qmatmul f32/f16 | 34–54x |
| copy2d_cat_f32 | 31.6x |
| matmul_linear_large / square / attn / batch | 7–18.7x |
| copy_upload_f32 | 17.7x |
| reduce/arg_reduce *_strided | 3.7–5.6x |
| broadcast_add_contiguous_f32 | 3.9x |
| where_cond bf16 family | 1.5–4.6x |
| affine_f32 | 2.3x |
| matmul_gemv | 1.72x |

### vulkan weak spots vs cuda
| spot | vk/cu |
|---|---|
| affine f16/bf16 (mul+add) | 6.5–7.2x |
| copy_upload_f32 | 13.1x (WDDM staging path; download 2.1x) |
| matmul_linear_large | 2.94x |
| sqrt f16/bf16 | 3.6x |
| matmul_batch_1000 / attn_4d | 1.45–1.63x |
| where_cond bf16 | 1.76x |

### vulkan wins vs cuda
- k-quant qmatmul: 4x faster.
- f32 square GEMM: 1.2–1.3x faster; strided reduce/arg_reduce 1.1–1.9x faster;
  where/masked_fill f16/f32 6–8x faster; cat/copy_strided/contiguous 3–7x faster;
  random_uniform 2x faster.

## Model-level (targeted, greedy, same machine)

| model (512-token prefill / 64-token decode) | cuda | vulkan | wgpu |
|---|---|---|---|
| qwen3-0.6B Q8_0 GGUF prefill | 65.0 tok/s | **137.1** | **214.5** |
| qwen3-0.6B Q8_0 decode | 51.8 tok/s | **70.8** | 43.9 |
| llama-3.2-1B f32 prefill | **1025.7** | 318.6 | 135.4 |
| llama-3.2-1B f32 decode | **5.70** | 3.29 | 4.69 |

Interpretation: quantized (GGUF) workloads — both fork backends beat CUDA on this GPU.
f32 llama prefill tracks the f32 GEMM holes (wgpu 15x, vulkan 1.3–2.9x on the involved shapes).
The earlier one-off "vulkan prefill 39.6 tok/s" reading was WDDM ambient noise — the
targeted bench reproduces 137 tok/s consistently.

## Notes / infra
- `cuda_affine_fp8` bench panicked on sm_86: CUDA PTX guards fp8 kernels behind
  `__CUDA_ARCH__ >= 890` (Ada). Bench now skips fp8 on CUDA devices (affine.rs guard);
  vulkan/wgpu compile fp8 unconditionally and bench fine. Not a code bug — a hardware
  capability guard; on sm_89+ CUDA the bench would run.
- RTX 3060 used for this table (machine's current GPU); earlier campaign tables used a
  16GB Ada-class card — do not mix numbers across the two.
