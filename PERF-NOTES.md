# PERF-NOTES — GPU backend performance status (post quick-win fix series)

Hardware used for numbers below: RTX 3060 12GB (sm_86), driver 596.49, CUDA 12.8,
Windows/WDDM. Full `candle-core` bench_main suite, `--sample-size 30`, same process
runs cuda+vulkan+wgpu backends; medians from criterion. Rev: wgpu/vulkan branch,
8-commit fix series 65b33be2..5f5eb526 (see "Fixed this session").

## Fixed this session (branch wgpu/vulkan, all measured on RTX 3060)

| fix | commit | before → after | note |
|---|---|---|---|
| vulkan upload staging path | 65b33be2 | copy_upload 1.38ms → 355µs (−76%); from_slice(2MB) 1.3ms → 211µs; zeros 190x | BAR1 write-combined → cached host RAM staging; cmd_fill_buffer zeros (530µs → 1.8µs); borrowed cpu_storage_to_bytes |
| wgpu copy2d gather | b2a4f82d | copy2d_cat 1.01ms → 50–82µs (−92%..−95%) | one strided gather dispatch instead of 512 copy2d commands |
| wgpu strided reduce | 627f6b8e | reduce/arg_reduce *_strided 12ms → 78–216µs (−99%); contiguous arg −60% | strided reduce kernels + length-1-dim identity (Sum/Max/Min → clone, Arg → zeros) |
| wgpu upload copies | 8d7b1d35 | copy_upload 3.2ms → 1.2–1.4ms (−62%) | removed 3 redundant host copies (borrowed bytes, Cow padded write, direct slice storage) |
| wgpu bf16 affine (packed) | 39c6a65b | affine_bf16 200µs → 28.3µs (−85%) | also fixed real bug: bf16_scale_shader params shorter than ScaleParams → kernel silently no-op; in-place aliasing branch removed |
| wgpu where bf16 | b23c6928 | where_cond_bf16 412µs → 43.2µs (−89%) | raw bit-selection in i16_main (packed halves), bf16_where_via_f32 deleted |
| wgpu qmatmul F32/F16 K-major GEMV | 32f718ac | qmatmul F32/F16 1.0ms → 159µs (−83%); dense m=1 matmul 1.16ms → 217µs (5.3x) | m==1 rank-2 guard routes K-major storage to gemv.wgsl (generic strides) |
| wgpu conv_transpose2d gather | a1644634 | conv_tr f32/f16 1.06ms → 24.5–24.7µs (−97.9%), bf16 ~25µs | one gather dispatch replaces host ids/mask build+upload + matmul/mul/index_add/zeros chain; tensor-level wgpu_decomp disabled. cuda ≈ 26µs → parity |
| vulkan linear_large padded-B ALIGNED GEMM | 5f5eb526 | matmul_linear_large 24.1ms → 14.1–14.5ms (−41%), 2.94x → 1.62x cuda | pad candle-M rows to a BN=64 multiple (one contiguous copy), run exact-fp32 ALIGNED kernel (unguarded vec4 A/B); store guards (p.N = real M) skip the pad tail, no zero-fill needed |

Fixed earlier in the campaign (already in history): vulkan fused affine/sqrt
(6.5–7.2x and 3.6x holes closed), vulkan fast upload path superseded by 65b33be2.

## cuda vs vulkan vs wgpu (µs, medians; vk/cu = vulkan over cuda)

### GEMM
| bench | cuda | vulkan | wgpu | vk/cu | wg/cu |
|---|---|---|---|---|---|
| matmul_square_1024 | 293.5 | 213.9 | 4673 | **0.73x** | 15.9x |
| matmul_square_512 | 46.9 | 39–52 (WDDM drift) | 712 | ~0.9–1.1x | 15.2x |
| matmul_gemv | 55.3 | 54.8 | 94.7 | 0.99x | 1.71x |
| matmul_linear_large | 8733 | **14100–14500** (was 25698) | 165130 | **1.62x** (was 2.94x) | 18.9x |
| matmul_attn_4d (large/small) | 2246/2282 | 3186/3196 | 16300 | 1.42x | **7.2x** |
| matmul_batch_1000 | 737 | 995–1200 | 5278 | 1.35–1.63x | **7.2x** |

### quantized matmul (Q8_0..Q2K, f32/f16)
- **vulkan k-quants 4x faster than CUDA** (0.24–0.27x of cuda across Q8_0..Q2K, q2k 0.46x).
- wgpu k-quants ≈ cuda parity (0.65–1.31x, ambient ±20%).
- wgpu qmatmul F32/F16: **FIXED** (1.0ms → 159µs, K-major GEMV route; ~5–10x cuda still,
  but the 34–54x hole is closed — remaining gap is the m==1 GEMV latency ceiling, see future work).

### wgpu remaining holes vs cuda (post-fix, ordered by severity)
| hole | wg/cu | status |
|---|---|---|
| matmul_linear_large / square / attn / batch | 7.2–18.9x | open (dense tiled GEMM; microbench ratios partly measurement-inflated — cuda async submit vs wgpu blocking sync; honest gap visible at model level: llama f32 prefill) |
| broadcast_add_contiguous_f32 | 3.9x | open |
| affine_f32 | 2.3x | open (79µs; f16 hub still slower than the packed bf16 path) |
| matmul_gemv | 1.71x | open (K-major m==1: 156µs — one thread per output, latency-bound; workgroup-reduction variant is future work) |
| qmatmul F32/F16 (m==1 GEMV ceiling) | ~5–10x | partially closed (was 34–54x) |

### vulkan remaining weak spots vs cuda (post-fix)
| spot | vk/cu | status |
|---|---|---|
| matmul_batch_1000 / attn_4d (rank-3 batched, small m/n/k) | 1.35–1.63x | open (per-batch tiles; split-K not applicable — 6.5k+ workgroups already saturate) |
| where_cond bf16 | 1.76x | open |

### vulkan wins vs cuda
- k-quant qmatmul: 4x faster.
- f32 square GEMM: 1.2–1.3x faster (cm1 coopmat; **caveat below**); strided
  reduce/arg_reduce 1.1–1.9x faster; where/masked_fill f16/f32 6–8x faster;
  cat/copy_strided/contiguous 3–7x faster; random_uniform 2x faster.

## Model-level (greedy, same machine; coherent-output checks re-verified after the fix series)

| model | cuda | vulkan | wgpu |
|---|---|---|---|
| qwen3-0.6B Q8_0 prefill (512 tok) | 65.0 tok/s | **137.1** | **214.5** |
| qwen3-0.6B Q8_0 decode | 51.8 tok/s | 57.8–70.8 | 41.3–49.7 |
| llama-3.2-1B f32 prefill | **1025.7** | 318.6 | 135.4 |
| llama-3.2-1B f32 decode | **5.70** | 3.20 | 4.63 |

Post-fix verification (short prompts): qwen3 Q8_0 vulkan 81.8 prefill (13 tok) /
57.8 decode; wgpu 43.2 / 41.3; llama f32 vulkan 3.20 decode, wgpu 4.63 — all
coherent, no regressions vs the pre-fix table (small-prompt runs are noisier than
the 512-tok methodology above).

## Gotchas discovered this session

- **Vulkan f32 coopmat (cm1) is TF32-class precision**: `matmul_f32_f32_aligned_cm1`
  (selected by default for 64-aligned F32 GEMMs with cooperative_matrix) measured
  max_abs 1.36e-1 on 1024^3 (±1 inputs), vs 5.1e-4 for the exact scalar
  `matmul_f32_f32_aligned_fp32`. The unaligned cm1 variant is already opt-in
  (CANDLE_VULKAN_F32_UNALIGNED_COOPMAT=1) for this reason; the ALIGNED cm1 default
  still is not — flagged for follow-up (flipping it trades square_1024 f32
  precision for ~1.3x speed; decide per-use-case).
- **Guarded-B ALIGNED variant loses the win**: per-element guard branches on the B
  loads eat the vec4-load advantage (probe ~33ms vs 24ms scalar on linear_large).
  Padding the row count instead costs one contiguous copy (~0.6ms) and keeps every
  load unguarded/in-bounds.
- **wgpu conv_transpose2d tensor-level decomp intercepted F32/F16** before the
  storage hook: the storage-level scatter chain (host ids/mask per call) and the
  tensor-level decomp were both live; disabling the decomp routes everything to
  the gather kernel.
- **WDDM ambient variance is large**: same-process criterion re-runs of untouched
  benches swing ±10–30% (copy2d_cat 82µs → 435µs → 50µs across three consecutive
  runs; square_512 39→52µs). Never trust a single change% for <100µs benches —
  re-run before concluding a regression.
- Criterion microbench cross-backend ratios vs cuda are inflated: cuda bench
  measures async submit, wgpu/vulkan block on GPU completion. Model-level numbers
  are the honest gap.

## Future work (ordered by expected impact)

1. wgpu dense tiled GEMM (square/attn/batch 7–19x micro, llama f32 prefill 7.6x
   model-level): reg_tile/coop64/warptile infra exists; needs shape-driven tile
   selection + the transposed-view-RHS materialize fusion (dense m=16/64/1024
   probe: 1.0/0.8/4.9ms).
2. wgpu K-major m==1 GEMV workgroup-reduction variant (currently one thread per
   output, 1024 threads total for 1024x1024 — latency-bound at 156µs).
3. Vulkan rank-3 batched matmul (batch_1000 1.35x, attn_4d 1.42x): per-batch
   coopmat/tiles; split-K irrelevant (parallelism already high).
4. Decide the ALIGNED cm1 TF32 default question (see gotcha) or add an
   exact-scalar override env for precision-critical callers.
5. wgpu affine_f32 (2.3x): pack f32 loads like the bf16 path.
6. Vulkan where_cond bf16 (1.76x).

## Notes / infra
- `cuda_affine_fp8` bench panicked on sm_86: CUDA PTX guards fp8 kernels behind
  `__CUDA_ARCH__ >= 890` (Ada). Bench now skips fp8 on CUDA devices (affine.rs guard);
  vulkan/wgpu compile fp8 unconditionally and bench fine. Not a code bug — a hardware
  capability guard; on sm_89+ CUDA the bench would run.
- RTX 3060 used for this table (machine's current GPU); earlier campaign tables used a
  16GB Ada-class card — do not mix numbers across the two.
- Probes committed alongside fixes: vk_upload_probe, wgpu_affine_probe,
  wgpu_qmatmul_probe, wgpu_convtr_probe, vk_gemm_probe (chrome-trace / parity +
  timing one-offs).
