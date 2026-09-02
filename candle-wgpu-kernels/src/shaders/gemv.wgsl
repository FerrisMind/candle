// Dense m == 1 matrix-vector product (GEMV): out[col] = sum_k x[k] * W[k, col].
//
// This is the fast path for decode-time weight matmuls (qkv/o_proj/gate_up/down/
// lm_head), which are ALL M == 1. The generic tiled GEMM (mul_mat.wgsl) dispatches
// one thread per output element but reads the transposed RHS with a per-row stride,
// which is catastrophic for m == 1 (measured ~5-12 GFLOP/s). This kernel:
//   - one thread per output column (n threads total),
//   - reads the activation x[k] broadcast across the warp (all lanes read x[kk]),
//   - reads W coalesced across the warp (32 consecutive columns at a fixed kk),
//   - accumulates in f32 single-accumulator, sequential FMA order (identical
//     accumulation order to the generic mul_mat path, so numerics match within the
//     backend parity tolerance and stay deterministic run-to-run).
//
// Layout assumptions (contiguous, guaranteed by the dispatch guard):
//   x : (1, k)        -> x[offset_x + kk]
//   W : (k, n)        -> W[offset_w + col * stride_n + kk * stride_k]
//   dst : (1, n)      -> dst[offset_d + col]
// The guard requires unit column stride (stride_n == 1) so a warp's 32 consecutive
// output columns read contiguous W addresses at each kk (coalesced).

enable f16;

struct GemvParams {
    k: u32,
    n: u32,
    offset_x: u32,
    offset_w: u32,
    stride_k: u32,
    stride_n: u32,
    offset_d: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
};

@group(0) @binding(0) var<storage, read> x: array<SRC_TYPE>; // (1, k) activation
@group(0) @binding(1) var<storage, read> w: array<SRC_TYPE>; // (k, n) weight, column-contiguous
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // (1, n)
@group(0) @binding(3) var<uniform> params: GemvParams;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let col = wg_id.x * WG_SIZE + lid.x;
    if (col >= params.n) {
        return;
    }
    let w_col = params.offset_w + col * params.stride_n;
    let x_base = params.offset_x;

    var acc = 0.0f;
    var kk = 0u;
    for (; kk + 3u < params.k; kk = kk + 4u) {
        // 4 in-flight FMAs into the SAME accumulator preserve the sequential
        // accumulation order while giving the scheduler independent loads to hide.
        acc = fma(f32(x[x_base + kk]), f32(w[w_col + kk * params.stride_k]), acc);
        acc = fma(f32(x[x_base + kk + 1u]), f32(w[w_col + (kk + 1u) * params.stride_k]), acc);
        acc = fma(f32(x[x_base + kk + 2u]), f32(w[w_col + (kk + 2u) * params.stride_k]), acc);
        acc = fma(f32(x[x_base + kk + 3u]), f32(w[w_col + (kk + 3u) * params.stride_k]), acc);
    }
    for (; kk < params.k; kk = kk + 1u) {
        acc = fma(f32(x[x_base + kk]), f32(w[w_col + kk * params.stride_k]), acc);
    }
    dst[params.offset_d + col] = acc;
}
