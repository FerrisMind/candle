// Natural-layout context GEMV for the m == 1 attention ctx matmul:
//   out[b, i] = sum_l A[b, l] * V[b, l, i]
// where V is the NATURAL KV-cache layout (batch, l, head_dim) with head_dim
// contiguous inner. This is the wgpu port of the Vulkan `ctx_gemv_f32.comp`
// (eaab92cc); the score matmul keeps `batched_gemv_f32` (reduces over head_dim at
// fixed position), the CONTEXT matmul reduces over the kv length l instead, so
// V here is read in its native strided-over-l layout — no per-step V^T
// materialization. A `narrow`ed prefix of a grown KV-cache backing (capacity >
// live length, batch stride == capacity stride) routes here through the same
// `batch_stride` params (stride-aware, port of the same Vulkan change).
//
// Thread mapping (deterministic, no subgroup intrinsics — portable):
//   workgroup = K_SPLIT warps x 32 lanes = 256 threads. lane -> head_dim column
//   within a 32-wide tile; wslice -> which l-slice. Grid.x tiles the head_dim
//   output (32 columns per warp); grid.y = batch. Each warp reads V coalesced
//   (32 consecutive head_dim floats) and A[l] broadcast, then K_SPLIT partials
//   are summed through workgroup memory in fixed index order.

struct CtxGemvParams {
    k_dim: u32, // reduction length l (== the matmul's K == kv length)
    n_dim: u32, // output length i (== the matmul's N == head_dim)
    batch: u32,
    offset_a: u32,
    offset_b: u32,
    offset_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
};

@group(0) @binding(0) var<storage, read> src_a: array<f32>;
@group(0) @binding(1) var<storage, read> src_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: CtxGemvParams;

// Number of warps (l-slices) per workgroup. 8 -> 256 threads.
const K_SPLIT: u32 = 8u;
// Output columns handled per warp (== coalesced read width along head_dim).
const I_GROUP: u32 = 32u;

var<workgroup> sh_part: array<f32, K_SPLIT * I_GROUP>;

@compute @workgroup_size(K_SPLIT * I_GROUP)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Flatten the 2D grid (x = head_dim tile, y = batch) under the per-dim cap
    // without changing batch semantics (mirrors the other wgpu compute kernels).
    let output_groups = (params.n_dim + I_GROUP - 1u) / I_GROUP;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= params.batch) {
        return;
    }
    let out_group = wg_linear % output_groups;

    let wslice = local_id.x / I_GROUP; // which l-slice (0..K_SPLIT-1)
    let lane = local_id.x % I_GROUP; // head_dim column within tile
    let out_idx = out_group * I_GROUP + lane; // head_dim output column
    let valid = out_idx < params.n_dim;

    let a_base = params.offset_a + batch_idx * params.batch_stride_a;
    let b_base = params.offset_b + batch_idx * params.batch_stride_b;
    let d_base = params.offset_d + batch_idx * params.batch_stride_d;

    // Contiguous l-range per warp: every worker owns a slice of the reduction.
    let l_sz = params.k_dim / K_SPLIT;
    let l_start = wslice * l_sz;
    // Last slice absorbs the remainder so every reduction element is covered.
    let l_end = select(l_start + l_sz, params.k_dim, wslice == K_SPLIT - 1u);

    var acc: f32 = 0.0f;
    if (valid) {
        for (var ll = l_start; ll < l_end; ll++) {
            // src_b[b_base + ll*n_dim + out_idx]: adjacent lanes -> adjacent i.
            acc = fma(src_a[a_base + ll], src_b[b_base + ll * params.n_dim + out_idx], acc);
        }
    }

    sh_part[wslice * I_GROUP + lane] = acc;
    workgroupBarrier();

    if (wslice == 0u) {
        var sum = sh_part[lane];
        for (var t = 1u; t < K_SPLIT; t++) {
            sum = sum + sh_part[t * I_GROUP + lane];
        }
        if (valid) {
            dst[d_base + out_idx] = sum;
        }
    }
}