// Batched matrix-vector product for the m == 1 case: C[b, i] = sum_k A[b, k] * B[b, i, k]
// (one batch-vector A per batch, one full (n x k) matrix B per batch).
//
// This is the wgpu port of the Vulkan kernel `batched_gemv_f32.comp` (B10). The
// attention score/ctx matmuls are m == 1 over batch == num_heads. Previously the
// generic GEMM dispatched one large workgroup per output element, which is
// catastrophic for m == 1. This kernel processes ALL batches in one dispatch and
// reads B once, coalesced along K. Algorithm is identical to the Vulkan version
// (deterministic f32 FMA order; shared-memory reduction, no subgroup intrinsics),
// so numerics match the CPU reference within the parity tolerance.
//
// Layout assumptions (contiguous, guaranteed by the dispatch guard):
//   A : (batch, k)      -> src_a[offset_a + batch_stride_a * b + kk]
//   B : (batch, n, k)   -> src_b[offset_b + batch_stride_b * b + out_idx * k_dim + kk]
//   D : (batch, n)      -> dst[offset_d + batch_stride_d * b + out_idx]
//
// Workgroup = 128 threads = 4 "output groups" (one warp/32 lanes per output column).
// Within an output group the K reduction is split across the 32 lanes (coalesced
// B reads), then the 32 partials are reduced through workgroup memory. This gives a
// deterministic accumulation order and high occupancy even when n is small
// (ctx matmul n == head_dim == 128) but k is large (k == seq_len).

struct BatchedGemvParams {
    k_dim: u32,
    n_dim: u32,
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
@group(0) @binding(3) var<uniform> params: BatchedGemvParams;

// Number of output groups (warps) per workgroup == 128 / 32.
const OUT_GROUPS: u32 = 4u;
// Group size used for the coalesced K split.
const K_GROUP: u32 = 32u;
const WG_SIZE: u32 = OUT_GROUPS * K_GROUP; // 128

// Flattened as [group][lane]: each lane's partial within an output group.
var<workgroup> sh_part: array<f32, OUT_GROUPS * K_GROUP>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let group = local_id.x / K_GROUP; // which output column (0..OUT_GROUPS-1)
    let lane = local_id.x % K_GROUP; // which K slice

    // Flatten the 2D grid (x = output-column tile, y = batch) so the dispatcher may
    // split the total under the workgroup-per-dimension cap without changing the
    // batch semantics (mirrors the other wgpu compute kernels).
    let output_groups = (params.n_dim + OUT_GROUPS - 1u) / OUT_GROUPS;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= params.batch) {
        return;
    }

    let out_idx = (wg_linear % output_groups) * OUT_GROUPS + group;
    let valid = out_idx < params.n_dim;

    let a_base = params.offset_a + batch_idx * params.batch_stride_a;
    let b_base = params.offset_b + batch_idx * params.batch_stride_b;
    let d_base = params.offset_d + batch_idx * params.batch_stride_d;

    var acc: f32 = 0.0f;
    for (var kk = lane; kk < params.k_dim; kk += K_GROUP) {
        // Guard the B read so a partial output tile never touches out-of-range rows.
        let bv = select(0.0f, src_b[b_base + out_idx * params.k_dim + kk], valid);
        acc = fma(src_a[a_base + kk], bv, acc);
    }

    sh_part[group * K_GROUP + lane] = acc;
    workgroupBarrier();

    if (lane == 0u) {
        var sum = sh_part[group * K_GROUP + 0u];
        for (var t = 1u; t < K_GROUP; t++) {
            sum = sum + sh_part[group * K_GROUP + t];
        }
        if (valid) {
            dst[d_base + out_idx] = sum;
        }
    }
}
