// Dense F32 GEMM via wgpu cooperative matrix (8x8 tensor-core tiles).
// Requires: enable wgpu_cooperative_matrix + Features::EXPERIMENTAL_COOPERATIVE_MATRIX.
// Workgroup = one subgroup (32) computing one 8x8 output tile; K stepped by 8.
//
// Binding convention matches warptile/reg_tile:
//   src0 = B^T (params.m = candle N rows, K cols) or physical (K,N) via stride_0k
//   src1 = A   (params.n = candle M rows, K cols)
//   dst[col * params.m + row]

enable wgpu_cooperative_matrix;

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32,
    stride_0k: u32,
    stride_1k: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

const TM: u32 = 8u;
const TN: u32 = 8u;
const TK: u32 = 8u;

// Stage tiles in workgroup memory for coopLoad (contiguous 8x8, stride 8).
var<workgroup> tile_a: array<f32, 64>; // src0 panel (B^T): 8 rows x 8 k
var<workgroup> tile_b: array<f32, 64>; // src1 panel (A):   8 rows x 8 k

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = local_id.x;

    let tiles_m = (params.m + TM - 1u) / TM;
    let tiles_n = (params.n + TN - 1u) / TN;
    let tiles_per_batch = tiles_m * tiles_n;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let batch_idx = wg_linear / tiles_per_batch;
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }
    let tile_in_batch = wg_linear % tiles_per_batch;
    let tile_m = tile_in_batch % tiles_m;
    let tile_n = tile_in_batch / tiles_m;

    let row_base = tile_m * TM;
    let col_base = tile_n * TN;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    // Accumulator C = A * B style in coop roles: we multiply left=src0 rows, right=src1.
    // Using role C for acc, A for src0 tile, B for src1 tile.
    var acc: coop_mat8x8<f32, C>;
    // Zero by loading a zeroed tile once at start.
    if (tid < 64u) {
        tile_a[tid] = 0.0;
    }
    workgroupBarrier();
    acc = coopLoad<coop_mat8x8<f32, C>>(&tile_a[0], 8u);

    for (var k0 = 0u; k0 < params.k; k0 += TK) {
        // Fill 8x8 tiles (64 elems) with 32 threads × 2 loads.
        for (var i = 0u; i < 2u; i++) {
            let elem = tid * 2u + i;
            let r = elem / TK;
            let c = elem % TK;
            let gr0 = row_base + r;
            let gk = k0 + c;
            var v0 = 0.0;
            if (gr0 < params.m && gk < params.k) {
                v0 = src0[src0_batch + gr0 * params.stride_01 + gk * params.stride_0k];
            }
            tile_a[r * TK + c] = v0;

            let gr1 = col_base + r;
            var v1 = 0.0;
            if (gr1 < params.n && gk < params.k) {
                v1 = src1[src1_batch + gr1 * params.stride_11 + gk * params.stride_1k];
            }
            tile_b[r * TK + c] = v1;
        }
        workgroupBarrier();

        let a_mat = coopLoad<coop_mat8x8<f32, A>>(&tile_a[0], 8u);
        // B is loaded as transposed relative to row-major K-inner storage so
        // MMA does (8xm)×(kx8) correctly: tile_b is row=global_n, col=k → load T.
        let b_mat = coopLoadT<coop_mat8x8<f32, B>>(&tile_b[0], 8u);
        acc = coopMultiplyAdd(a_mat, b_mat, acc);

        workgroupBarrier();
    }

    // Store acc into tile_a then scatter to dst.
    coopStore(acc, &tile_a[0], 8u);
    workgroupBarrier();

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst_batch = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var i = 0u; i < 2u; i++) {
        let elem = tid * 2u + i;
        let r = elem % TM; // row along params.m
        let c = elem / TM; // col along params.n  — depends on store layout
        // coopStore writes row-major with stride 8: index = row * 8 + col
        // We stored with stride 8 as (row_m, col_n) → idx = r * 8 + c if r is row.
        // Parse as row-major: r = elem / 8, c = elem % 8
        let rr = elem / TN;
        let cc = elem % TN;
        let global_row = row_base + rr;
        let global_col = col_base + cc;
        if (global_row < params.m && global_col < params.n) {
            dst[dst_batch + global_col * params.m + global_row] = tile_a[rr * TN + cc];
        }
    }
}
