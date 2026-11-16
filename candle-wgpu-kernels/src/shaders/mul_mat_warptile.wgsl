// Dense F32 warptile GEMM inspired by ggml-vulkan mul_mm (non-coopmat path).
// Workgroup: 128 threads covering a 64x64 output tile with BK=32.
// Per-thread register tile: TM=4 x TN=8 (=32 outputs); 128*32 = 4096 = 64*64.
// Inner K loop steps by 4 to cut loop overhead.
//
// Binding convention matches mul_mat_reg_tile:
//   src0 = B^T  (params.m rows = candle N, K cols)  — or physical (K,N) via stride_0k
//   src1 = A    (params.n rows = candle M, K cols)
//   dst[col * params.m + row] with col=candle M, row=candle N

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32, // candle N
    n: u32, // candle M
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

// Match candle-wgpu bind-group layout (storage buffers are read_write).
@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const TM: u32 = 4u;
const TN: u32 = 8u;
const THREADS_M: u32 = BM / TM; // 16
const THREADS_N: u32 = BN / TN; // 8
const TOTAL_THREADS: u32 = THREADS_M * THREADS_N; // 128
const BK_PAD: u32 = BK + 1u;

var<workgroup> sh_a: array<f32, BM * BK_PAD>;
var<workgroup> sh_b: array<f32, BN * BK_PAD>;

fn wt_store(base: u32, row: u32, col: u32, a0: f32, a1: f32, a2: f32, a3: f32) {
    if (col < params.n) {
        let b = base + col * params.m + row;
        if (row < params.m) { dst[b] = a0; }
        if (row + 1u < params.m) { dst[b + 1u] = a1; }
        if (row + 2u < params.m) { dst[b + 2u] = a2; }
        if (row + 3u < params.m) { dst[b + 3u] = a3; }
    }
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = local_id.x;
    let local_m = tid % THREADS_M;
    let local_n = tid / THREADS_M;

    let tiles_m = (params.m + BM - 1u) / BM;
    let tiles_n = (params.n + BN - 1u) / BN;
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

    let row_base = tile_m * BM;
    let col_base = tile_n * BN;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    // Unrolled 4x8 register tile (32 named f32 accumulators). The previous
    // array<array<f32, TN>, TM> was dynamically indexed, so Naga lowered it to
    // a Function-local array and every FMA round-tripped through local memory
    // (measured ~0.4 TFLOP/s through wgpu on an RTX 3060, 3% of f32 peak).
    // Named scalars keep the accumulators in registers; the 64x64 workgroup
    // tile, BK=32 staging and bounds handling are unchanged.
    let out_row = row_base + local_m * TM;
    let out_col = col_base + local_n * TN;
    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc02 = 0.0;
    var acc03 = 0.0;
    var acc04 = 0.0;
    var acc05 = 0.0;
    var acc06 = 0.0;
    var acc07 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var acc16 = 0.0;
    var acc17 = 0.0;
    var acc20 = 0.0;
    var acc21 = 0.0;
    var acc22 = 0.0;
    var acc23 = 0.0;
    var acc24 = 0.0;
    var acc25 = 0.0;
    var acc26 = 0.0;
    var acc27 = 0.0;
    var acc30 = 0.0;
    var acc31 = 0.0;
    var acc32 = 0.0;
    var acc33 = 0.0;
    var acc34 = 0.0;
    var acc35 = 0.0;
    var acc36 = 0.0;
    var acc37 = 0.0;

    let loads_per = (BM * BK) / TOTAL_THREADS; // 16

    for (var k0 = 0u; k0 < params.k; k0 += BK) {
        for (var i = 0u; i < loads_per; i++) {
            let elem = tid * loads_per + i;
            let r = elem / BK;
            let c = elem % BK;
            let gr = row_base + r;
            let gk = k0 + c;
            var v = 0.0;
            if (gr < params.m && gk < params.k) {
                v = src0[src0_batch + gr * params.stride_01 + gk * params.stride_0k];
            }
            sh_a[r * BK_PAD + c] = v;
        }
        for (var i = 0u; i < loads_per; i++) {
            let elem = tid * loads_per + i;
            let r = elem / BK;
            let c = elem % BK;
            let gr = col_base + r;
            let gk = k0 + c;
            var v = 0.0;
            if (gr < params.n && gk < params.k) {
                v = src1[src1_batch + gr * params.stride_11 + gk * params.stride_1k];
            }
            sh_b[r * BK_PAD + c] = v;
        }

        workgroupBarrier();

        let k_end = min(BK, params.k - k0);
        var kk = 0u;
        for (; kk + 4u <= k_end; kk += 4u) {
            let a_base = (local_m * TM) * BK_PAD + kk;
            let a00 = sh_a[a_base];
            let a01 = sh_a[a_base + 1u];
            let a02 = sh_a[a_base + 2u];
            let a03 = sh_a[a_base + 3u];
            let a10 = sh_a[a_base + BK_PAD];
            let a11 = sh_a[a_base + BK_PAD + 1u];
            let a12 = sh_a[a_base + BK_PAD + 2u];
            let a13 = sh_a[a_base + BK_PAD + 3u];
            let a20 = sh_a[a_base + 2u * BK_PAD];
            let a21 = sh_a[a_base + 2u * BK_PAD + 1u];
            let a22 = sh_a[a_base + 2u * BK_PAD + 2u];
            let a23 = sh_a[a_base + 2u * BK_PAD + 3u];
            let a30 = sh_a[a_base + 3u * BK_PAD];
            let a31 = sh_a[a_base + 3u * BK_PAD + 1u];
            let a32 = sh_a[a_base + 3u * BK_PAD + 2u];
            let a33 = sh_a[a_base + 3u * BK_PAD + 3u];
            let b_base = (local_n * TN) * BK_PAD + kk;
            acc00 = acc00 + (a00 * sh_b[b_base] + a01 * sh_b[b_base + 1u] + a02 * sh_b[b_base + 2u] + a03 * sh_b[b_base + 3u]);
            acc01 = acc01 + (a00 * sh_b[b_base + BK_PAD] + a01 * sh_b[b_base + BK_PAD + 1u] + a02 * sh_b[b_base + BK_PAD + 2u] + a03 * sh_b[b_base + BK_PAD + 3u]);
            acc02 = acc02 + (a00 * sh_b[b_base + 2u * BK_PAD] + a01 * sh_b[b_base + 2u * BK_PAD + 1u] + a02 * sh_b[b_base + 2u * BK_PAD + 2u] + a03 * sh_b[b_base + 2u * BK_PAD + 3u]);
            acc03 = acc03 + (a00 * sh_b[b_base + 3u * BK_PAD] + a01 * sh_b[b_base + 3u * BK_PAD + 1u] + a02 * sh_b[b_base + 3u * BK_PAD + 2u] + a03 * sh_b[b_base + 3u * BK_PAD + 3u]);
            acc04 = acc04 + (a00 * sh_b[b_base + 4u * BK_PAD] + a01 * sh_b[b_base + 4u * BK_PAD + 1u] + a02 * sh_b[b_base + 4u * BK_PAD + 2u] + a03 * sh_b[b_base + 4u * BK_PAD + 3u]);
            acc05 = acc05 + (a00 * sh_b[b_base + 5u * BK_PAD] + a01 * sh_b[b_base + 5u * BK_PAD + 1u] + a02 * sh_b[b_base + 5u * BK_PAD + 2u] + a03 * sh_b[b_base + 5u * BK_PAD + 3u]);
            acc06 = acc06 + (a00 * sh_b[b_base + 6u * BK_PAD] + a01 * sh_b[b_base + 6u * BK_PAD + 1u] + a02 * sh_b[b_base + 6u * BK_PAD + 2u] + a03 * sh_b[b_base + 6u * BK_PAD + 3u]);
            acc07 = acc07 + (a00 * sh_b[b_base + 7u * BK_PAD] + a01 * sh_b[b_base + 7u * BK_PAD + 1u] + a02 * sh_b[b_base + 7u * BK_PAD + 2u] + a03 * sh_b[b_base + 7u * BK_PAD + 3u]);
            acc10 = acc10 + (a10 * sh_b[b_base] + a11 * sh_b[b_base + 1u] + a12 * sh_b[b_base + 2u] + a13 * sh_b[b_base + 3u]);
            acc11 = acc11 + (a10 * sh_b[b_base + BK_PAD] + a11 * sh_b[b_base + BK_PAD + 1u] + a12 * sh_b[b_base + BK_PAD + 2u] + a13 * sh_b[b_base + BK_PAD + 3u]);
            acc12 = acc12 + (a10 * sh_b[b_base + 2u * BK_PAD] + a11 * sh_b[b_base + 2u * BK_PAD + 1u] + a12 * sh_b[b_base + 2u * BK_PAD + 2u] + a13 * sh_b[b_base + 2u * BK_PAD + 3u]);
            acc13 = acc13 + (a10 * sh_b[b_base + 3u * BK_PAD] + a11 * sh_b[b_base + 3u * BK_PAD + 1u] + a12 * sh_b[b_base + 3u * BK_PAD + 2u] + a13 * sh_b[b_base + 3u * BK_PAD + 3u]);
            acc14 = acc14 + (a10 * sh_b[b_base + 4u * BK_PAD] + a11 * sh_b[b_base + 4u * BK_PAD + 1u] + a12 * sh_b[b_base + 4u * BK_PAD + 2u] + a13 * sh_b[b_base + 4u * BK_PAD + 3u]);
            acc15 = acc15 + (a10 * sh_b[b_base + 5u * BK_PAD] + a11 * sh_b[b_base + 5u * BK_PAD + 1u] + a12 * sh_b[b_base + 5u * BK_PAD + 2u] + a13 * sh_b[b_base + 5u * BK_PAD + 3u]);
            acc16 = acc16 + (a10 * sh_b[b_base + 6u * BK_PAD] + a11 * sh_b[b_base + 6u * BK_PAD + 1u] + a12 * sh_b[b_base + 6u * BK_PAD + 2u] + a13 * sh_b[b_base + 6u * BK_PAD + 3u]);
            acc17 = acc17 + (a10 * sh_b[b_base + 7u * BK_PAD] + a11 * sh_b[b_base + 7u * BK_PAD + 1u] + a12 * sh_b[b_base + 7u * BK_PAD + 2u] + a13 * sh_b[b_base + 7u * BK_PAD + 3u]);
            acc20 = acc20 + (a20 * sh_b[b_base] + a21 * sh_b[b_base + 1u] + a22 * sh_b[b_base + 2u] + a23 * sh_b[b_base + 3u]);
            acc21 = acc21 + (a20 * sh_b[b_base + BK_PAD] + a21 * sh_b[b_base + BK_PAD + 1u] + a22 * sh_b[b_base + BK_PAD + 2u] + a23 * sh_b[b_base + BK_PAD + 3u]);
            acc22 = acc22 + (a20 * sh_b[b_base + 2u * BK_PAD] + a21 * sh_b[b_base + 2u * BK_PAD + 1u] + a22 * sh_b[b_base + 2u * BK_PAD + 2u] + a23 * sh_b[b_base + 2u * BK_PAD + 3u]);
            acc23 = acc23 + (a20 * sh_b[b_base + 3u * BK_PAD] + a21 * sh_b[b_base + 3u * BK_PAD + 1u] + a22 * sh_b[b_base + 3u * BK_PAD + 2u] + a23 * sh_b[b_base + 3u * BK_PAD + 3u]);
            acc24 = acc24 + (a20 * sh_b[b_base + 4u * BK_PAD] + a21 * sh_b[b_base + 4u * BK_PAD + 1u] + a22 * sh_b[b_base + 4u * BK_PAD + 2u] + a23 * sh_b[b_base + 4u * BK_PAD + 3u]);
            acc25 = acc25 + (a20 * sh_b[b_base + 5u * BK_PAD] + a21 * sh_b[b_base + 5u * BK_PAD + 1u] + a22 * sh_b[b_base + 5u * BK_PAD + 2u] + a23 * sh_b[b_base + 5u * BK_PAD + 3u]);
            acc26 = acc26 + (a20 * sh_b[b_base + 6u * BK_PAD] + a21 * sh_b[b_base + 6u * BK_PAD + 1u] + a22 * sh_b[b_base + 6u * BK_PAD + 2u] + a23 * sh_b[b_base + 6u * BK_PAD + 3u]);
            acc27 = acc27 + (a20 * sh_b[b_base + 7u * BK_PAD] + a21 * sh_b[b_base + 7u * BK_PAD + 1u] + a22 * sh_b[b_base + 7u * BK_PAD + 2u] + a23 * sh_b[b_base + 7u * BK_PAD + 3u]);
            acc30 = acc30 + (a30 * sh_b[b_base] + a31 * sh_b[b_base + 1u] + a32 * sh_b[b_base + 2u] + a33 * sh_b[b_base + 3u]);
            acc31 = acc31 + (a30 * sh_b[b_base + BK_PAD] + a31 * sh_b[b_base + BK_PAD + 1u] + a32 * sh_b[b_base + BK_PAD + 2u] + a33 * sh_b[b_base + BK_PAD + 3u]);
            acc32 = acc32 + (a30 * sh_b[b_base + 2u * BK_PAD] + a31 * sh_b[b_base + 2u * BK_PAD + 1u] + a32 * sh_b[b_base + 2u * BK_PAD + 2u] + a33 * sh_b[b_base + 2u * BK_PAD + 3u]);
            acc33 = acc33 + (a30 * sh_b[b_base + 3u * BK_PAD] + a31 * sh_b[b_base + 3u * BK_PAD + 1u] + a32 * sh_b[b_base + 3u * BK_PAD + 2u] + a33 * sh_b[b_base + 3u * BK_PAD + 3u]);
            acc34 = acc34 + (a30 * sh_b[b_base + 4u * BK_PAD] + a31 * sh_b[b_base + 4u * BK_PAD + 1u] + a32 * sh_b[b_base + 4u * BK_PAD + 2u] + a33 * sh_b[b_base + 4u * BK_PAD + 3u]);
            acc35 = acc35 + (a30 * sh_b[b_base + 5u * BK_PAD] + a31 * sh_b[b_base + 5u * BK_PAD + 1u] + a32 * sh_b[b_base + 5u * BK_PAD + 2u] + a33 * sh_b[b_base + 5u * BK_PAD + 3u]);
            acc36 = acc36 + (a30 * sh_b[b_base + 6u * BK_PAD] + a31 * sh_b[b_base + 6u * BK_PAD + 1u] + a32 * sh_b[b_base + 6u * BK_PAD + 2u] + a33 * sh_b[b_base + 6u * BK_PAD + 3u]);
            acc37 = acc37 + (a30 * sh_b[b_base + 7u * BK_PAD] + a31 * sh_b[b_base + 7u * BK_PAD + 1u] + a32 * sh_b[b_base + 7u * BK_PAD + 2u] + a33 * sh_b[b_base + 7u * BK_PAD + 3u]);
        }
        for (; kk < k_end; kk++) {
            let a_base = (local_m * TM) * BK_PAD + kk;
            let a0 = sh_a[a_base];
            let a1 = sh_a[a_base + BK_PAD];
            let a2 = sh_a[a_base + 2u * BK_PAD];
            let a3 = sh_a[a_base + 3u * BK_PAD];
            let b_base = (local_n * TN) * BK_PAD + kk;
            let b0 = sh_b[b_base];
            let b1 = sh_b[b_base + BK_PAD];
            let b2 = sh_b[b_base + 2u * BK_PAD];
            let b3 = sh_b[b_base + 3u * BK_PAD];
            let b4 = sh_b[b_base + 4u * BK_PAD];
            let b5 = sh_b[b_base + 5u * BK_PAD];
            let b6 = sh_b[b_base + 6u * BK_PAD];
            let b7 = sh_b[b_base + 7u * BK_PAD];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc02 = acc02 + a0 * b2;
            acc03 = acc03 + a0 * b3;
            acc04 = acc04 + a0 * b4;
            acc05 = acc05 + a0 * b5;
            acc06 = acc06 + a0 * b6;
            acc07 = acc07 + a0 * b7;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
            acc12 = acc12 + a1 * b2;
            acc13 = acc13 + a1 * b3;
            acc14 = acc14 + a1 * b4;
            acc15 = acc15 + a1 * b5;
            acc16 = acc16 + a1 * b6;
            acc17 = acc17 + a1 * b7;
            acc20 = acc20 + a2 * b0;
            acc21 = acc21 + a2 * b1;
            acc22 = acc22 + a2 * b2;
            acc23 = acc23 + a2 * b3;
            acc24 = acc24 + a2 * b4;
            acc25 = acc25 + a2 * b5;
            acc26 = acc26 + a2 * b6;
            acc27 = acc27 + a2 * b7;
            acc30 = acc30 + a3 * b0;
            acc31 = acc31 + a3 * b1;
            acc32 = acc32 + a3 * b2;
            acc33 = acc33 + a3 * b3;
            acc34 = acc34 + a3 * b4;
            acc35 = acc35 + a3 * b5;
            acc36 = acc36 + a3 * b6;
            acc37 = acc37 + a3 * b7;
        }

        workgroupBarrier();
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst_batch = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    wt_store(dst_batch, out_row, out_col, acc00, acc10, acc20, acc30);
    wt_store(dst_batch, out_row, out_col + 1u, acc01, acc11, acc21, acc31);
    wt_store(dst_batch, out_row, out_col + 2u, acc02, acc12, acc22, acc32);
    wt_store(dst_batch, out_row, out_col + 3u, acc03, acc13, acc23, acc33);
    wt_store(dst_batch, out_row, out_col + 4u, acc04, acc14, acc24, acc34);
    wt_store(dst_batch, out_row, out_col + 5u, acc05, acc15, acc25, acc35);
    wt_store(dst_batch, out_row, out_col + 6u, acc06, acc16, acc26, acc36);
    wt_store(dst_batch, out_row, out_col + 7u, acc07, acc17, acc27, acc37);
}
