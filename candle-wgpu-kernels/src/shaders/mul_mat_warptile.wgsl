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

    var acc: array<array<f32, TN>, TM>;
    for (var tm = 0u; tm < TM; tm++) {
        for (var tn = 0u; tn < TN; tn++) {
            acc[tm][tn] = 0.0;
        }
    }

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
            var a0: array<f32, TM>;
            var a1: array<f32, TM>;
            var a2: array<f32, TM>;
            var a3: array<f32, TM>;
            for (var tm = 0u; tm < TM; tm++) {
                let base = (local_m * TM + tm) * BK_PAD + kk;
                a0[tm] = sh_a[base];
                a1[tm] = sh_a[base + 1u];
                a2[tm] = sh_a[base + 2u];
                a3[tm] = sh_a[base + 3u];
            }
            for (var tn = 0u; tn < TN; tn++) {
                let base = (local_n * TN + tn) * BK_PAD + kk;
                let b0 = sh_b[base];
                let b1 = sh_b[base + 1u];
                let b2 = sh_b[base + 2u];
                let b3 = sh_b[base + 3u];
                for (var tm = 0u; tm < TM; tm++) {
                    acc[tm][tn] += a0[tm] * b0 + a1[tm] * b1 + a2[tm] * b2 + a3[tm] * b3;
                }
            }
        }
        for (; kk < k_end; kk++) {
            var a_reg: array<f32, TM>;
            for (var tm = 0u; tm < TM; tm++) {
                a_reg[tm] = sh_a[(local_m * TM + tm) * BK_PAD + kk];
            }
            for (var tn = 0u; tn < TN; tn++) {
                let bv = sh_b[(local_n * TN + tn) * BK_PAD + kk];
                for (var tm = 0u; tm < TM; tm++) {
                    acc[tm][tn] += a_reg[tm] * bv;
                }
            }
        }

        workgroupBarrier();
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst_batch = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < TN; tn++) {
        let global_col = col_base + local_n * TN + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < TM; tm++) {
                let global_row = row_base + local_m * TM + tm;
                if (global_row < params.m) {
                    dst[dst_batch + global_col * params.m + global_row] = acc[tm][tn];
                }
            }
        }
    }
}
