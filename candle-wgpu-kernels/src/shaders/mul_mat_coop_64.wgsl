// Dense F32 GEMM via wgpu cooperative matrix (Ampere+ Vulkan).
// Hardware: 16x16 f16 A/B, f32 C (RTX 3060 cooperative_matrix_properties).
//
// Workgroup = 512 threads = 16 warps, 64x64 output block:
//   ti = sg % 4, tj = sg / 4
// Used for tall/wide residual GEMMs; large squares use mul_mat_coop.wgsl (128×64).
//
// Binding (warptile convention):
//   src0 = B^T / virtual B^T (params.m = candle N)
//   src1 = A                 (params.n = candle M)
//   dst[col * params.m + row]

enable f16;
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

const TM: u32 = 16u;
const TN: u32 = 16u;
const TK: u32 = 16u;
const WG_M: u32 = 64u;
const WG_N: u32 = 64u;
const SG_M: u32 = 4u;

var<workgroup> tile_bt: array<f16, 1024>; // 64*16
var<workgroup> tile_a: array<f16, 1024>;  // 64*16
var<workgroup> tile_c: array<f32, 4096>;  // 16 * 256

@compute @workgroup_size(512)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = lid.x;
    let sg_id = tid / 32u;
    let lane = tid % 32u;
    let ti = sg_id % SG_M;
    let tj = sg_id / SG_M;

    let tiles_m = (params.m + WG_M - 1u) / WG_M;
    let tiles_n = (params.n + WG_N - 1u) / WG_N;
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

    let row_base = tile_m * WG_M;
    let col_base = tile_n * WG_N;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    let c_base = sg_id * 256u;
    for (var i = 0u; i < 8u; i++) {
        tile_c[c_base + lane * 8u + i] = 0.0;
    }
    workgroupBarrier();
    var acc = coopLoadT<coop_mat16x16<f32, C>>(&tile_c[c_base], 16u);

    // Note: do not branch around coop ops on non-provably-uniform predicates
    // (naga UniformityRequirements::COOP_OPS). Bounds checks on loads are OK.
    for (var k0 = 0u; k0 < params.k; k0 += TK) {
        for (var i = 0u; i < 2u; i++) {
            let elem = tid * 2u + i;
            let r = elem / TK;
            let c = elem % TK;
            let gr_n = row_base + r;
            let gk = k0 + c;
            var v_bt: f16 = 0.0h;
            if (gr_n < params.m && gk < params.k) {
                v_bt = f16(src0[src0_batch + gr_n * params.stride_01 + gk * params.stride_0k]);
            }
            tile_bt[elem] = v_bt;

            let gr_m = col_base + r;
            var v_a: f16 = 0.0h;
            if (gr_m < params.n && gk < params.k) {
                v_a = f16(src1[src1_batch + gr_m * params.stride_11 + gk * params.stride_1k]);
            }
            tile_a[elem] = v_a;
        }
        workgroupBarrier();

        let a_mat = coopLoadT<coop_mat16x16<f16, A>>(&tile_bt[(ti * TM) * TK], 16u);
        let b_mat = coopLoad<coop_mat16x16<f16, B>>(&tile_a[(tj * TN) * TK], 16u);
        acc = coopMultiplyAdd(a_mat, b_mat, acc);

        workgroupBarrier();
    }

    coopStoreT(acc, &tile_c[c_base], 16u);
    workgroupBarrier();

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst_batch = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var i = 0u; i < 8u; i++) {
        let elem = lane * 8u + i;
        let n_local = elem / TN;
        let m_local = elem % TN;
        let global_row = row_base + ti * TM + n_local;
        let global_col = col_base + tj * TN + m_local;
        if (global_row < params.m && global_col < params.n) {
            dst[dst_batch + global_col * params.m + global_row] = tile_c[c_base + elem];
        }
    }
}
