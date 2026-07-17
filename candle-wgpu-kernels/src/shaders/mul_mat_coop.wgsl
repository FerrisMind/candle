// Dense F32 GEMM via wgpu cooperative matrix on hardware that supports
// 16x16 f16 A/B with f32 accumulator (NVIDIA Ampere+ Vulkan path).
//
// Query: Adapter::cooperative_matrix_properties() — RTX 3060 reports:
//   m=16 n=16 k=16 ab=F16 cr=F32 (and related 16x8 variants).
// There is NO f32 8x8 config on this GPU; using unsupported sizes yields zeros.
//
// Binding convention (same as warptile):
//   src0 = B^T or virtual B^T via stride_0k  (params.m = candle N)
//   src1 = A                                 (params.n = candle M)
//   dst[col * params.m + row]
//
// Math: C[m,n] = sum A[m,k]*B[k,n] = (B^T * A^T)[n,m]
// Stage panels in f16 (tensor-core inputs), accumulate in f32.

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

const TM: u32 = 16u; // along params.m (candle N)
const TN: u32 = 16u; // along params.n (candle M)
const TK: u32 = 16u;
const TILE: u32 = 256u; // 16*16

// f16 staging for A/B roles; f32 staging for C result.
var<workgroup> tile_bt: array<f16, TILE>; // B^T : N x K row-major
var<workgroup> tile_a: array<f16, TILE>;  // A   : M x K row-major
var<workgroup> tile_c: array<f32, TILE>;  // C   : N x M row-major

// 16×16 WG: one lane per tile element (matches official wgpu coopmat example).
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;

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

    tile_c[tid] = 0.0;
    workgroupBarrier();
    var acc = coopLoadT<coop_mat16x16<f32, C>>(&tile_c[0], 16u);

    for (var k0 = 0u; k0 < params.k; k0 += TK) {
        let r = tid / TK;
        let c = tid % TK;

        let gr_n = row_base + r;
        let gk = k0 + c;
        var v_bt: f16 = 0.0h;
        if (gr_n < params.m && gk < params.k) {
            v_bt = f16(src0[src0_batch + gr_n * params.stride_01 + gk * params.stride_0k]);
        }
        tile_bt[r * TK + c] = v_bt;

        let gr_m = col_base + r;
        var v_a: f16 = 0.0h;
        if (gr_m < params.n && gk < params.k) {
            v_a = f16(src1[src1_batch + gr_m * params.stride_11 + gk * params.stride_1k]);
        }
        tile_a[r * TK + c] = v_a;

        workgroupBarrier();

        let a_mat = coopLoadT<coop_mat16x16<f16, A>>(&tile_bt[0], 16u);
        let b_mat = coopLoad<coop_mat16x16<f16, B>>(&tile_a[0], 16u);
        acc = coopMultiplyAdd(a_mat, b_mat, acc);

        workgroupBarrier();
    }

    coopStoreT(acc, &tile_c[0], 16u);
    workgroupBarrier();

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst_batch = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    let n_local = tid / TN;
    let m_local = tid % TN;
    let global_row = row_base + n_local;
    let global_col = col_base + m_local;
    if (global_row < params.m && global_col < params.n) {
        dst[dst_batch + global_col * params.m + global_row] = tile_c[n_local * TN + m_local];
    }
}
