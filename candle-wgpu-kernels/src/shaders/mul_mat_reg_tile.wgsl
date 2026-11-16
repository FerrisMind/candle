enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#include "common_decls.tmpl"

#include "mul_mat_decls.tmpl"

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
    // Inner K strides (1 for contiguous-K storage). Non-unit stride_0k lets
    // src0 be a contiguous (K, N) matrix viewed as (N, K) without materialize.
    stride_0k: u32,
    stride_1k: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
const TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

// SHMEM_ELEM is f32 for FLOAT_ACC_SHMEM (dense F32 accuracy) and f16 otherwise.
#ifdef FLOAT_ACC_SHMEM
var<workgroup> shmem: array<f32, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;
#else
var<workgroup> shmem: array<f16, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;
#endif

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    let wg_n_count = (params.n + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_per_matrix = wg_m_count * wg_n_count;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    let batch_idx = wg_linear / wg_per_matrix;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
    let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    // Unrolled 4x4 register tile. The old `acc: array<array<f32, ..>>` was
    // indexed by runtime loop counters, so Naga emitted it as an OpVariable
    // Function array and every inner-loop FMA became an AccessChain + Load +
    // AccessChain + Store to local memory (measured ~0.4 TFLOP/s, 3% of an
    // RTX 3060's f32 peak). Sixteen named scalars keep the accumulators in
    // registers. TILE_M/TILE_N are fixed at 4 for every user of this shader
    // (dense and quantized); matmul_fast_shader asserts it.
    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc02 = 0.0;
    var acc03 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc20 = 0.0;
    var acc21 = 0.0;
    var acc22 = 0.0;
    var acc23 = 0.0;
    var acc30 = 0.0;
    var acc31 = 0.0;
    var acc32 = 0.0;
    var acc33 = 0.0;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // see mul_mat_decls.tmpl
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);
        let a_base = local_m * TILE_M * TILE_K;
        let b_base = TILE_SRC0_SHMEM + local_n * TILE_N * TILE_K;

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            let a0 = f32(shmem[a_base + k_inner]);
            let a1 = f32(shmem[a_base + TILE_K + k_inner]);
            let a2 = f32(shmem[a_base + 2u * TILE_K + k_inner]);
            let a3 = f32(shmem[a_base + 3u * TILE_K + k_inner]);
            let b0 = f32(shmem[b_base + k_inner]);
            let b1 = f32(shmem[b_base + TILE_K + k_inner]);
            let b2 = f32(shmem[b_base + 2u * TILE_K + k_inner]);
            let b3 = f32(shmem[b_base + 3u * TILE_K + k_inner]);
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc02 = acc02 + a0 * b2;
            acc03 = acc03 + a0 * b3;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
            acc12 = acc12 + a1 * b2;
            acc13 = acc13 + a1 * b3;
            acc20 = acc20 + a2 * b0;
            acc21 = acc21 + a2 * b1;
            acc22 = acc22 + a2 * b2;
            acc23 = acc23 + a2 * b3;
            acc30 = acc30 + a3 * b0;
            acc31 = acc31 + a3 * b1;
            acc32 = acc32 + a3 * b2;
            acc33 = acc33 + a3 * b3;
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;
    let row0 = output_row_base;
    let row1 = output_row_base + 1u;
    let row2 = output_row_base + 2u;
    let row3 = output_row_base + 3u;
    let col0 = output_col_base;
    let col1 = output_col_base + 1u;
    let col2 = output_col_base + 2u;
    let col3 = output_col_base + 3u;

#ifdef VEC
    // VEC store is only selected for TILE_M-aligned m and row bases, so a
    // single row0 check covers the whole vec4.
    if (col0 < params.n && row0 < params.m) {
        let b = (dst_batch_offset + col0 * params.m + row0) / VEC_SIZE;
        dst[b] = vec4<f32>(acc00, acc10, acc20, acc30);
    }
    if (col1 < params.n && row0 < params.m) {
        let b = (dst_batch_offset + col1 * params.m + row0) / VEC_SIZE;
        dst[b] = vec4<f32>(acc01, acc11, acc21, acc31);
    }
    if (col2 < params.n && row0 < params.m) {
        let b = (dst_batch_offset + col2 * params.m + row0) / VEC_SIZE;
        dst[b] = vec4<f32>(acc02, acc12, acc22, acc32);
    }
    if (col3 < params.n && row0 < params.m) {
        let b = (dst_batch_offset + col3 * params.m + row0) / VEC_SIZE;
        dst[b] = vec4<f32>(acc03, acc13, acc23, acc33);
    }
#else
    if (col0 < params.n) {
        let b = dst_batch_offset + col0 * params.m + row0;
        if (row0 < params.m) { dst[b] = acc00; }
        if (row1 < params.m) { dst[b + 1u] = acc10; }
        if (row2 < params.m) { dst[b + 2u] = acc20; }
        if (row3 < params.m) { dst[b + 3u] = acc30; }
    }
    if (col1 < params.n) {
        let b = dst_batch_offset + col1 * params.m + row0;
        if (row0 < params.m) { dst[b] = acc01; }
        if (row1 < params.m) { dst[b + 1u] = acc11; }
        if (row2 < params.m) { dst[b + 2u] = acc21; }
        if (row3 < params.m) { dst[b + 3u] = acc31; }
    }
    if (col2 < params.n) {
        let b = dst_batch_offset + col2 * params.m + row0;
        if (row0 < params.m) { dst[b] = acc02; }
        if (row1 < params.m) { dst[b + 1u] = acc12; }
        if (row2 < params.m) { dst[b + 2u] = acc22; }
        if (row3 < params.m) { dst[b + 3u] = acc32; }
    }
    if (col3 < params.n) {
        let b = dst_batch_offset + col3 * params.m + row0;
        if (row0 < params.m) { dst[b] = acc03; }
        if (row1 < params.m) { dst[b + 1u] = acc13; }
        if (row2 < params.m) { dst[b + 2u] = acc23; }
        if (row3 < params.m) { dst[b + 3u] = acc33; }
    }
#endif
}
