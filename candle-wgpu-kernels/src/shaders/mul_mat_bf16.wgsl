fn bf16_bits(v: f32) -> u32 {
    return ((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu;
}

fn atomic_store_bf16(dst_elem: u32, v: f32) {
    let wi = dst_elem / 2u;
    let half = dst_elem % 2u;
    let p = bf16_bits(v);
    let shift = half * 16u;
    let mask = select(0xffff0000u, 0x0000ffffu, half == 0u);
    loop {
        let old = atomicLoad(&dst[wi]);
        let desired = (old & mask) | (p << shift);
        let res = atomicCompareExchangeWeak(&dst[wi], old, desired);
        if res.exchanged {
            break;
        }
    }
}

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
};

@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> src1: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let global_idx = wg_linear * 256u + local_id.x;

    let total = params.m * params.n * params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (global_idx >= total) {
        return;
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = global_idx / dst3_stride;
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst3_rem = global_idx % dst3_stride;

    let dst2_idx = dst3_rem / dst2_stride;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;
    let dst2_rem = dst3_rem % dst2_stride;

    let row = dst2_rem / params.m;
    let col = dst2_rem % params.m;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11;

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        let a_idx = src0_idx_base + i;
        let b_idx = src1_idx_base + i;
        let a_word = src0[a_idx / 2u];
        let b_word = src1[b_idx / 2u];
        let a = bitcast<f32>(((a_word >> (16u * (a_idx % 2u))) & 0xffffu) << 16u);
        let b = bitcast<f32>(((b_word >> (16u * (b_idx % 2u))) & 0xffffu) << 16u);
        sum += a * b;
    }

    let dst_elem = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.m + col;
    atomic_store_bf16(dst_elem, sum);
}
