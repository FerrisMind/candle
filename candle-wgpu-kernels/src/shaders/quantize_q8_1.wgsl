enable f16;

struct BlockQ8_1 {
    d: f16,
    s: f16,
    qs: array<u32, 8>,
};

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<BlockQ8_1>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> shmem_max: array<f32, 64>;
var<workgroup> shmem_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let tid = lid.x;
    let block_in_wg = tid / 8u;
    let iqs = tid % 8u;
    let ib = wgid.x * 8u + block_in_wg;

    if (ib >= params.num_blocks) {
        return;
    }

    let a_idx = ib * 32u + iqs * 4u;
    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    var v2: f32 = 0.0;
    var v3: f32 = 0.0;

    if (a_idx < params.ne) { v0 = src[a_idx]; }
    if (a_idx + 1u < params.ne) { v1 = src[a_idx + 1u]; }
    if (a_idx + 2u < params.ne) { v2 = src[a_idx + 2u]; }
    if (a_idx + 3u < params.ne) { v3 = src[a_idx + 3u]; }

    let tmax = max(max(abs(v0), abs(v1)), max(abs(v2), abs(v3)));
    shmem_max[tid] = tmax;

    workgroupBarrier();

    let base_tid = block_in_wg * 8u;
    var amax: f32 = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        amax = max(amax, shmem_max[base_tid + i]);
    }

    let d: f32 = amax / 127.0;
    let id: f32 = select(1.0 / d, 0.0, d == 0.0);

    let q0 = i32(round(v0 * id));
    let q1 = i32(round(v1 * id));
    let q2 = i32(round(v2 * id));
    let q3 = i32(round(v3 * id));

    let u0 = u32(q0 & 0xFF);
    let u1 = u32(q1 & 0xFF);
    let u2 = u32(q2 & 0xFF);
    let u3 = u32(q3 & 0xFF);
    let packed_qs = u0 | (u1 << 8u) | (u2 << 16u) | (u3 << 24u);

    dst[ib].qs[iqs] = packed_qs;

    let tsum = f32(q0) + f32(q1) + f32(q2) + f32(q3);
    shmem_sum[tid] = tsum;

    workgroupBarrier();

    if (iqs == 0u) {
        var sum_all: f32 = 0.0;
        for (var i = 0u; i < 8u; i = i + 1u) {
            sum_all = sum_all + shmem_sum[base_tid + i];
        }
        dst[ib].d = f16(d);
        dst[ib].s = f16(sum_all);
    }
}
