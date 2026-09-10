// Fused LayerNorm(+affine) -> RoPE for 2D-pair rope on head rows.
//
// One workgroup (64 lanes) per row of the (b, heads, n, head_dim) view.
// The input may be a strided view of a qkv projection buffer (only the last
// dim must be contiguous, stride 1), so the separate .contiguous() copy and
// the 8-op rope chain (cat + 2 broadcast_mul + add) collapse into one
// dispatch.
//
//   norm:  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta   (biased var)
//   rope:  out[j] = y[j] * cos[j] + y[j ^ quarter] * sin_alt[j]
//
// sin_alt carries the rotate-half negation baked in (see lux3d RopeEmbeddings),
// and the partner index is j ^ quarter because quarter is a power of two, so
// the XOR flips exactly the within-half pairing.
struct Params {
    offset_src: u32,
    stride_s0: u32,
    stride_s1: u32,
    stride_s2: u32,
    ne0: u32, // head_dim (== 64, host-gated)
    ne1: u32, // heads
    ne2: u32, // n
    ne3: u32, // batch
    quarter: u32,
    apply_norm: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> src: array<f32>;
@group(0) @binding(1)
var<storage, read> gb: array<f32>; // [gamma(ne0), beta(ne0)]
@group(0) @binding(2)
var<storage, read> cos_t: array<f32>;
@group(0) @binding(3)
var<storage, read> sin_t: array<f32>;
@group(0) @binding(4)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(5)
var<uniform> params: Params;

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let row = wid.x + wid.y * num_wg.x;
    let rows = params.ne3 * params.ne1 * params.ne2;
    if (row >= rows) {
        return; // uniform across the workgroup: no barrier divergence
    }
    let heads_n = params.ne1 * params.ne2;
    let i3 = row / heads_n;
    let rem = row % heads_n;
    let i1 = rem / params.ne2;
    let i2 = rem % params.ne2;

    let base = params.offset_src + i3 * params.stride_s0 + i1 * params.stride_s1 + i2 * params.stride_s2;
    var x = src[base + lane];

    if (params.apply_norm == 1u) {
        red[lane] = x;
        workgroupBarrier();
        for (var off: u32 = 32u; off >= 1u; off = off >> 1u) {
            if (lane < off) {
                red[lane] = red[lane] + red[lane + off];
            }
            workgroupBarrier();
        }
        let mean = red[0] / 64.0;
        workgroupBarrier();
        let xc = x - mean;
        red[lane] = xc * xc;
        workgroupBarrier();
        for (var off: u32 = 32u; off >= 1u; off = off >> 1u) {
            if (lane < off) {
                red[lane] = red[lane] + red[lane + off];
            }
            workgroupBarrier();
        }
        let variance = red[0] / 64.0;
        workgroupBarrier();
        let rstd = 1.0 / sqrt(variance + params.eps);
        x = xc * rstd * gb[lane] + gb[params.ne0 + lane];
    }

    red[lane] = x;
    workgroupBarrier();
    let partner = red[lane ^ params.quarter];
    let trow = (i3 * params.ne2 + i2) * params.ne0 + lane;
    let out = x * cos_t[trow] + partner * sin_t[trow];
    dst[row * params.ne0 + lane] = out;
}
