struct Params {
    offset_src: u32,
    offset_alpha: u32,
    offset_beta: u32,
    offset_dst: u32,

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_alpha1: u32,
    stride_alpha2: u32,
    stride_alpha3: u32,

    stride_beta1: u32,
    stride_beta2: u32,
    stride_beta3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    alpha_ne0: u32,
    alpha_ne1: u32,
    alpha_ne2: u32,
    alpha_ne3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> alpha: array<f32>;

@group(0) @binding(2)
var<storage, read_write> beta: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

var<workgroup> w_sum: array<f32, WG_SIZE>;
var<workgroup> w_sum2: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x + wid.y * num_wg.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    let i_alpha_row = params.offset_alpha + (i3 % params.alpha_ne3) * params.stride_alpha3 + (i2 % params.alpha_ne2) * params.stride_alpha2 + (i1 % params.alpha_ne1) * params.stride_alpha1;
    let i_beta_row = params.offset_beta + (i3 % params.alpha_ne3) * params.stride_beta3 + (i2 % params.alpha_ne2) * params.stride_beta2 + (i1 % params.alpha_ne1) * params.stride_beta1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    var partial_sum = 0.0f;
    var partial_sum2 = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0u; j < elems; j++) {
        if (col >= params.ne0) { break; }
        let v = src[i_src_row + col];
        partial_sum += v;
        partial_sum2 += v * v;
        col += WG_SIZE;
    }

    w_sum[lid.x] = partial_sum;
    w_sum2[lid.x] = partial_sum2;
    workgroupBarrier();

    var offset: u32 = WG_SIZE / 2u;
    while (offset > 0u) {
        if (lid.x < offset) {
            w_sum[lid.x] += w_sum[lid.x + offset];
            w_sum2[lid.x] += w_sum2[lid.x + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }

    let total = f32(params.ne0);
    let mean = w_sum[0] / total;
    let var_ = w_sum2[0] / total - mean * mean;
    let inv_std = inverseSqrt(var_ + params.eps);

    col = lid.x;
    for (var j: u32 = 0u; j < elems; j++) {
        if (col >= params.ne0) { break; }
        let a = alpha[i_alpha_row + col % params.alpha_ne0];
        let b = beta[i_beta_row + col % params.alpha_ne0];
        let v = src[i_src_row + col];
        dst[i_dst_row + col] = ((v - mean) * inv_std) * a + b;
        col += WG_SIZE;
    }
}
