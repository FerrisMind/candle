enable f16;

// Reproduce the CPU QMatMul A-side (LHS/activation) quantization contract for
// the k-quants (Q2_K..Q6_K), whose `GgmlType::VecDotType` is `BlockQ8K`:
// `from_float` rounds the activation per **256**-element block with the signed
// extreme (`max` = the element of largest absolute value, sign preserved),
// `iscale = -127/max`, `q = round(x*iscale)` clamped to [-128, 127], and the
// dequantized value is `d*q` with `d = 1/iscale` in **f32**.
//
// This is NOT the Q8_1 grid (per-32, f16 scale) the backend used to feed every
// quantized dtype: the grids differ by ~1e-5 nmse per layer against the CPU
// reference, which compounds to ~1e-2 in whole-model logits — Qwen3-0.6B
// Q4_K_M missed the model-matrix gate on both backends with identical failure
// at the base commit. Q4_0/Q5_0/Q8_0 keep the Q8_1 shader because their CPU
// `VecDotType` is `BlockQ8_0`, whose rounding grid is identical.
//
// The rounded f32 values feed the quantized matmul/matvec kernels, which
// multiply them by the dequantized weights — the same arithmetic the CPU
// `vec_dot_q*k_q8k` performs on the Q8K integers.

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
    src_is_f16: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> wg_amax: array<f32, 64>;
var<workgroup> wg_maxv: array<f32, 64>;

fn load_elem(idx: u32) -> f32 {
    if (params.src_is_f16 == 0u) {
        return bitcast<f32>(src[idx]);
    }
    let word = src[idx / 2u];
    let half = idx % 2u;
    return f32(unpack2x16float(word)[half]);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // One workgroup per 256-element block; each thread covers 4 elements.
    let ib = wid.x;
    let t = lid.x;
    let base = ib * 256u + t * 4u;

    var amax: f32 = 0.0;
    var maxv: f32 = 0.0;
    if (ib < params.num_blocks) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let idx = base + j;
            if (idx < params.ne) {
                let v = load_elem(idx);
                if (abs(v) > amax) {
                    amax = abs(v);
                    maxv = v;
                }
            }
        }
    }
    wg_amax[t] = amax;
    wg_maxv[t] = maxv;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (t < s) {
            if (wg_amax[t + s] > wg_amax[t]) {
                wg_amax[t] = wg_amax[t + s];
                wg_maxv[t] = wg_maxv[t + s];
            }
        }
        workgroupBarrier();
    }

    let mx = wg_maxv[0];
    // `BlockQ8K::from_float` maps an all-zero block to `d = 0`, `qs = 0`.
    let iscale = select(-127.0 / mx, 0.0, mx == 0.0);
    let d = select(1.0 / iscale, 0.0, mx == 0.0);

    if (ib < params.num_blocks) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let idx = base + j;
            if (idx < params.ne) {
                let q = clamp(round(load_elem(idx) * iscale), -128.0, 127.0);
                dst[idx] = d * q;
            }
        }
    }
}
