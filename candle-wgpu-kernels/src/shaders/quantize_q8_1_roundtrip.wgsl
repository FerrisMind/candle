enable f16;

// Reproduce the CPU QMatMul A-side (LHS/activation) quantization contract for
// Q8_1 weights: the activation f32/f16 is quantized to q8_1 (per 32-element
// block, scale = amax/127, stored as f16) and then immediately dequantized back
// to f32. The result is the f32 value the CPU `vec_dot_q8_1_q8_1` would use for
// the LHS term, so feeding it to the fused q8_1 kernel (which multiplies it by
// the q8_1-dequantized weights) reproduces `sum(qs_lhs[i]*qs_rhs[i])*d_lhs*d_rhs`.
//
// Mirrors `BlockQ8_1::from_float` + `to_float` (f16 `d`), except the block-sum
// `s` is unused by the dot and therefore skipped. `d` is pushed through
// workgroup memory to force a real f32->f16->f32 round-trip: a direct
// `f32(f16(d))` expression is folded back to the full f32 `d` by the compiler,
// which would drop the f16 scale rounding that the CPU contract requires.

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
    src_is_f16: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> wg_d: array<f16, 64>;

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
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ib = gid.x;
    let base = ib * 32u;

    var amax: f32 = 0.0;
    if (ib < params.num_blocks) {
        for (var j = 0u; j < 32u; j = j + 1u) {
            let idx = base + j;
            if (idx < params.ne) {
                amax = max(amax, abs(load_elem(idx)));
            }
        }
    }

    let d = amax / 127.0;
    let id = select(1.0 / d, 0.0, d == 0.0);
    // Force a genuine f16 round-trip so the dequantized LHS uses the same
    // f16-rounded scale the quantized q8_1 weights store.
    wg_d[lid.x] = f16(d);
    workgroupBarrier();
    let df16 = f32(wg_d[lid.x]);

    if (ib < params.num_blocks) {
        for (var j = 0u; j < 32u; j = j + 1u) {
            let idx = base + j;
            if (idx < params.ne) {
                let q = i32(round(load_elem(idx) * id));
                dst[idx] = f32(q) * df16;
            }
        }
    }
}
