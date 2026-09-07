// Native 2D conv_transpose2d (transposed convolution) for F32 / F16 / BF16,
// one thread per output element. Direct gather:
//
//   out[b, co, oh, ow] = sum_{ci, kh, kw} in[b, ci, ih, iw] * w[ci, co, kh, kw]
//
// where ih = (oh + padding - kh * dilation) / stride when it divides evenly
// and stays inside the input (otherwise the tap contributes 0). This replaces
// the im2col-scatter chain (matmul + mask-mul + index_add + zeros) whose
// host-built ids/mask vectors and their per-call uploads dominated runtime.
//
// All accumulations are f32 (matches the CUDA parity path); stores round back
// to the native dtype. Thread-decomposition is NCHW with w innermost, so a
// warp iterates the same (kh, kw, ci) tap and reads consecutive iw — coalesced.
//
// Preprocessor contract (see candle-wgpu-kernels/src/lib.rs):
//   * WG_SIZE   -> workgroup size.
//   * SRC_TYPE  -> f32 / f16 / u32 (BF16 tensors are packed two per u32 word).
//   * DST_TYPE  -> f32 / f16 / u32.
//   * SRC_BF16 / DST_BF16 -> decode/encode BF16 halves (CAS stores: adjacent
//     output elements share a u32 word and are written by distinct threads).
//   * dtype F16 keeps the f16 extension directive; F32/BF16 strip it.

enable f16;

struct Params {
    ne: u32,         // total output elements (b * c_out * out_h * out_w)
    offset_src: u32, // in elements
    offset_w: u32,   // in elements
    offset_dst: u32, // in elements
    st_in_b: u32,    // input strides, innermost-first from dims4: [w, h, c, b]
    st_in_c: u32,
    st_in_h: u32,
    st_in_w: u32,
    st_k_ci: u32, // kernel strides, innermost-first: [kw, kh, c_out, c_in]
    st_k_co: u32,
    st_k_h: u32,
    st_k_w: u32,
    out_h: u32,
    out_w: u32,
    c_out: u32,
    c_in: u32,
    i_h: u32,
    i_w: u32,
    k_h: u32,
    k_w: u32,
    stride: u32,
    dilation: u32,
    padding: u32,
    _pad0: u32,
};

#ifdef SRC_BF16
@group(0) @binding(0)
var<storage, read_write> src: array<u32>;

fn bf16_to_f32(word: u32, half: u32) -> f32 {
    let shift = half * 16u;
    return bitcast<f32>(((word >> shift) & 0xffffu) << 16u);
}

fn load_src(elem: u32) -> f32 {
    return bf16_to_f32(src[elem / 2u], elem % 2u);
}
#else
@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

fn load_src(elem: u32) -> f32 {
    return f32(src[elem]);
}
#endif

#ifdef SRC_BF16
@group(0) @binding(1)
var<storage, read_write> w: array<u32>;

fn load_w(elem: u32) -> f32 {
    return bf16_to_f32(w[elem / 2u], elem % 2u);
}
#else
@group(0) @binding(1)
var<storage, read_write> w: array<SRC_TYPE>;

fn load_w(elem: u32) -> f32 {
    return f32(w[elem]);
}
#endif

#ifdef DST_BF16
@group(0) @binding(2)
var<storage, read_write> dst: array<atomic<u32>>;

fn bf16_bits(v: f32) -> u32 {
    return ((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu;
}

fn store_dst(elem: u32, v: f32) {
    let wi = elem / 2u;
    let half = elem % 2u;
    let p = bf16_bits(v);
    let shift = half * 16u;
    // Keep the sibling half, replace this half (same ordering as bf16_store_half).
    let mask = select(0x0000ffffu, 0xffff0000u, half == 0u);
    // Output element pairs share a u32 word; CAS so a concurrent writer of the
    // sibling half cannot clobber this one.
    loop {
        let old = atomicLoad(&dst[wi]);
        let desired = (old & mask) | (p << shift);
        let res = atomicCompareExchangeWeak(&dst[wi], old, desired);
        if res.exchanged {
            break;
        }
    }
}
#else
@group(0) @binding(2)
var<storage, read_write> dst: array<DST_TYPE>;

fn store_dst(elem: u32, v: f32) {
    dst[elem] = DST_TYPE(v);
}
#endif

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lin = gid.x;
    if (lin >= params.ne) {
        return;
    }
    let ow = lin % params.out_w;
    let t0 = lin / params.out_w;
    let oh = t0 % params.out_h;
    let bc = t0 / params.out_h;
    let co = bc % params.c_out;
    let b = bc / params.c_out;

    let in_bc_base = params.offset_src + b * params.st_in_b;
    let w_co_base = params.offset_w + co * params.st_k_co;
    let dst_elem = params.offset_dst + lin;

    var acc: f32 = 0.0;
    let sh = i32(params.stride);
    for (var kh = 0u; kh < params.k_h; kh += 1u) {
        let v_h = i32(oh) + i32(params.padding) - i32(kh * params.dilation);
        if (v_h < 0 || v_h % sh != 0) {
            continue;
        }
        let ih = v_h / sh;
        if (ih >= i32(params.i_h)) {
            continue;
        }
        let in_h_base = in_bc_base + u32(ih) * params.st_in_h;
        let w_h_base = w_co_base + kh * params.st_k_h;
        for (var kw = 0u; kw < params.k_w; kw += 1u) {
            let v_w = i32(ow) + i32(params.padding) - i32(kw * params.dilation);
            if (v_w < 0 || v_w % sh != 0) {
                continue;
            }
            let iw = v_w / sh;
            if (iw >= i32(params.i_w)) {
                continue;
            }
            let in_base = in_h_base + u32(iw) * params.st_in_w;
            let w_base = w_h_base + kw * params.st_k_w;
            for (var ci = 0u; ci < params.c_in; ci += 1u) {
                acc += load_src(in_base + ci * params.st_in_c)
                     * load_w(w_base + ci * params.st_k_ci);
            }
        }
    }
    store_dst(dst_elem, acc);
}
