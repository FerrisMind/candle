// Native 2D pool (max / avg) for F32 / F16 / BF16, one thread per output
// element. Unlike the legacy im2col + reduce chain, this keeps the tensor in
// its source dtype (no F32 hub round-trip) and handles arbitrary batch/channel
// strides directly through the stride fields, so non-contiguous inputs do not
// need to be materialized.
//
// Preprocessor contract (see candle-wgpu-kernels/src/lib.rs):
//   * WG_SIZE      -> workgroup size (replaced as a whole identifier).
//   * SRC_TYPE     -> f32 / f16 / u32 (BF16 tensors are stored as u32 words,
//                     two halves per word).
//   * DST_TYPE     -> f32 / f16 / u32.
//   * POOL_MAX     -> max pool (compare on the source dtype semantics; for
//                     BF16 the f32 decode is order-preserving so comparing the
//                     f32 values is equivalent).
//   * POOL_AVG     -> avg pool (accumulate in f32, divide by kh*kw, round back
//                     to the source dtype — CUDA parity).
//   * SRC_BF16     -> decode BF16 halves to f32 on load.
//   * DST_BF16     -> encode f32 to BF16 halves with a CAS so concurrent
//                     writers of the two halves of one u32 word do not race.
//   * dtype F16 keeps the f16 extension directive injected by lib.rs; dtype
//     F32 strips it (f32/bf16 variants carry no f16 types).

enable f16;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    stride0: u32,
    stride1: u32,
    stride2: u32,
    stride3: u32,
    ne: u32,   // total number of output elements (b*c*out_h*out_w)
    bc: u32,   // b*c
    c: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    _pad0: u32,
};

#ifdef SRC_BF16
@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

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

#ifdef DST_BF16
@group(0) @binding(1)
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
@group(0) @binding(1)
var<storage, read_write> dst: array<DST_TYPE>;

fn store_dst(elem: u32, v: f32) {
    dst[elem] = DST_TYPE(v);
}
#endif

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }
    let lin = gid.x;
    let ow = lin % params.out_w;
    let t0 = lin / params.out_w;
    let oh = t0 % params.out_h;
    let bc_idx = t0 / params.out_h;
    let b_idx = bc_idx / params.c;
    let c_idx = bc_idx % params.c;
    // Top-left of the kernel window for this output element.
    let base = params.offset_src + b_idx * params.stride0 + c_idx * params.stride1 +
               oh * params.sh * params.stride2 + ow * params.sw * params.stride3;
    let dst_elem = params.offset_dst + lin;

#ifdef POOL_MAX
    var acc = load_src(base);
    for (var m = 0u; m < params.kh; m += 1u) {
        for (var n = 0u; n < params.kw; n += 1u) {
            let v = load_src(base + m * params.stride2 + n * params.stride3);
            if (v > acc) {
                acc = v;
            }
        }
    }
    store_dst(dst_elem, acc);
#else
    var acc: f32 = 0.0;
    for (var m = 0u; m < params.kh; m += 1u) {
        for (var n = 0u; n < params.kw; n += 1u) {
            acc += load_src(base + m * params.stride2 + n * params.stride3);
        }
    }
    acc = acc / f32(params.kh * params.kw);
    store_dst(dst_elem, acc);
#endif
}
