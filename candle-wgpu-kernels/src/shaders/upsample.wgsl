// Native-dtype gather upsample (nearest1d / nearest2d / bilinear2d).
//
// The source is read in its stored dtype (no F32 hub round-trip), bilinear
// interpolation is accumulated in f32, and the result is rounded to the
// source dtype exactly once on store.
//
// Mode is selected by #ifdef batch token:
//   NEAREST1D  — input treated as [b*c, len]; single axis table `tab_a`
//   NEAREST2D  — input [b*c, h, w]; per-axis source-index tables tab_a/tab_b
//   BILINEAR2D — input [b*c, h, w]; per-axis (idx0, idx1, weight) tables
//
// Dtype is selected by #ifdef token, the host also replaces the SRC_TYPE
// token used by the storage declarations:
//   SRC_F32  — SRC_TYPE=f32 (native f32 storage)
//   SRC_F16  — SRC_TYPE=f16 (enable f16; native half storage)
//   SRC_BF16 — SRC_TYPE=u32 (two bf16 halves per u32 word; packed-half
//              atomic-CAS destination writes)

#ifdef SRC_F16
enable f16;
#endif

// Source storage: plain element type for every variant (F32/F16 natively, u32
// words for BF16).
@group(0) @binding(0) var<storage, read_write> src: array<SRC_TYPE>;

// Destination storage: BF16 writes go through per-half atomic CAS on packed u32
// words (`array<atomic<u32>>`); F32/F16 write the plain element type directly.
#ifndef SRC_BF16
@group(0) @binding(1) var<storage, read_write> dst: array<SRC_TYPE>;
#endif
#ifdef SRC_BF16
@group(0) @binding(1) var<storage, read_write> dst: array<atomic<u32>>;
#endif

#ifdef SRC_BF16
fn bf16_bits(v: f32) -> u32 {
    return ((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu;
}
fn bf16_to_f32(word: u32, half: u32) -> f32 {
    let shift = half * 16u;
    return bitcast<f32>(((word >> shift) & 0xffffu) << 16u);
}
fn bf16_store_at(elem: u32, v: f32) {
    // Packed-half CAS: concurrent invocations writing the same u32 word never
    // lose a half (mirrors bf16_store_half of BF16_WGSL_HELPERS).
    let wi = elem / 2u;
    let half = elem % 2u;
    let p = bf16_bits(v);
    let shift = half * 16u;
    let mask = select(0x0000ffffu, 0xffff0000u, half == 0u);
    loop {
        let old = atomicLoad(&dst[wi]);
        let desired = (old & mask) | (p << shift);
        let res = atomicCompareExchangeWeak(&dst[wi], old, desired);
        if res.exchanged {
            break;
        }
    }
}
#endif

fn load_src(i: u32) -> f32 {
#ifdef SRC_F32
    return src[i];
#elif defined(SRC_F16)
    return f32(src[i]);
#elif defined(SRC_BF16)
    return bf16_to_f32(src[i / 2u], i % 2u);
#endif
}

fn store_dst(i: u32, v: f32) {
#ifdef SRC_F32
    dst[i] = v;
#elif defined(SRC_F16)
    dst[i] = f16(v);
#elif defined(SRC_BF16)
    bf16_store_at(i, v);
#endif
}

// Axis coordinate tables. Host precomputed in f64 so index/weight selection is
// bit-identical to the CPU reference (see wgpu_backend edge case comments):
//   NEAREST:     [o]         = source index of output o (as f32)
//   BILINEAR:    [3o], [...] = idx0, idx1, weight for output o
@group(0) @binding(2) var<storage, read_write> tab_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> tab_b: array<f32>;

struct Params {
    ne: u32, // total output elements
    bc: u32, // rows (b*c)
    src_h: u32,
    src_w: u32,
    out_h: u32,
    out_w: u32,
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
}

@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let linear = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    if (linear >= params.ne) {
        return;
    }
    let i = linear;

#ifdef NEAREST1D
    // src is treated as contiguous [bc, len] (len == params.src_w).
    let row = i / params.out_w;
    let col = i % params.out_w;
    let src_col = u32(tab_a[col]);
    store_dst(
        params.offset_dst + i,
        load_src(params.offset_src + row * params.src_w + src_col),
    );
#endif

#ifdef NEAREST2D
    let byc = i / (params.out_h * params.out_w);
    let rem = i % (params.out_h * params.out_w);
    let oh = rem / params.out_w;
    let ow = rem % params.out_w;
    let sh = u32(tab_a[oh]);
    let sw = u32(tab_b[ow]);
    store_dst(
        params.offset_dst + i,
        load_src(params.offset_src + byc * params.src_h * params.src_w + sh * params.src_w + sw),
    );
#endif

#ifdef BILINEAR2D
    let byc = i / (params.out_h * params.out_w);
    let rem = i % (params.out_h * params.out_w);
    let oh = rem / params.out_w;
    let ow = rem % params.out_w;
    let h0 = u32(tab_a[3u * oh]);
    let h1 = u32(tab_a[3u * oh + 1u]);
    let wh = tab_a[3u * oh + 2u];
    let w0 = u32(tab_b[3u * ow]);
    let w1 = u32(tab_b[3u * ow + 1u]);
    let ww = tab_b[3u * ow + 2u];
    let base = params.offset_src + byc * params.src_h * params.src_w;
    let idx00 = base + h0 * params.src_w + w0;
    let idx10 = base + h0 * params.src_w + w1;
    let idx01 = base + h1 * params.src_w + w0;
    let idx11 = base + h1 * params.src_w + w1;
    let v00 = load_src(idx00);
    let v10 = load_src(idx10);
    let v01 = load_src(idx01);
    let v11 = load_src(idx11);
    let v_top = v00 * (1.0 - ww) + v10 * ww;
    let v_bottom = v01 * (1.0 - ww) + v11 * ww;
    let value = v_top * (1.0 - wh) + v_bottom * wh;
    store_dst(params.offset_dst + i, value);
#endif
}
