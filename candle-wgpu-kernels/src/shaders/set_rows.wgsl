enable f16;

#ifdef DST_F32
#define DST_INNER_TYPE f32
#else
#define DST_INNER_TYPE f16
#endif

#ifdef VEC4
#define SRC_TYPE vec4<f32>
#define DST_TYPE vec4<DST_INNER_TYPE>
#define VEC_SIZE 4
#else
#define SRC_TYPE f32
#define DST_TYPE DST_INNER_TYPE
#define VEC_SIZE 1
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

#ifdef ADD
@group(0) @binding(2)
var<storage, read_write> dst: array<atomic<u32>>;
#else
@group(0) @binding(2)
var<storage, read_write> dst: array<DST_TYPE>;
#endif

#ifdef I64_IDX
@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;
#define PARAMS_BINDING 4
#else
#define PARAMS_BINDING 3
#endif

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

#ifdef ADD
// Keep f32 and f16 CAS helpers behind separate defines so F32 preprocess can
// strip `enable f16;` without leaving f16 tokens in the module (naga rejects
// f16 types unless the extension is enabled).
#ifndef ADD_F16
fn atomic_add_f32(dst_idx: u32, value: f32) {
    loop {
        let old_bits = atomicLoad(&dst[dst_idx]);
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let result = atomicCompareExchangeWeak(&dst[dst_idx], old_bits, new_bits);
        if result.exchanged {
            return;
        }
    }
}
#else
// Native f16 scatter-add: CAS on the u32 word packing two halves.
// `f16_elem_idx` is in f16 elements (same indexing as non-atomic f16 set_rows).
fn atomic_add_f16(f16_elem_idx: u32, value: f16) {
    let word_idx = f16_elem_idx / 2u;
    let shift = (f16_elem_idx % 2u) * 16u;
    let mask = 0xFFFFu << shift;
    loop {
        let old_bits = atomicLoad(&dst[word_idx]);
        let old_h = (old_bits >> shift) & 0xFFFFu;
        let old_v = f16(unpack2x16float(old_h).x);
        let new_v = old_v + value;
        let new_h = pack2x16float(vec2<f32>(f32(new_v), 0.0)) & 0xFFFFu;
        let new_bits = (old_bits & ~mask) | (new_h << shift);
        let result = atomicCompareExchangeWeak(&dst[word_idx], old_bits, new_bits);
        if result.exchanged {
            return;
        }
    }
}
#endif
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= (params.ne3 * params.ne2 * params.n_rows * params.ne0) / VEC_SIZE) {
        return;
    }

    // getting the row from gid
    let elems_per_row = params.ne0 / VEC_SIZE;
    var i = gid.x / elems_per_row;

    let i_src3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_src2 = i / params.n_rows;
    let i_src1 = i % params.n_rows;

    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

#ifdef I64_IDX
    let idx_high = (params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2) * 2;

    let idx_val = idx[idx_high];
    let idx_low_val = idx[idx_high + 1];

    if (idx_low_val != 0) {
        // Upper bits of index are not zero, output will be incorrect
        atomicStore(&error, 1);
        return;
    }
#else
    let idx_i = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;
    let idx_val = idx[idx_i];
#endif

    let i_dst_row = params.offset_dst + idx_val * params.stride_dst1 + i_src2 * params.stride_dst2 + i_src3 * params.stride_dst3;
    let i_src_row = params.offset_src + i_src1 * params.stride_src1 + i_src2 * params.stride_src2 + i_src3 * params.stride_src3;

    let col_idx = (gid.x % elems_per_row);
#ifdef ADD
#ifdef ADD_F16
    atomic_add_f16(i_dst_row/VEC_SIZE + col_idx, f16(src[i_src_row/VEC_SIZE + col_idx]));
#else
    atomic_add_f32(i_dst_row/VEC_SIZE + col_idx, f32(src[i_src_row/VEC_SIZE + col_idx]));
#endif
#else
    dst[i_dst_row/VEC_SIZE + col_idx] = DST_TYPE(src[i_src_row/VEC_SIZE + col_idx]);
#endif
}
