@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> dst: array<u32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // src/dst dimensions
    src_ne0: u32,
    ne1: u32,
    ne2: u32,

    ne0: u32,
    top_k: u32,

    npr: u32,   // tiles per row
    nrows: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shmem_idx: array<u32, WG_SIZE>;

#if ORDER == 0
#define SWAP_COMPARE_UP >
#define SWAP_COMPARE_DOWN <
#else
#define SWAP_COMPARE_UP <
#define SWAP_COMPARE_DOWN >
#endif
// Tie comparators do NOT flip with ORDER: equal keys always resolve so the
// smaller original index ends first in the final output (CPU sort_by
// stability in both directions).
#define TIE_UP >
#define TIE_DOWN <

#ifdef KEY_FN
// Orderable-key helpers (CUDA sort.cu trick): map values to u32 keys that
// preserve source order under unsigned comparison.
fn orderable_from_f32(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    let mask = select(0xFFFFFFFFu, 0x80000000u, (b & 0x80000000u) == 0u);
    return b ^ mask;
}

fn orderable_from_i32(v: i32) -> u32 {
    return bitcast<u32>(v) ^ 0x80000000u;
}

fn orderable_from_u8(v: u32) -> u32 {
    return v;
}

fn decode_f8e4m3(x: u32) -> f32 {
    let b = x & 0xFFu;
    let sign = select(1.0, -1.0, (b & 0x80u) != 0u);
    let exp = (b >> 3u) & 0xFu;
    let man = b & 0x7u;
    if exp == 0u {
        if man == 0u {
            return sign * 0.0;
        }
        return sign * (f32(man) / 8.0) * 0.015625;
    }
    if exp == 0xFu && man == 0x7u {
        return bitcast<f32>(0x7FC00000u);
    }
    return sign * (1.0 + f32(man) / 8.0) * exp2(f32(exp) - 7.0);
}

fn orderable_from_f8e4m3_bits(b: u32) -> u32 {
    let v = decode_f8e4m3(b & 0xFFu);
    return orderable_from_f32(v);
}

fn sort_key(i: u32) -> u32 {
    return KEY_BODY;
}
#else
fn sort_key(i: u32) -> u32 {
    return bitcast<u32>(src[i]);
}
#endif

fn should_swap_up(a_idx: u32, b_idx: u32, row_base: u32) -> bool {
    // Up-box sorts in the FINAL direction. Total order (key, index): equal
    // keys keep ascending original index (CPU sort_by stability in both
    // directions), so the tie comparator flips with the key comparator.
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {
        return !b_oob;
    }
    if (b_oob) {
        return false;
    }
    let ka = sort_key(row_base + a_idx);
    let kb = sort_key(row_base + b_idx);
    if (ka == kb) {
        return a_idx TIE_UP b_idx;
    }
    return ka SWAP_COMPARE_UP kb;
}

fn should_swap_down(a_idx: u32, b_idx: u32, row_base: u32) -> bool {
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {
        return false;
    }
    if (b_oob) {
        return true;
    }
    let ka = sort_key(row_base + a_idx);
    let kb = sort_key(row_base + b_idx);
    if (ka == kb) {
        return a_idx TIE_DOWN b_idx;
    }
    return ka SWAP_COMPARE_DOWN kb;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let linear = wid.x + wid.y * num_wg.x;
    // guard against overprovisioned workgroups
    if (linear >= params.npr * params.nrows) {
        return;
    }
    let tile = linear % params.npr;
    var row = linear / params.npr;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_base = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let tile_base = tile * WG_SIZE;
    let idx = tile_base + lid.x;
    shmem_idx[lid.x] = select(params.src_ne0, idx, idx < params.src_ne0);
    workgroupBarrier();

    var k = 2u;
    while (k <= WG_SIZE) {
        var j = k >> 1;
        while (j > 0) {
            let ixj = lid.x ^ j;
            if (ixj > lid.x) {
                let dir_up = (lid.x & k) == 0;
                let a_idx = shmem_idx[lid.x];
                let b_idx = shmem_idx[ixj];
                let should_swap = select(
                    should_swap_down(a_idx, b_idx, row_base),
                    should_swap_up(a_idx, b_idx, row_base),
                    dir_up);
                if (should_swap) {
                    shmem_idx[lid.x] = b_idx;
                    shmem_idx[ixj] = a_idx;
                }
            }
            workgroupBarrier();
            j >>= 1;
        }
        k <<= 1;
    }

    let out_idx = tile * params.top_k + lid.x;
    if (out_idx < params.ne0 && lid.x < params.top_k) {
        let row_dst = params.offset_dst +
            i1 * params.stride_dst1 +
            i2 * params.stride_dst2 +
            i3 * params.stride_dst3;
        dst[row_dst + out_idx] = shmem_idx[lid.x];
    }
}

// cache-buster v2: full hash validation probe
