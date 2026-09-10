// Last-dim argmax/argmin over a possibly strided source: one workgroup per
// row, reduced-dim stride taken from params. mode: 0 = argmax, 1 = argmin.
// Ties resolve to the smaller within-row index (same as the contiguous
// argmax.wgsl kernel).
@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    stride_src0: u32, // reduced dim
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    ne0: u32, // reduced dim len
    ne1: u32,
    ne2: u32,

    row_begin: u32,
    mode: u32,
    _pad0: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

const FLOAT_MIN: f32 = -3.4028235e38;
const FLOAT_MAX: f32 = 3.4028235e38;

struct Pair {
    value: f32,
    index: i32,
};

var<workgroup> shared_pair: array<Pair, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = params.row_begin + wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;

    var local_pair = Pair(FLOAT_MAX, -1);
    if (params.mode == 0u) {
        local_pair = Pair(FLOAT_MIN, -1);
    }
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        let val = src[row + col * params.stride_src0];
        if (params.mode == 0u) {
            if (val > local_pair.value) {
                local_pair = Pair(val, i32(col));
            }
        } else {
            if (val < local_pair.value) {
                local_pair = Pair(val, i32(col));
            }
        }
    }
    shared_pair[lid.x] = local_pair;
    workgroupBarrier();

    var offset: u32 = WG_SIZE >> 1;
    while (offset > 0u) {
        if (lid.x < offset) {
            let a = shared_pair[lid.x];
            let b = shared_pair[lid.x + offset];
            if (params.mode == 0u) {
                if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
                    shared_pair[lid.x] = b;
                }
            } else {
                if (b.value < a.value || (b.value == a.value && b.index < a.index)) {
                    shared_pair[lid.x] = b;
                }
            }
        }
        workgroupBarrier();
        offset >>= 1;
    }

    if (lid.x == 0u) {
        dst[params.offset_dst + params.row_begin + wid.x] = shared_pair[0].index;
    }
}
