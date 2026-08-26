// Compacted index-select (gather) used when the source table is too large to bind
// in one storage binding (> max_storage_buffer_binding_size). The source is split
// into contiguous row segments; for each segment a compacted list of (relative-src-
// row, dst-offset) pairs drives a single dispatch, so arbitrary scattered dst
// positions are written correctly without binding the whole src.
//
// Bindings:
//   0 src      : array<SRC_TYPE>  — the bound segment (index 0 == segment base row)
//   1 sub_ids  : array<u32>       — per output: relative src row (0..segment_rows)
//   2 dst_map  : array<u32>       — per output: dst element offset (flat, into dst)
//   3 dst      : array<f32>       — the full destination (written via dst_map)
//   4 params   : uniform
//
// One thread per compacted output row; each copies `right_size` contiguous elements.

enable f16;

struct MapParams {
    right_size: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
};

@group(0) @binding(0) var<storage, read> src: array<SRC_TYPE>;
@group(0) @binding(1) var<storage, read> sub_ids: array<u32>;
@group(0) @binding(2) var<storage, read> dst_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: MapParams;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let i = wg.x * WG_SIZE + lid.x;
    if (i >= params.count) {
        return;
    }
    let row = sub_ids[i];
    let dst_off = dst_map[i];
    let src_base = row * params.right_size;
    for (var c: u32 = 0u; c < params.right_size; c++) {
        dst[dst_off + c] = f32(src[src_base + c]);
    }
}
