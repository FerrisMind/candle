#include "rand_common.wgsl"

struct Params {
    seed_lo: u32,
    seed_hi: u32,
    min_val: f32,
    max_val: f32,
    ne: u32,
}

@group(0) @binding(0)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }
    let u = rand01_from_seed(params.seed_lo, params.seed_hi, gid.x);
    let span = params.max_val - params.min_val;
    dst[gid.x] = DataType(params.min_val + u * span);
}
