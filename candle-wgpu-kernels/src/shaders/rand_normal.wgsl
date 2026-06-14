#include "rand_common.wgsl"

struct Params {
    seed_lo: u32,
    seed_hi: u32,
    mean: f32,
    stddev: f32,
    ne: u32,
}

@group(0) @binding(0)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(1)
var<uniform> params: Params;

const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }
    let u1 = max(rand01_from_seed(params.seed_lo, params.seed_hi, gid.x * 2u), 1.19209290e-7);
    let u2 = rand01_from_seed(params.seed_lo, params.seed_hi, gid.x * 2u + 1u);
    let mag = params.stddev * sqrt(-2.0 * log(u1));
    let z0 = mag * cos(TWO_PI * u2) + params.mean;
    dst[gid.x] = DataType(z0);
}
