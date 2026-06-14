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

fn splitmix32(state: u32) -> u32 {
    var z = state + 0x9E3779B9u;
    z = (z ^ (z >> 16u)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13u)) * 0xC2B2AE35u;
    return z ^ (z >> 16u);
}

fn rand01(seed_lo: u32, seed_hi: u32, idx: u32) -> f32 {
    let mixed_hi = (seed_hi << 16u) | (seed_hi >> 16u);
    let state = seed_lo ^ idx ^ mixed_hi;
    let r = splitmix32(state);
    return f32(r >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }
    let u1 = max(rand01(params.seed_lo, params.seed_hi, gid.x * 2u), 1.19209290e-7);
    let u2 = rand01(params.seed_lo, params.seed_hi, gid.x * 2u + 1u);
    let mag = params.stddev * sqrt(-2.0 * log(u1));
    let z0 = mag * cos(TWO_PI * u2) + params.mean;
    dst[gid.x] = DataType(z0);
}
