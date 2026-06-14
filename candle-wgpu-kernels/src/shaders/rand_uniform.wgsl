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
    let u = rand01(params.seed_lo, params.seed_hi, gid.x);
    let span = params.max_val - params.min_val;
    dst[gid.x] = DataType(params.min_val + u * span);
}
