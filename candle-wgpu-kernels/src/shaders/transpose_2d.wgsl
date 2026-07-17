// Contiguous rank-2 transpose: out[j, i] = in[i, j] for shape (rows, cols) -> (cols, rows).
struct TransposeParams {
    rows: u32, // of input
    cols: u32, // of input
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: TransposeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;
    if (idx >= total) {
        return;
    }
    let i = idx / params.cols; // row of src
    let j = idx % params.cols; // col of src
    // dst is (cols, rows): row j, col i
    dst[j * params.rows + i] = src[idx];
}
