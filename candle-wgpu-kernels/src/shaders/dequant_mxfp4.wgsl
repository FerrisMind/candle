struct DequantParams {
    nel: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;

fn e8m0_to_fp32(e: u32) -> f32 {
    if (e == 0u) {
        return 0.0;
    }
    return bitcast<f32>(e << 23u);
}

fn kvalue_mxfp4(idx: u32) -> f32 {
    let kvals = array<f32, 16>(
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
        0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0
    );
    return kvals[idx & 0xFu];
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let block_idx = gid.x;
    let total_blocks = params.nel / 32u;
    if (block_idx >= total_blocks) {
        return;
    }

    // MXFP4 block is 17 bytes (packed into 5 u32 words, or aligned to 20 bytes = 5 words)
    // Word 0 byte 0 is scale e8m0
    let block_word_base = block_idx * 5u;
    let w0 = src[block_word_base];
    let e_u8 = w0 & 0xFFu;
    let d = e8m0_to_fp32(e_u8);

    let out_base = block_idx * 32u;

    // Remaining bytes of word 0 (bytes 1..3) and words 1..4 contain 16 bytes = 32 nibbles
    // Unpack 16 bytes across 4 words
    for (var l = 0u; l < 4u; l = l + 1u) {
        let w = src[block_word_base + 1u + l];
        for (var b = 0u; b < 4u; b = b + 1u) {
            let byte_val = (w >> (b * 8u)) & 0xFFu;
            let v0 = d * 0.5 * kvalue_mxfp4(byte_val & 0xFu);
            let v1 = d * 0.5 * kvalue_mxfp4((byte_val >> 4u) & 0xFu);
            let out_idx0 = out_base + l * 8u + b;
            let out_idx1 = out_base + l * 8u + b + 16u;
            if (out_idx0 < params.nel) {
                dst[out_idx0] = v0;
            }
            if (out_idx1 < params.nel) {
                dst[out_idx1] = v1;
            }
        }
    }
}
