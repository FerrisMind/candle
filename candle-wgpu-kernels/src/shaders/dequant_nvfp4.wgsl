struct DequantParams {
    nel: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;

fn ue4m3_to_fp32(u: u32) -> f32 {
    if (u == 0u || u == 127u) {
        return 0.0;
    }
    let exp = (u >> 3u) & 15u;
    let man = u & 7u;
    if (exp == 0u) {
        return f32(man) * (1.0 / 512.0);
    }
    let full_exp = exp + 120u;
    let full_man = man << 20u;
    return bitcast<f32>((full_exp << 23u) | full_man);
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
    let sub_idx = gid.x;
    let total_subs = params.nel / 16u;
    if (sub_idx >= total_subs) {
        return;
    }

    let block_idx = sub_idx / 4u;
    let sub_in_block = sub_idx % 4u;

    // NVFP4 block is 32 bytes (8 u32 words): 4 bytes of scale d[4] (1 word), then 28 bytes of qs (7 words)
    // d[4] is packed into word 0
    let block_word_base = block_idx * 8u;
    let scales_word = src[block_word_base];
    let d_u8 = (scales_word >> (sub_in_block * 8u)) & 0xFFu;
    let d = ue4m3_to_fp32(d_u8);

    // sub-block has 8 bytes = 2 u32 words of packed nibbles
    let qs_word_base = block_word_base + 1u + sub_in_block * 2u;
    let w0 = src[qs_word_base];
    let w1 = src[qs_word_base + 1u];

    let out_base = sub_idx * 16u;

    // Word 0: 4 bytes = 8 elements
    for (var b = 0u; b < 4u; b = b + 1u) {
        let byte_val = (w0 >> (b * 8u)) & 0xFFu;
        let v0 = d * 0.5 * kvalue_mxfp4(byte_val & 0xFu);
        let v1 = d * 0.5 * kvalue_mxfp4((byte_val >> 4u) & 0xFu);
        if (out_base + b < params.nel) {
            dst[out_base + b] = v0;
        }
        if (out_base + b + 8u < params.nel) {
            dst[out_base + b + 8u] = v1;
        }
    }

    // Word 1: 4 bytes = 8 elements
    for (var b = 0u; b < 4u; b = b + 1u) {
        let byte_val = (w1 >> (b * 8u)) & 0xFFu;
        let v0 = d * 0.5 * kvalue_mxfp4(byte_val & 0xFu);
        let v1 = d * 0.5 * kvalue_mxfp4((byte_val >> 4u) & 0xFu);
        if (out_base + b + 4u < params.nel) {
            dst[out_base + b + 4u] = v0;
        }
        if (out_base + b + 12u < params.nel) {
            dst[out_base + b + 12u] = v1;
        }
    }
}
