// E4M3 decode/encode helpers.
// Reference: float8 crate convert_fp8_to_fp16 / convert_to_fp8, NVIDIA cuda_fp8.hpp

fn fp8e4m3_to_f16(bits: u32) -> u32 {
    // Replicate float8 crate convert_fp8_to_fp16 exactly:
    // ur = (bits as u16) << 8, then extract sign/exp/mant from ur.
    let ur = (bits & 0xFFu) << 8u;
    let sign = ur & 0x8000u;
    var exponent = ((ur & 0x7800u) >> 1u) + 0x2000u;
    var mantissa = (ur & 0x0700u) >> 1u;
    let absx = bits & 0x7Fu;

    if absx == 0x7Fu {
        return 0x7FFFu;
    }
    if exponent == 0x2000u {
        if mantissa != 0u {
            mantissa = mantissa << 1u;
            while (mantissa & 0x0400u) == 0u {
                mantissa = mantissa << 1u;
                exponent = exponent - 0x0400u;
            }
            mantissa = mantissa & 0x03FFu;
        } else {
            exponent = 0u;
        }
        return sign | exponent | mantissa;
    }
    return sign | exponent | mantissa;
}

fn f16_bits_to_f32_bits(h: u32) -> u32 {
    // Deterministic f16 -> f32 bit expansion (no float conversion, NaN-safe).
    // fp8e4m3_to_f16 already normalizes subnormals, so exp==0 implies zero.
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let man = h & 0x03FFu;
    // fp8e4m3_to_f16 normalizes fp8 subnormals, so f16 exp==0 implies zero.
    if exp == 0u {
        return sign;
    }
    if exp == 0x1Fu {
        if man == 0u {
            return sign | 0x7F800000u; // Inf
        }
        // Canonical f32 NaN (matches CUDA fp8->float conversion semantics).
        return 0x7FC00000u;
    }
    // Re-bias exponent: f16 bias 15 -> f32 bias 127 (+112).
    return sign | ((exp + 112u) << 23u) | (man << 13u);
}

fn f32_to_fp8e4m3(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign_bit = (bits >> 31u) & 1u;
    let abs_bits = bits & 0x7FFFFFFFu;

    // NaN -> 0x7F
    if abs_bits > 0x7F800000u {
        return 0x7Fu | (sign_bit << 7u);
    }
    // Inf -> saturate to 0x7E
    if abs_bits >= 0x7F800000u {
        return 0x7Eu | (sign_bit << 7u);
    }
    // Zero
    if abs_bits == 0u {
        return sign_bit << 7u;
    }

    let abs_v = abs(v);
    // Overflow: >= 448.0 -> 0x7E
    if abs_v >= 448.0 {
        return 0x7Eu | (sign_bit << 7u);
    }
    // Underflow: < half of smallest subnormal -> zero
    if abs_v < 0.0009765625 {
        return sign_bit << 7u;
    }

    let f32_exp_unbiased = i32((abs_bits >> 23u) & 0xFFu) - 127i;
    let f32_mant = abs_bits & 0x7FFFFFu;
    let target_exp = f32_exp_unbiased + 7i;

    if target_exp <= 0i {
        // Subnormal
        let shift = u32(1i - target_exp);
        let full_mant = 0x800000u | f32_mant;
        let shifted = full_mant >> (23u - 3u + shift);
        let round_pos = 23u - 3u + shift - 1u;
        let round_bit = (full_mant >> round_pos) & 1u;
        let has_sticky = (full_mant & ((1u << round_pos) - 1u)) != 0u;
        var result = shifted;
        if round_bit != 0u && (has_sticky || (result & 1u) != 0u) {
            result += 1u;
        }
        if result > 0x07u {
            result = 0x08u;
        }
        return result | (sign_bit << 7u);
    }

    // e4m3 exponent field max is 15 (biased): values up to 448 are legal.
    // Only target_exp > 15 (i.e. >= 16) overflows and saturates to 0x7E.
    if target_exp >= 16i {
        return 0x7Eu | (sign_bit << 7u);
    }

    // Normal
    let mantissa = (f32_mant >> (23u - 3u)) & 0x07u;
    var result = (u32(target_exp) << 3u) | mantissa;
    let round_pos = 23u - 3u - 1u;
    let round_bit = (f32_mant >> round_pos) & 1u;
    let has_sticky = (f32_mant & ((1u << round_pos) - 1u)) != 0u;
    if round_bit != 0u && (has_sticky || (result & 1u) != 0u) {
        result += 1u;
    }
    if result >= 0x7Eu {
        return 0x7Eu | (sign_bit << 7u);
    }
    return result | (sign_bit << 7u);
}

// ============================================================
// Entry points
// ============================================================

struct Params {
    ne: u32,       // number of logical elements (f8 bytes for decode, f32 for encode)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

#ifdef DECODE
// f8e4m3 -> f32: src is packed u32 (4 bytes per word), dst is f32 bits
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let wi = gid.x;
    let base = wi * 4u;
    if base >= params.ne { return; }
    let word = src[wi];
    if base < params.ne {
        dst[base] = f16_bits_to_f32_bits(fp8e4m3_to_f16(word & 0xFFu));
    }
    if base + 1u < params.ne {
        dst[base + 1u] = f16_bits_to_f32_bits(fp8e4m3_to_f16((word >> 8u) & 0xFFu));
    }
    if base + 2u < params.ne {
        dst[base + 2u] = f16_bits_to_f32_bits(fp8e4m3_to_f16((word >> 16u) & 0xFFu));
    }
    if base + 3u < params.ne {
        dst[base + 3u] = f16_bits_to_f32_bits(fp8e4m3_to_f16((word >> 24u) & 0xFFu));
    }
}
#endif

#ifdef ENCODE
// f32 -> f8e4m3: src is f32 bits (array<u32>), dst is packed u32
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let wi = gid.x;
    let base = wi * 4u;
    if base >= params.ne { return; }
    var b0 = 0u;
    var b1 = 0u;
    var b2 = 0u;
    var b3 = 0u;
    if base < params.ne {
        b0 = f32_to_fp8e4m3(bitcast<f32>(src[base]));
    }
    if base + 1u < params.ne {
        b1 = f32_to_fp8e4m3(bitcast<f32>(src[base + 1u]));
    }
    if base + 2u < params.ne {
        b2 = f32_to_fp8e4m3(bitcast<f32>(src[base + 2u]));
    }
    if base + 3u < params.ne {
        b3 = f32_to_fp8e4m3(bitcast<f32>(src[base + 3u]));
    }
    dst[wi] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
#endif