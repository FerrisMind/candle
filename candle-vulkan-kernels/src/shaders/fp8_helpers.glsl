// FP8 E4M3 conversion helpers for Vulkan shaders.
// Ported from the float8 crate (MIT-licensed) — matches the Rust f8e4m3
// round-trip semantics exactly.
//
// Reference: float8-0.7.0/src/lib.rs (convert_fp8_to_fp16, convert_to_fp8)

#ifndef FP8_HELPERS_GLSL
#define FP8_HELPERS_GLSL

// Decode an E4M3 byte into a float32_t. Matches F8E4M3::to_f32().
float f8e4m3_to_f32(uint8_t x) {
    // convert_fp8_to_fp16 for E4M3, then f16→f32.
    uint sign = uint(x) & 0x80u;
    uint exponent = ((uint(x) & 0x78u) >> 1u) + 0x2000u;
    uint mantissa = (uint(x) & 0x07u) << 1u;
    uint absx = uint(x) & 0x7Fu;

    uint ur;
    if (absx == 0x7Fu) {
        // NaN — return canonical NaN
        ur = 0x7FFFu;
    } else if (exponent == 0x2000u) {
        // Zero or denormal
        if (mantissa != 0u) {
            // Normalize denormal
            mantissa = mantissa << 1u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa = mantissa << 1u;
                exponent = exponent - 0x0400u;
            }
            mantissa = mantissa & 0x03FFu;
        } else {
            exponent = 0u;
        }
        ur = sign | exponent | mantissa;
    } else {
        ur = sign | exponent | mantissa;
    }

    // Convert f16 bits to f32
    uint s = (ur >> 15u) & 1u;
    uint e = (ur >> 10u) & 0x1Fu;
    uint m = ur & 0x3FFu;

    uint f32;
    if (e == 0u) {
        // Zero or subnormal
        if (m == 0u) {
            f32 = s << 31u;
        } else {
            // Normalize subnormal
            e = 1u;
            while ((m & 0x400u) == 0u) {
                m = m << 1u;
                e = e - 1u;
            }
            m = m & 0x3FFu;
            e = 127u - 15u + e;
            f32 = (s << 31u) | (e << 23u) | (m << 13u);
        }
    } else if (e == 31u) {
        // Inf or NaN
        f32 = (s << 31u) | 0x7F800000u | (m << 13u);
    } else {
        // Normal
        e = 127u - 15u + e;
        f32 = (s << 31u) | (e << 23u) | (m << 13u);
    }
    return uintBitsToFloat(f32);
}

// Encode a float32_t into an E4M3 byte. Matches F8E4M3::from_f32().
// Saturates to finite max (no inf in E4M3 sat-finite mode).
uint8_t f32_to_f8e4m3(float v) {
    const uint FP8_MAXNORM = 0x7Eu;
    const uint FP8_MANTISSA_MASK = 0x7u;
    const uint FP8_EXP_BIAS = 7u;
    const uint FP8_SIGNIFICAND_BITS = 4u;
    // DP_HALF_ULP for f64: 1 << (53 - significand - 1) = 1 << 48
    // But we're working in f32, so adjust: f32 has 24 significand bits.
    // We need: round-off = 1 << (24 - significand_bits - 1) = 1 << 19
    const uint FP32_HALF_ULP = 1u << (24u - FP8_SIGNIFICAND_BITS - 1u);
    const uint FP8_OVERFLOW_THRESHOLD = 0x407D0000u; // f32 bits for ~448.0
    const uint FP8_MINNORM = 0x3F900000u; // f32 bits for 2^-6 = 0.015625
    const uint FP8_MINDENORM_O2 = 0x3F500000u; // f32 bits for 2^-7 / 2 = ~0.003906

    uint bits = floatBitsToUint(v);
    uint sign = (bits >> 31u) << 7u;
    uint absx = bits & 0x7FFFFFFFu;

    uint res;
    if (absx <= FP8_MINDENORM_O2) {
        res = 0u;
    } else if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        // NaN → preserve as NaN
        res = 0x7Fu;
    } else if (absx > FP8_OVERFLOW_THRESHOLD) {
        // Saturate to max finite
        res = FP8_MAXNORM;
    } else if (absx >= FP8_MINNORM) {
        // Normal range
        int exp = int((absx >> 23u) & 0xFFu) - 127 + int(FP8_EXP_BIAS);
        uint mantissa = (absx >> (23u - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK;
        res = (uint(exp) << (FP8_SIGNIFICAND_BITS - 1u)) | mantissa;
        // Round to nearest even
        uint round = absx & ((FP32_HALF_ULP << 1u) - 1u);
        if ((round > FP32_HALF_ULP) || ((round == FP32_HALF_ULP) && ((mantissa & 1u) != 0u))) {
            res = res + 1u;
        }
    } else {
        // Denormal
        int shift = 1 - int((absx >> 23u) & 0xFFu) + 127 - int(FP8_EXP_BIAS);
        uint mantissa = ((absx >> (23u - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK)
                      | (1u << (FP8_SIGNIFICAND_BITS - 1u));
        res = mantissa >> uint(shift);
        uint round = (absx | (1u << 22u)) & ((FP32_HALF_ULP << uint(shift + 1)) - 1u);
        if ((round > (FP32_HALF_ULP << uint(shift)))
            || ((round == (FP32_HALF_ULP << uint(shift))) && ((res & 1u) != 0u)))
        {
            res = res + 1u;
        }
    }

    return uint8_t(res | sign);
}

#endif // FP8_HELPERS_GLSL