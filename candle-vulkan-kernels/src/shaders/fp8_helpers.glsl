// FP8 E4M3 conversion helpers for Vulkan shaders.
// Ported from the float8 crate (MIT-licensed) — matches the Rust f8e4m3
// round-trip semantics exactly.
//
// Reference: float8-0.7.0/src/lib.rs (convert_fp8_to_fp16, convert_to_fp8)

#ifndef FP8_HELPERS_GLSL
#define FP8_HELPERS_GLSL

// Decode an E4M3 byte into a float32_t. Matches F8E4M3::to_f32().
float f8e4m3_to_f32(uint8_t x) {
    // Direct e4m3 -> f32 bit construction (positions: sign 31, exp 23..26,
    // mantissa 20..22). The original port mixed f16-shift math from the
    // float8 crate (which operates on ur = byte<<8) with raw-byte masks —
    // producing wrong values. Mirrors the validated wgpu decode path.
    uint b = uint(x);
    uint sign = (b & 0x80u) << 24u;         // -> bit 31
    uint exp  = (b & 0x78u) << 20u;         // 4 exp bits -> 23..26 (biased +120 handled below)
    uint man  = (b & 0x07u) << 20u;         // into temp f32-bit position for extraction
    uint absx = b & 0x7Fu;

    if (absx == 0x7Fu) {
        // NaN (e4m3 has no Inf): canonical f32 NaN.
        return uintBitsToFloat(0x7FC00000u);
    }
    uint e4 = (b >> 3u) & 0xFu;
    uint m4 = b & 0x7u;
    if (e4 == 0u) {
        if (m4 == 0u) {
            return uintBitsToFloat(sign);
        }
        // Subnormal: value = m/8 * 2^-6
        float v = float(m4) * 0.125 * 0.015625;
        return uintBitsToFloat(sign) < 0.0 ? -v : v;
    }
    // Normal: (1 + m/8) * 2^(e-7); f32 bias: e + 120
    uint f32 = sign | ((e4 + 120u) << 23u) | (m4 << 20u);
    return uintBitsToFloat(f32);
}

// Encode a float32_t into an E4M3 byte. Matches F8E4M3::from_f32().
// Saturates to finite max (no inf in E4M3 sat-finite mode).
uint8_t f32_to_f8e4m3(float v) {
    // Direct f32 -> e4m3 encode with round-to-nearest-even, saturating to the
    // finite max (e4m3 has no Inf). Mirrors the validated wgpu encode path
    // (which matched the float8 crate exactly).
    uint bits = floatBitsToUint(v);
    uint sign = (bits >> 24u) & 0x80u;   // capture top bit into byte position
    uint absx = bits & 0x7FFFFFFFu;

    // NaN -> canonical NaN byte 0x7F
    if (absx > 0x7F800000u) {
        return uint8_t(0x7Fu);
    }
    // Inf or overflow (>= 448) -> saturate to max finite 0x7E
    if (absx >= 0x43E00000u) {
        return uint8_t(sign | 0x7Eu);
    }
    if (absx == 0u) {
        return uint8_t(sign);
    }
    // Underflow to zero: < half of smallest subnormal (2^-9)
    if (absx < 0x3B000000u) {
        return uint8_t(sign);
    }

    int e32 = int((absx >> 23u) & 0xFFu) - 127;   // unbiased f32 exponent
    uint m32 = absx & 0x7FFFFFu;                   // 23-bit mantissa

    // e4m3 normal range: exponent in [-6, 8]
    if (e32 >= -6) {
        uint e4 = uint(e32 + 7);                   // biased e4m3 exponent
        uint m4 = m32 >> 20u;                      // top 3 mantissa bits
        uint rem = m32 & 0xFFFFFu;                 // rounding bits below
        uint key = (e4 << 3u) | m4;
        // RNE: round bit 0x80000 (half), sticky = anything below.
        uint round_bit = (rem >> 19u) & 1u;
        uint sticky = rem & 0x7FFFFu;
        if (round_bit == 1u && (sticky != 0u || (key & 1u) != 0u)) {
            key += 1u;
        }
        if (key >= 0x7Eu) {
            // Overflow beyond max finite.
            key = 0x7Eu;
        }
        return uint8_t(sign | key);
    }
    // Subnormal: value = m4/8 * 2^-6; e32 in [-9, -7]. Quantize the 24-bit
    // fixed fraction v = all_bits * 2^(e32-23) to units of 2^-9.
    // shift = bits dropped from the 24-bit fraction so remaining = v * 2^9.
    int shift = 20 + (-6 - e32);   // e.g. e32=-7 -> 19, e32=-9 -> 23
    uint quant = (0x800000u | m32) >> uint(shift);
    uint key = quant;
        // RNE on the dropped bits: reconstruct dropped mask
    uint dropped_mask = (1u << uint(shift)) - 1u;
    uint all_bits = 0x800000u | m32;
    uint dropped = all_bits & dropped_mask;
    uint half_way = 1u << uint(shift - 1);
    if (dropped > half_way || (dropped == half_way && (key & 1u) != 0u)) {
        key += 1u;
    }
    if (key >= 8u) {
        // Carried into the normal range: exponent becomes 1, mantissa 0.
        return uint8_t(sign | 0x08u);
    }
    return uint8_t(sign | key);
}

#endif // FP8_HELPERS_GLSL