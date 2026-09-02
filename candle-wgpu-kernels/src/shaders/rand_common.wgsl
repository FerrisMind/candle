fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn u64_add_u32(a: vec2<u32>, b: u32) -> vec2<u32> {
    return u64_add(a, vec2<u32>(b, 0u));
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn u64_shr(a: vec2<u32>, bits: u32) -> vec2<u32> {
    if (bits == 0u) {
        return a;
    }
    if (bits >= 32u) {
        return vec2<u32>(a.y >> (bits - 32u), 0u);
    }
    return vec2<u32>((a.x >> bits) | (a.y << (32u - bits)), a.y >> bits);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Correct 64x64 -> low 64 bits. Each u32 term carries its own hi-32 via
    // the 16-bit decomposition; the (a0*b1 + a1*b0) term only contributes its
    // low 32 bits to the high word (bits 32-63).
    let a0 = a.x;
    let a1 = a.y;
    let b0 = b.x;
    let b1 = b.y;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let mid = p01 + p10;
    let hi = u64_mul_hi32(a0, b0) + mid;
    return vec2<u32>(p00, hi);
}

fn u64_mul_hi32(a: u32, b: u32) -> u32 {
    // hi-32 of a 32x32 -> 64 product, via 16x16 sub-products (each fits u32).
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p01 = a_hi * b_lo;
    let p10 = a_lo * b_hi;
    let p11 = a_hi * b_hi;
    let m = p01 + p10;
    let m_bit32 = select(0u, 1u, m < p01);
    let m_shift16 = m_bit32 * 0x10000u + (m >> 16u);
    let l = a_lo * b_lo;
    let base = (m & 0xFFFFu) << 16u;
    let low = base + l;
    let low_carry = select(0u, 1u, low < base);
    return p11 + m_shift16 + low_carry;
}

fn splitmix64(x: vec2<u32>) -> vec2<u32> {
    var v = u64_add(x, vec2<u32>(0x7F4A7C15u, 0x9E3779B9u));
    v = u64_xor(v, u64_shr(v, 30u));
    v = u64_mul(v, vec2<u32>(0x1CE4E5B9u, 0xBF58476Du));
    v = u64_xor(v, u64_shr(v, 27u));
    v = u64_mul(v, vec2<u32>(0x133111EBu, 0x94D049BBu));
    return u64_xor(v, u64_shr(v, 31u));
}

fn rand01_from_seed(seed_lo: u32, seed_hi: u32, idx: u32) -> f32 {
    let seed = vec2<u32>(seed_lo, seed_hi);
    let state = u64_add_u32(seed, idx);
    let r = splitmix64(state);
    return f32(r.y >> 8u) * (1.0 / 16777216.0);
}
