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
    let a0 = a.x;
    let a1 = a.y;
    let b0 = b.x;
    let b1 = b.y;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10;
    let mid_lo = mid;
    var mid_hi = select(0u, 1u, mid < p01);
    let lo = p00 + mid_lo;
    let carry = select(0u, 1u, lo < p00);
    let hi = p11 + mid_hi + carry;
    return vec2<u32>(lo, hi);
}

fn splitmix64(x: vec2<u32>) -> vec2<u32> {
    var v = u64_add(x, vec2<u32>(0x9E3779B9u, 0x7F4A7C15u));
    v = u64_xor(v, u64_shr(v, 30u));
    v = u64_mul(v, vec2<u32>(0xBF58476Du, 0x1CE4E5B9u));
    v = u64_xor(v, u64_shr(v, 27u));
    v = u64_mul(v, vec2<u32>(0x94D049BBu, 0x133111EBu));
    return u64_xor(v, u64_shr(v, 31u));
}

fn rand01_from_seed(seed_lo: u32, seed_hi: u32, idx: u32) -> f32 {
    let seed = vec2<u32>(seed_lo, seed_hi);
    let state = u64_add_u32(seed, idx);
    let r = splitmix64(state);
    return f32(r.y >> 8u) * (1.0 / 16777216.0);
}
