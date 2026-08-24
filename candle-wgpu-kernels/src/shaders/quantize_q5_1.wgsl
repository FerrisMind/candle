enable f16;

// CPU Q5_1 block = {d: f16, m: f16, qh: [u8; 4], qs: [u8; 16]} = 24 bytes; qs
// byte j packs the low nibble of element j with the low nibble of element
// 16+j and the 5th bit of each goes to qh bit j (resp. j+16). WGSL has no
// byte storage types, so blocks are packed through 32-bit words: each
// workgroup owns 8 consecutive blocks (8*24 = 192 bytes = 48 words), one
// writer per word.
//
// Matches CPU quantize_row_q5_1 exactly: min/max via f32::min/max (IEEE
// minNum/maxNum, so a present -0.0 wins the min tie and +0.0 wins the max
// tie), d = (max-min)/31, id = 1/d via div_ieee (WGSL permits 2-ulp
// division; this driver's OpFDiv is ~1 ulp low), xi = (x-min)*id + 0.5 trunc
// (0..31), qh 5th bits, d and m stored as f16.

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
    scale_bits: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> shmem_min: array<f32, 64>;
var<workgroup> shmem_max: array<f32, 64>;
var<workgroup> shmem_nib: array<u32, 256>; // 8 blocks * 32 nibbles
var<workgroup> shmem_dm: array<u32, 8>; // packed d,m f16 bits per block
var<workgroup> shmem_qh: array<u32, 8>; // qh u32 per block

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8u)) & 0xFFu;
}

// Rust f32::min/f32::max are IEEE minNum/maxNum. On a +-0.0 numeric tie
// minNum picks -0.0 and maxNum picks +0.0; WGSL min/max do not guarantee the
// zero sign, so replicate the Rust tie semantics explicitly.
fn rust_min(a: f32, b: f32) -> f32 {
    if (b < a) { return b; }
    if (a < b) { return a; }
    if ((bitcast<u32>(a) & 0x80000000u) != 0u) { return a; }
    return b;
}

fn rust_max(a: f32, b: f32) -> f32 {
    if (b > a) { return b; }
    if (a > b) { return a; }
    if ((bitcast<u32>(a) & 0x80000000u) == 0u) { return a; }
    return b;
}

// See quantize_q4_1.wgsl: WGSL permits division to be 2 ulp off the IEEE
// result and this driver's OpFDiv measures ~1 ulp low, which breaks byte
// parity with the CPU quantizer at rounding boundaries. Pick whichever of the
// 5 surrounding f32 bit patterns minimizes the (correctly rounded, fma)
// residual |b*q - a|; bit-level/add ops prevent algebraic collapsing.
fn div_ieee(a: f32, b: f32) -> f32 {
    let q = a / b;
    let qb = bitcast<u32>(q);
    var best: f32 = abs(fma(b, q, -a));
    var bestq: u32 = qb;
    let c1 = bitcast<f32>(qb + 1u);
    let r1 = abs(fma(b, c1, -a));
    if (r1 < best) { best = r1; bestq = qb + 1u; }
    let c2 = bitcast<f32>(qb + 2u);
    let r2 = abs(fma(b, c2, -a));
    if (r2 < best) { best = r2; bestq = qb + 2u; }
    let cm1 = bitcast<f32>(qb - 1u);
    let rm1 = abs(fma(b, cm1, -a));
    if (rm1 < best) { best = rm1; bestq = qb - 1u; }
    let cm2 = bitcast<f32>(qb - 2u);
    let rm2 = abs(fma(b, cm2, -a));
    if (rm2 < best) { best = rm2; bestq = qb - 2u; }
    return bitcast<f32>(bestq);
}

// CPU: xi = (x0 + 0.5) as u8 with truncation (no explicit clamp; x0 is in
// [0,31] so the value always fits the 5 bits consumed by qs/qh).
fn quantize5_1_nibble(t: f32) -> u32 {
    return min(31u, u32(t));
}

// Workgroup-local byte offset (0..192) -> the 4 bytes of one u32 word.
fn assemble_q5_1_word(abs_byte: u32) -> u32 {
    var word: u32 = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let b = abs_byte + k;
        let block_in_wg = b / 24u;
        let byte_in_block = b % 24u;
        var byte: u32 = 0u;
        if (block_in_wg < 8u) {
            if (byte_in_block < 4u) {
                byte = get_byte(shmem_dm[block_in_wg], byte_in_block);
            } else if (byte_in_block < 8u) {
                byte = (shmem_qh[block_in_wg] >> (8u * (byte_in_block - 4u))) & 0xFFu;
            } else {
                let qj = byte_in_block - 8u;
                // Q5_1 nibbles carry the 5th bit (0..31); qs stores only the
                // low 4 bits and the 5th bit goes to qh, so mask like the CPU.
                let lo = shmem_nib[block_in_wg * 32u + qj] & 0xFu;
                let hi = shmem_nib[block_in_wg * 32u + qj + 16u] & 0xFu;
                byte = lo | (hi << 4u);
            }
        }
        word = word | (byte << (k * 8u));
    }
    return word;
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let tid = lid.x;
    let block_in_wg = tid / 8u;
    let iqs = tid % 8u;
    let ib = wgid.x * 8u + block_in_wg;

    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    var v2: f32 = 0.0;
    var v3: f32 = 0.0;
    if (ib < params.num_blocks) {
        let a_idx = ib * 32u + iqs * 4u;
        if (a_idx < params.ne) { v0 = src[a_idx]; }
        if (a_idx + 1u < params.ne) { v1 = src[a_idx + 1u]; }
        if (a_idx + 2u < params.ne) { v2 = src[a_idx + 2u]; }
        if (a_idx + 3u < params.ne) { v3 = src[a_idx + 3u]; }
    }

    // Per-thread min/max over the 4 elements, same order as the CPU scan.
    var lmin: f32 = v0;
    var lmax: f32 = v0;
    lmin = rust_min(lmin, v1);
    lmax = rust_max(lmax, v1);
    lmin = rust_min(lmin, v2);
    lmax = rust_max(lmax, v2);
    lmin = rust_min(lmin, v3);
    lmax = rust_max(lmax, v3);
    shmem_min[tid] = lmin;
    shmem_max[tid] = lmax;

    workgroupBarrier();

    let base_tid = block_in_wg * 8u;
    var mn: f32 = shmem_min[base_tid];
    var mx: f32 = shmem_max[base_tid];
    for (var i = 1u; i < 8u; i = i + 1u) {
        mn = rust_min(mn, shmem_min[base_tid + i]);
        mx = rust_max(mx, shmem_max[base_tid + i]);
    }
    // Scale divisor arrives as runtime data: with a compile-time divisor the
    // pipeline specializes a/b and defeats the div_ieee residual fix (see
    // quantize_q4_1.wgsl).
    let d = div_ieee(mx - mn, bitcast<f32>(params.scale_bits));
    let id: f32 = select(div_ieee(1.0, d), 0.0, d == 0.0);

    if (ib < params.num_blocks) {
        let n0 = quantize5_1_nibble((v0 - mn) * id + 0.5);
        let n1 = quantize5_1_nibble((v1 - mn) * id + 0.5);
        let n2 = quantize5_1_nibble((v2 - mn) * id + 0.5);
        let n3 = quantize5_1_nibble((v3 - mn) * id + 0.5);
        shmem_nib[block_in_wg * 32u + iqs * 4u + 0u] = n0;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 1u] = n1;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 2u] = n2;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 3u] = n3;
    }

    workgroupBarrier();

    // qh = OR of each nibble's 5th bit at bit j (elements 0..15) and j+16
    // (elements 16..31), matching the CPU's qh accumulation.
    if (iqs == 0u) {
        shmem_dm[block_in_wg] = pack2x16float(vec2<f32>(d, mn));
        var qh: u32 = 0u;
        for (var j = 0u; j < 16u; j = j + 1u) {
            qh = qh | (((shmem_nib[block_in_wg * 32u + j] >> 4u) & 1u) << j);
            qh = qh | (((shmem_nib[block_in_wg * 32u + j + 16u] >> 4u) & 1u) << (j + 16u));
        }
        shmem_qh[block_in_wg] = qh;
    }

    workgroupBarrier();

    // 48 words per workgroup (192 bytes / 4), one writer per word. Workgroup-
    // LOCAL byte offsets only (0..192); see quantize_q4_1.wgsl for why a
    // global offset would break every workgroup after the first.
    if (tid < 48u) {
        dst[wgid.x * 48u + tid] = assemble_q5_1_word(tid * 4u);
    }
}