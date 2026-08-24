enable f16;

// CPU Q5_0 block = {d: f16, qh: [u8; 4], qs: [u8; 16]} = 22 bytes; qs byte j
// packs the low nibble of element j with the low nibble of element 16+j and
// the 5th bit of each goes to qh bit j (resp. j+16). WGSL has no byte storage
// types, so blocks are packed through 32-bit words: each workgroup owns 8
// consecutive blocks (8*22 = 176 bytes = 44 words), one writer per word.
//
// Matches CPU quantize_row_q5_0 exactly: amax/max come from a first-
// occurrence scan (strictly-greater, so abs ties keep the earlier element),
// d = max / -16 (exact, power of two), id = 1/d via div_ieee (WGSL permits
// 2-ulp division; this driver's OpFDiv is ~1 ulp low), xi = min(31,
// (x*id + 16.5) trunc), d stored as f16.

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> shmem_abs: array<f32, 64>;
var<workgroup> shmem_val: array<f32, 64>; // signed value at max abs
var<workgroup> shmem_nib: array<u32, 256>; // 8 blocks * 32 nibbles
var<workgroup> shmem_d: array<u32, 8>; // packed f16 d bits per block
var<workgroup> shmem_qh: array<u32, 8>; // qh u32 per block

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8u)) & 0xFFu;
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

// CPU: xi = min(31, (x*id + 16.5) as i8) with truncation. x*id lies in
// [-16,16] so the shifted value stays positive and the saturation facet of
// the i8 cast is inert.
fn quantize5_nibble(t: f32) -> u32 {
    return min(31u, u32(t));
}

// Workgroup-local byte offset (0..176) -> the 4 bytes of one u32 word.
fn assemble_q5_0_word(abs_byte: u32) -> u32 {
    var word: u32 = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let b = abs_byte + k;
        let block_in_wg = b / 22u;
        let byte_in_block = b % 22u;
        var byte: u32 = 0u;
        if (block_in_wg < 8u) {
            if (byte_in_block < 2u) {
                byte = get_byte(shmem_d[block_in_wg], byte_in_block);
            } else if (byte_in_block < 6u) {
                byte = (shmem_qh[block_in_wg] >> (8u * (byte_in_block - 2u))) & 0xFFu;
            } else {
                let qj = byte_in_block - 6u;
                // Q5_0 nibbles carry the 5th bit (0..31); qs stores only the
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

    // Per-thread local max abs (strictly-greater so ties keep the earlier
    // element, matching the CPU scan over the same 4 values).
    var la: f32 = 0.0;
    var lv: f32 = 0.0;
    if (abs(v0) > la) { la = abs(v0); lv = v0; }
    if (abs(v1) > la) { la = abs(v1); lv = v1; }
    if (abs(v2) > la) { la = abs(v2); lv = v2; }
    if (abs(v3) > la) { la = abs(v3); lv = v3; }
    shmem_abs[tid] = la;
    shmem_val[tid] = lv;

    workgroupBarrier();

    let base_tid = block_in_wg * 8u;
    var a: f32 = 0.0;
    var m: f32 = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (shmem_abs[base_tid + i] > a) {
            a = shmem_abs[base_tid + i];
            m = shmem_val[base_tid + i];
        }
    }
    let d: f32 = m / -16.0;
    let id: f32 = select(div_ieee(1.0, d), 0.0, d == 0.0);

    if (ib < params.num_blocks) {
        let n0 = quantize5_nibble(v0 * id + 16.5);
        let n1 = quantize5_nibble(v1 * id + 16.5);
        let n2 = quantize5_nibble(v2 * id + 16.5);
        let n3 = quantize5_nibble(v3 * id + 16.5);
        shmem_nib[block_in_wg * 32u + iqs * 4u + 0u] = n0;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 1u] = n1;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 2u] = n2;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 3u] = n3;
    }

    workgroupBarrier();

    // qh = OR of each nibble's 5th bit at bit j (elements 0..15) and j+16
    // (elements 16..31), matching the CPU's qh accumulation.
    if (iqs == 0u) {
        shmem_d[block_in_wg] = pack2x16float(vec2<f32>(d, 0.0));
        var qh: u32 = 0u;
        for (var j = 0u; j < 16u; j = j + 1u) {
            qh = qh | (((shmem_nib[block_in_wg * 32u + j] >> 4u) & 1u) << j);
            qh = qh | (((shmem_nib[block_in_wg * 32u + j + 16u] >> 4u) & 1u) << (j + 16u));
        }
        shmem_qh[block_in_wg] = qh;
    }

    workgroupBarrier();

    // 44 words per workgroup (176 bytes / 4), one writer per word. Workgroup-
    // LOCAL byte offsets only (0..176); see quantize_q4_1.wgsl for why a
    // global offset would break every workgroup after the first.
    if (tid < 44u) {
        dst[wgid.x * 44u + tid] = assemble_q5_0_word(tid * 4u);
    }
}