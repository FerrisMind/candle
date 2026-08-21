enable f16;

// CPU Q4_0 block = {d: f16, qs: [u8; 16]} = 18 bytes; qs byte j packs element
// j (low nibble) and element 16+j (high nibble). WGSL has no byte storage
// types, so blocks are packed through 32-bit words: each workgroup owns 8
// consecutive blocks (8*18 = 144 bytes = 36 words), one writer per word, no
// writes race on the misaligned boundaries. d = max_abs_value / -8 with the
// signed value at max abs, matching CPU quantize_row_q4_0 exactly.

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
var<workgroup> shmem_d: array<u32, 8>;

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8u)) & 0xFFu;
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

    // Per-thread local max (strictly-greater so ties keep the earlier elem).
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
    let d: f32 = m / -8.0;
    let id: f32 = select(1.0 / d, 0.0, d == 0.0);

    if (ib < params.num_blocks) {
        let n0 = quantize_nibble(v0 * id);
        let n1 = quantize_nibble(v1 * id);
        let n2 = quantize_nibble(v2 * id);
        let n3 = quantize_nibble(v3 * id);
        shmem_nib[block_in_wg * 32u + iqs * 4u + 0u] = n0;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 1u] = n1;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 2u] = n2;
        shmem_nib[block_in_wg * 32u + iqs * 4u + 3u] = n3;
    }

    workgroupBarrier();

    if (iqs == 0u) {
        shmem_d[block_in_wg] = pack2x16float(vec2<f32>(d, 0.0));
    }

    workgroupBarrier();

    // 36 words per workgroup (144 bytes / 4). One writer per word.
    if (tid < 36u) {
        dst[wgid.x * 36u + tid] = assemble_q4_word(wgid.x * 144u + tid * 4u);
    }
}

// CPU: xi = min(15, (x*id + 8.5) as u8) with truncation; negatives clamp to 0.
fn quantize_nibble(t: f32) -> u32 {
    let shifted = t + 8.5;
    var ni: u32 = 0u;
    if (shifted > 0.0) {
        ni = u32(shifted);
    }
    return min(ni, 15u);
}

fn assemble_q4_word(abs_byte: u32) -> u32 {
    var word: u32 = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let b = abs_byte + k;
        let block_in_wg = b / 18u;
        let byte_in_block = b % 18u;
        var byte: u32 = 0u;
        if (block_in_wg < 8u) {
            if (byte_in_block < 2u) {
                byte = get_byte(shmem_d[block_in_wg], byte_in_block);
            } else {
                let qj = byte_in_block - 2u;
                let lo = shmem_nib[block_in_wg * 32u + qj];
                let hi = shmem_nib[block_in_wg * 32u + qj + 16u];
                byte = lo | (hi << 4u);
            }
        }
        word = word | (byte << (k * 8u));
    }
    return word;
}