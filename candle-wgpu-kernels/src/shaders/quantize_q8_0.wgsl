enable f16;

// CPU Q8_0 block = {d: f16, qs: [i8; 32]} = 34 bytes. WGSL has no byte
// storage types, so we write the packed stream through 32-bit words: each
// workgroup owns exactly 8 consecutive blocks (8*34 = 272 bytes = 68 words),
// and every word is built by a single thread, so no write races on the
// misaligned 34-byte block boundaries. The unaligned tail of the final
// workgroup (fewer than 8 valid blocks) lands in a padded buffer tail and is
// never read back (len_bytes truncates at num_blocks * 34).

struct QuantizeParams {
    ne: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: QuantizeParams;

var<workgroup> shmem_max: array<f32, 64>;
var<workgroup> shmem_q: array<u32, 256>; // 8 blocks * 32 q bytes
var<workgroup> shmem_d: array<u32, 8>; // packed f16 bit pattern per block

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

    let tmax = max(max(abs(v0), abs(v1)), max(abs(v2), abs(v3)));
    shmem_max[tid] = tmax;

    workgroupBarrier();

    let base_tid = block_in_wg * 8u;
    var amax: f32 = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        amax = max(amax, shmem_max[base_tid + i]);
    }
    let d: f32 = amax / 127.0;
    let id: f32 = select(1.0 / d, 0.0, d == 0.0);

    let q0 = i32(round(v0 * id));
    let q1 = i32(round(v1 * id));
    let q2 = i32(round(v2 * id));
    let q3 = i32(round(v3 * id));

    if (ib < params.num_blocks) {
        // Low byte of the i8 two's-complement value matches CPU `as i8`.
        shmem_q[block_in_wg * 32u + iqs * 4u + 0u] = u32(q0 & 0xFF);
        shmem_q[block_in_wg * 32u + iqs * 4u + 1u] = u32(q1 & 0xFF);
        shmem_q[block_in_wg * 32u + iqs * 4u + 2u] = u32(q2 & 0xFF);
        shmem_q[block_in_wg * 32u + iqs * 4u + 3u] = u32(q3 & 0xFF);
    }

    workgroupBarrier();

    if (iqs == 0u) {
        shmem_d[block_in_wg] = pack2x16float(vec2<f32>(d, 0.0));
    }

    workgroupBarrier();

    // Word assembly: each of the 68 words (bytes 4w..4w+4 of the workgroup's
    // 272-byte range) is written by exactly one thread.
    let w0 = tid;
    if (w0 < 68u) {
        dst[wgid.x * 68u + w0] = assemble_word(wgid.x * 272u + w0 * 4u);
    }
    if (tid < 4u) {
        let wext = 64u + tid;
        dst[wgid.x * 68u + wext] = assemble_word(wgid.x * 272u + wext * 4u);
    }
}

fn assemble_word(abs_byte: u32) -> u32 {
    var word: u32 = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let b = abs_byte + k;
        let block_in_wg = b / 34u;
        let byte_in_block = b % 34u;
        let blk_valid = (block_in_wg < 8u); // 0..7 always within this wg's span
        var byte: u32 = 0u;
        if (blk_valid) {
            if (byte_in_block < 2u) {
                let dword = shmem_d[block_in_wg];
                byte = get_byte(dword, byte_in_block);
            } else {
                byte = shmem_q[block_in_wg * 32u + (byte_in_block - 2u)];
            }
        }
        word = word | (byte << (k * 8u));
    }
    return word;
}