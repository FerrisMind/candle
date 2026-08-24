// Flash Attention with a paged (block-table) KV cache for wgpu.
//
// Functional analogue of CUDA candle-flash-attn::flash_attn_with_kvcache
// (paged mode; causal => window_size_left=None, window_size_right=Some(0)).
//
// Semantics (strictly matching the CUDA reference):
//   * q           : (total_q, num_heads_q, head_dim), contiguous
//   * k / v       : (num_blocks, page_block_size, num_heads_kv, head_dim), contiguous
//   * block_table : I32 (batch, max_blocks_per_seq), last dim contiguous;
//                   block_table[b][i] = physical block holding page i of sequence b
//   * cu_seq_q / cu_seq_k : I32 cumulative per-sequence lengths, batch+1 entries
//   * causal      : keep k only when its in-sequence index <= q's in-sequence index
//   * output      : (total_q, num_heads_q, head_dim_v), dtype = dtype(q)
//
// Paged addressing: for the KV position with in-sequence index j of sequence b,
//   page_idx = j / page_block_size,  page_row = j % page_block_size,
//   block    = block_table[b * max_blocks + page_idx],
//   kv_row   = block * page_block_size + page_row
// (k/v are flattened as (num_blocks * page_block_size, num_heads_kv, head_dim)).
// Negative block entries are skipped (invalid pages contribute nothing).
//
// Each invocation handles one (global q row, q head) pair and runs online
// softmax over the KV span of the owning sequence (bounded by cu_seq_k).
// Accumulator and running max/sum are f32 (CUDA flash-attn parity).
//
// Grid: gid.x = global q row (+params.offset_q for row-chunked dispatch),
//       gid.y = q head.
// Workgroup: 256 threads; dispatch (ceil(rows/256), num_heads, 1).

struct Params {
    total_q: u32,
    num_blocks: u32,
    head_dim: u32,
    head_dim_v: u32,
    num_heads: u32,
    num_kv_heads: u32,
    batch_size: u32,
    max_seqlen_q: u32,
    max_seqlen_k: u32,
    offset_q: u32,
    page_block_size: u32,
    max_blocks: u32,
    causal: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> cu_seq_q: array<i32>;
@group(0) @binding(4) var<storage, read> cu_seq_k: array<i32>;
@group(0) @binding(5) var<storage, read> block_table: array<i32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_row = gid.x + params.offset_q;
    if (q_row >= params.total_q) { return; }
    let h_q = gid.y;
    if (h_q >= params.num_heads) { return; }

    let D = params.head_dim;
    let Dv = params.head_dim_v;
    let P = params.page_block_size;

    // Locate the owning sequence: cu_seq_q[b] <= q_row < cu_seq_q[b + 1].
    // Binary search over the batch because sequences are ragged.
    var b: u32 = 0u;
    var hi: u32 = params.batch_size;
    while (b + 1u < hi) {
        let mid = (b + hi) / 2u;
        if (cu_seq_q[mid] <= i32(q_row)) { b = mid; } else { hi = mid; }
    }
    let q_local = q_row - u32(cu_seq_q[b]);
    let kv_start = u32(cu_seq_k[b]);
    let kv_end = u32(cu_seq_k[b + 1u]);

    // GQA: map q head to its kv head.
    let h_kv = h_q / (params.num_heads / params.num_kv_heads);

    // Online softmax state (f32 accumulation).
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;

    // Output accumulator. head_dim_v must be <= 512 (validated host-side).
    var o_acc = array<f32, 512>();
    for (var d: u32 = 0u; d < Dv; d++) {
        o_acc[d] = 0.0;
    }

    // Element offsets (tensors are contiguous; k/v indexed via block_table).
    let q_base = q_row * params.num_heads * D + h_q * D;
    let o_base = q_row * params.num_heads * Dv + h_q * Dv;

    for (var kv: u32 = kv_start; kv < kv_end; kv++) {
        let k_local = kv - kv_start;
        // Causal mask (CUDA window_size_left=None, window_size_right=Some(0)):
        // keep k only if its in-sequence index <= the query's.
        if (params.causal != 0u && q_local < k_local) { continue; }

        // Paged addressing: locate the physical block for this page.
        let page_idx = k_local / P;
        let page_row = k_local % P;
        let blk = block_table[b * params.max_blocks + page_idx];
        if (blk < 0) { continue; }
        let kv_row = u32(blk) * P + page_row;

        var score: f32 = 0.0;
        let k_off = kv_row * params.num_kv_heads * D + h_kv * D;
        for (var d: u32 = 0u; d < D; d++) {
            score += Q[q_base + d] * K[k_off + d];
        }
        score *= params.scale;

        let m_new = max(m, score);
        let exp_diff = select(0.0, exp(m - m_new), m_new - m < 80.0);
        let w = exp(score - m_new);

        let v_off = kv_row * params.num_kv_heads * Dv + h_kv * Dv;
        for (var d: u32 = 0u; d < Dv; d++) {
            o_acc[d] = o_acc[d] * exp_diff + w * V[v_off + d];
        }
        l = l * exp_diff + w;
        m = m_new;
    }

    let inv_l = select(0.0, 1.0 / l, l > 0.0);
    for (var d: u32 = 0u; d < Dv; d++) {
        O[o_base + d] = o_acc[d] * inv_l;
    }
}
