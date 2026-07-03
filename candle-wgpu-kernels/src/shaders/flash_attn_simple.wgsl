// Fused Flash Attention with GQA support for wgpu.
// Computes: O = softmax(Q * K^T * scale + causal_mask) * V
// Uses online softmax to avoid materializing the full attention matrix.
// Handles GQA: num_heads may differ from num_kv_heads.
// Supports different QK and V head dimensions.
//
// Each thread processes one Q row across all KV positions.

struct Params {
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    head_dim_v: u32,
    num_heads: u32,
    num_kv_heads: u32,
    batch_size: u32,
    scale: f32,
    causal: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_q_rows = params.batch_size * params.num_heads * params.seq_q;
    if (gid.x >= total_q_rows) { return; }

    let D = params.head_dim;
    let Dv = params.head_dim_v;
    let bh = gid.x / params.seq_q;
    let s = gid.x % params.seq_q;
    let b = bh / params.num_heads;
    let h_q = bh % params.num_heads;

    // GQA: map Q head to KV head
    let gqa_factor = max(1u, params.num_heads / params.num_kv_heads);
    let h_kv = h_q / gqa_factor;

    let q_base = (b * params.num_heads + h_q) * params.seq_q * D;
    let k_base = (b * params.num_kv_heads + h_kv) * params.seq_kv * D;
    let v_base = (b * params.num_kv_heads + h_kv) * params.seq_kv * Dv;
    let o_base = (b * params.num_heads + h_q) * params.seq_q * Dv;

    // Online softmax state
    var m: f32 = -1e30;
    var l: f32 = 0.0;

    // Output accumulator (max Dv is 128)
    var o_acc = array<f32, 128>();
    for (var d: u32 = 0u; d < Dv; d++) {
        o_acc[d] = 0.0;
    }

    // Iterate over KV positions
    for (var kv: u32 = 0u; kv < params.seq_kv; kv++) {
        // Causal mask with cross-attention offset
        let causal_limit = s + (params.seq_kv - params.seq_q);
        if (params.causal != 0u && kv > causal_limit) { continue; }

        // Compute score = dot(Q[s], K[kv]) * scale
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < D; d++) {
            score += Q[q_base + s * D + d] * K[k_base + kv * D + d];
        }
        score *= params.scale;

        // Online softmax update
        let m_new = max(m, score);
        let exp_diff = select(0.0, exp(m - m_new), m_new - m < 80.0);
        let w = exp(score - m_new);

        // Rescale accumulator and add V contribution
        for (var d: u32 = 0u; d < Dv; d++) {
            o_acc[d] = o_acc[d] * exp_diff + w * V[v_base + kv * Dv + d];
        }

        l = l * exp_diff + w;
        m = m_new;
    }

    // Normalize and write output
    let inv_l = select(0.0, 1.0 / l, l > 0.0);
    for (var d: u32 = 0u; d < Dv; d++) {
        O[o_base + s * Dv + d] = o_acc[d] * inv_l;
    }
}
