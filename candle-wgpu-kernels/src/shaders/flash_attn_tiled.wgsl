// Tiled Flash Attention 2 for wgpu — drop-in replacement for
// flash_attn_simple.wgsl with identical Params/bindings.
// Reference: born flashAttentionShader (Flash Attention 2, Dao et al. 2023).
//
// One workgroup (64 lanes) owns a 64-row Q block of one (batch, head); K and
// V stream through 32-position workgroup-shared tiles, so each K/V element is
// read from global memory once per 64 Q rows instead of once per row — the
// naive kernel re-reads the whole K and V per row (for pi3x decode that is
// ~556 GB of traffic per odd block vs ~4.3 GB here). Online softmax is
// row-local (each lane owns one row), two-phase per tile: scores into
// per-lane registers, per-row block max, single accumulator rescale, then
// accumulate against the shared V tile.
//
// Host must dispatch (ceil(seq_q/64), num_heads, batch_size) workgroups and
// only select this shader when head_dim <= 64 && head_dim_v <= 64 and the
// adapter allows 16 KiB of workgroup storage (2 x 32x64 f32 tiles).
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
    window_size_left: u32,
    window_size_right: u32,
    softcap: f32,
    has_alibi: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> alibi_slopes: array<f32>;

const Q_WG: u32 = 64u;
const KV_TILE: u32 = 32u;
const DIM_MAX: u32 = 64u;
const NEG_BIG: f32 = -3.0e38;

var<workgroup> k_tile: array<f32, KV_TILE * DIM_MAX>;
var<workgroup> v_tile: array<f32, KV_TILE * DIM_MAX>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let b = wid.z;
    let h_q = wid.y;
    let s = wid.x * Q_WG + lid.x;
    let D = params.head_dim;
    let Dv = params.head_dim_v;

    // All lanes reach every workgroupBarrier; rows past seq_q stay idle but
    // must not diverge around the barriers.
    let active = s < params.seq_q && b < params.batch_size && h_q < params.num_heads;

    let gqa_factor = max(1u, params.num_heads / params.num_kv_heads);
    let h_kv = h_q / gqa_factor;
    let q_base = (b * params.num_heads + h_q) * params.seq_q * D;
    let k_base = (b * params.num_kv_heads + h_kv) * params.seq_kv * D;
    let v_base = (b * params.num_kv_heads + h_kv) * params.seq_kv * Dv;
    let o_base = (b * params.num_heads + h_q) * params.seq_q * Dv;
    let q_pos = i32(s + (params.seq_kv - params.seq_q));

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var q_reg = array<f32, DIM_MAX>();
    var o_acc = array<f32, DIM_MAX>();
    if (active) {
        for (var d: u32 = 0u; d < D; d++) {
            q_reg[d] = Q[q_base + s * D + d];
            o_acc[d] = 0.0;
        }
    }

    let kv_tiles = (params.seq_kv + KV_TILE - 1u) / KV_TILE;
    for (var tile: u32 = 0u; tile < kv_tiles; tile++) {
        let tile_start = tile * KV_TILE;
        let kv_count = min(KV_TILE, params.seq_kv - tile_start);

        // Cooperative K/V tile load into shared memory.
        for (var off: u32 = lid.x; off < kv_count * D; off += Q_WG) {
            let kv_i = off / D;
            let d_i = off - kv_i * D;
            k_tile[kv_i * DIM_MAX + d_i] = K[k_base + (tile_start + kv_i) * D + d_i];
            v_tile[kv_i * DIM_MAX + d_i] = V[v_base + (tile_start + kv_i) * Dv + d_i];
        }
        workgroupBarrier();

        if (active) {
            // Phase 1: scores for this tile, per-row block max.
            var scores = array<f32, KV_TILE>();
            var block_max: f32 = NEG_BIG;
            for (var j: u32 = 0u; j < kv_count; j++) {
                let k_pos = i32(tile_start + j);
                var masked = params.causal != 0u && k_pos > q_pos;
                masked = masked
                    || (params.window_size_left != 0u
                        && k_pos < q_pos - i32(params.window_size_left));
                masked = masked
                    || (params.window_size_right != 0u
                        && k_pos > q_pos + i32(params.window_size_right));
                var score: f32 = NEG_BIG;
                if (!masked) {
                    var dot: f32 = 0.0;
                    for (var d: u32 = 0u; d < D; d++) {
                        dot += q_reg[d] * k_tile[j * DIM_MAX + d];
                    }
                    score = dot * params.scale;
                    if (params.has_alibi != 0u) {
                        score += alibi_slopes[h_q] * f32(k_pos - q_pos);
                    }
                    if (params.softcap > 0.0) {
                        score = params.softcap * tanh(score / params.softcap);
                    }
                }
                scores[j] = score;
                block_max = max(block_max, score);
            }

            // Online softmax state update: single rescale per tile.
            let m_new = max(m, block_max);
            let correction = exp(m - m_new);
            l *= correction;
            for (var d: u32 = 0u; d < Dv; d++) {
                o_acc[d] *= correction;
            }
            m = m_new;

            // Phase 2: accumulate against the shared V tile. Fully masked
            // scores are NEG_BIG, so exp(score - m_new) is exactly 0 and the
            // contribution vanishes without a branch.
            for (var j: u32 = 0u; j < kv_count; j++) {
                let w = exp(scores[j] - m_new);
                if (w > 0.0) {
                    for (var d: u32 = 0u; d < Dv; d++) {
                        o_acc[d] += w * v_tile[j * DIM_MAX + d];
                    }
                    l += w;
                }
            }
        }
        workgroupBarrier();
    }

    if (active) {
        for (var d: u32 = 0u; d < Dv; d++) {
            let norm = select(0.0, o_acc[d] / l, l > 0.0);
            O[o_base + s * Dv + d] = norm;
        }
    }
}
