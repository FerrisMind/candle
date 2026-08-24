// Fused DECODE attention (single query token per head) for the m == 1 case.
//
// Replaces, per layer: repeat_kv × 2 (cats), the score GEMV, the causal-mask add,
// softmax, and the context matmul (6-7 model ops) with ONE dispatch.
//
//   out[h, d] = sum_p exp(scale * (Q[h] · K[kv_h][p])) * V[kv_h][p][d]
//               / sum_p exp(scale * (Q[h] · K[kv_h][p]))
//
// GQA is handled in-kernel: kv_h = h / (n_q_heads / n_kv_heads). Decode attends to
// ALL cached positions (no causal mask; the caller only routes l == 1 here).
//
// Algorithm (portable, deterministic, NO subgroup intrinsics — naga rejects
// `enable subgroups` on wgpu 29; workgroup-memory tree reductions only):
//   - ONE workgroup of 128 threads per query head (grid.x == n_q_heads).
//   - Phase 0: threads 0..127 load q[h][*] into workgroup memory once.
//   - KV positions are processed in TILES of Bc == 64 (online/flash softmax):
//       1. Score: thread p (0..63) does the FULL head_dim dot serially in ascending
//          d order (deterministic f32 FMA order), `score *= scale`, -> sco[p].
//       2. Tile max: shmem tree reduction over 64 -> tile_max; online M/L rescale
//          (apple flash_attn_base: M = max(Mold, rowwise), eM = exp(Mold - M),
//          L and O_ac rescale by eM, then P = exp(S - M), L += sum P, P · V).
//       3. V pass: thread d (0..127) rescales its accumulator, then loops p over
//          the tile accumulating `O_ac[d] += w_p * V[kv_h][p][d]` (coalesced along
//          d; V read once per tile, single-pass attention) and the row sum.
//   - Final: O[h][d] = O_ac[d] / L.
//
// Numerics: same FMA sets as the standard `q@k^T -> softmax -> P@V` chain
// (reordered + online rescale), expect |diff| ~1e-6 rel — well under the F32
// parity tolerance. All strides/offsets come from the layouts so a `narrow`ed
// prefix of a grown KV-cache backing buffer (capacity > live length) routes here
// without materializing V^T.

struct FusedDecodeAttnParams {
    kv_len: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_o: u32,
    q_head_stride: u32,
    q_d_stride: u32,
    k_head_stride: u32,
    k_pos_stride: u32,
    k_d_stride: u32,
    v_head_stride: u32,
    v_pos_stride: u32,
    v_d_stride: u32,
    o_head_stride: u32,
    o_d_stride: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> params: FusedDecodeAttnParams;

const WG_SIZE: u32 = 128u;
const BC: u32 = 64u; // positions per tile
const NEG_INF: f32 = -3.402823466e+38;

var<workgroup> qsh: array<f32, 128>; // q head vector (head_dim <= 128)
var<workgroup> sco: array<f32, 64>; // tile scores (scaled)
var<workgroup> red: array<f32, 64>; // tree-reduction scratch
var<workgroup> acc: array<f32, 128>; // per-d output accumulator

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    let tid = lid.x;
    let hd = params.head_dim;
    let kv_len = params.kv_len;
    let kv_head = h / (params.n_q_heads / params.n_kv_heads);

    // Phase 0: load the query head into workgroup memory (thread d owns dim d).
    if (tid < hd) {
        qsh[tid] = q[params.offset_q + h * params.q_head_stride + tid * params.q_d_stride];
        acc[tid] = 0.0f;
    }
    workgroupBarrier();

    var m_run: f32 = NEG_INF; // online running max (identical on every thread)
    var l_run: f32 = 0.0f; // online running row sum (identical on every thread)

    let k_head_base = params.offset_k + kv_head * params.k_head_stride;
    let v_head_base = params.offset_v + kv_head * params.v_head_stride;
    let n_tiles = (kv_len + BC - 1u) / BC;

    for (var t: u32 = 0u; t < n_tiles; t++) {
        let tile_base = t * BC;

        // ---- 1. scores (thread p in 0..63; threads 64..127 idle here) -------
        if (tid < BC) {
            let p = tile_base + tid;
            var dot: f32 = 0.0f;
            if (p < kv_len) {
                let kb = k_head_base + p * params.k_pos_stride;
                for (var d: u32 = 0u; d < hd; d++) {
                    dot = fma(qsh[d], k[kb + d * params.k_d_stride], dot);
                }
            }
            // Invalid trailing positions get NEG_INF so they never win the max,
            // and exp(NEG_INF - m) == 0 in the V pass.
            sco[tid] = select(NEG_INF, dot * params.scale, p < kv_len);
        }
        workgroupBarrier();

        // ---- 2. tile max via shmem tree over red[] --------------------------
        if (tid < BC) {
            red[tid] = sco[tid];
        }
        workgroupBarrier();
        for (var stride = BC / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                red[tid] = max(red[tid], red[tid + stride]);
            }
            workgroupBarrier();
        }
        let tile_max = red[0];

        // ---- 3. online M/L rescale + exp weights + V accumulation -----------
        let m_new = max(m_run, tile_max);
        let eM = exp(m_run - m_new);
        m_run = m_new;
        // L = L * eM + tile_sum (online denominator rescale, same as O_ac).
        l_run = l_run * eM;

        if (tid < hd) {
            acc[tid] = acc[tid] * eM;
        }

        // Thread d accumulates over the whole tile: P = exp(sco - m), then
        // acc[d] += P[p] * V[kv_h][p][d], and the (thread-uniform) row-sum l.
        var lsum: f32 = 0.0f;
        if (tid < hd) {
            for (var p: u32 = 0u; p < BC; p++) {
                let gp = tile_base + p;
                if (gp < kv_len) {
                    let s = sco[p];
                    let w = exp(s - m_run);
                    lsum += w;
                    let vb = v_head_base + gp * params.v_pos_stride + tid * params.v_d_stride;
                    acc[tid] += w * v[vb];
                }
            }
            l_run += lsum;
        }
        workgroupBarrier();
    }

    // Final: O[h][d] = O_ac[d] / L.
    if (tid < hd) {
        o[params.offset_o + h * params.o_head_stride + tid * params.o_d_stride] =
            acc[tid] / l_run;
    }
}