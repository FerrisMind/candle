enable f16;

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_F16
#define DataType f16
#endif

// Fused RoPE using precomputed cos/sin tables (the "rotary-half" / NeXT convention,
// same math as candle's CPU/CUDA RotaryEmb custom op).
//
// Inputs (all contiguous):
//   src  : (b, h, t, d)  f32/f16
//   cos  : (t, d/2)        when cs_batched == 0
//          (b, t, d/2)      when cs_batched == 1
//   sin  : same as cos
// Output: dst (b, h, t, d) written out-of-place unless INPLACE.
//
// Each thread rotates one (x[d], x[d + d/2]) pair:
//   dst[d]       = x[d] * cos - x[d+half] * sin
//   dst[d+half]  = x[d+half] * cos + x[d] * sin

struct Params {
    offset_src: u32,
    offset_dst: u32,
    offset_cos: u32,
    offset_sin: u32,
    n_threads: u32, // total pairs = b*h*t*(d/2)
    ne0: u32,       // d (head_dim)
    ne1: u32,       // t (seq_len)
    ne2: u32,       // h (num_heads)
    ne3: u32,       // b (batch)
    cs_batched: u32, // 0 => cos/sin (t, d/2); 1 => (b, t, d/2)
};

@group(0) @binding(0)
var<storage, read_write> src: array<DataType>;
@group(0) @binding(1)
var<storage, read_write> cos: array<DataType>;
@group(0) @binding(2)
var<storage, read_write> sin: array<DataType>;

#ifdef INPLACE
@group(0) @binding(4)
var<uniform> params: Params;
#else
@group(0) @binding(3)
var<storage, read_write> dst: array<DataType>;
@group(0) @binding(4)
var<uniform> params: Params;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let p = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    if (p >= params.n_threads) {
        return;
    }

    let half = params.ne0 / 2u;
    let j = p % half;
    var r = p / half;
    let t_i = r % params.ne1;
    r = r / params.ne1;
    let h_i = r % params.ne2;
    let b_i = r / params.ne2;

    let row_stride = params.ne1 * params.ne0;
    let src_base = params.offset_src + b_i * params.ne2 * row_stride + h_i * row_stride + t_i * params.ne0;
    let i_lo = src_base + j;
    let i_hi = src_base + j + half;

    let cs_base = t_i * half + j;
    let cs_idx = select(params.offset_cos + b_i * (params.ne1 * half) + cs_base,
                        params.offset_cos + cs_base,
                        params.cs_batched == 0u);
    let si_idx = select(params.offset_sin + b_i * (params.ne1 * half) + cs_base,
                        params.offset_sin + cs_base,
                        params.cs_batched == 0u);

    let cosm = cos[cs_idx];
    let sinm = sin[si_idx];
    let x_lo = src[i_lo];
    let x_hi = src[i_hi];

#ifdef INPLACE
    src[i_lo] = x_lo * cosm - x_hi * sinm;
    src[i_hi] = x_hi * cosm + x_lo * sinm;
#else
    dst[i_lo] = x_lo * cosm - x_hi * sinm;
    dst[i_hi] = x_hi * cosm + x_lo * sinm;
#endif
}
