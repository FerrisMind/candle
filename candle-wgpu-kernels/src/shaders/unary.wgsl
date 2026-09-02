#ifdef TYPE_F16
enable f16;
#define TYPE f16
#else
#define TYPE f32
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<TYPE>;

#ifndef INPLACE
@group(0) @binding(1)
var<storage, read_write> dst: array<TYPE>;
#define PARAMS_BINDING 2
#else
#define PARAMS_BINDING 1
#endif

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    ne0: u32,
    ne1: u32,
    ne2: u32,
#ifdef CLAMP
    clamp_min: f32,
    clamp_max: f32,
#endif
#ifdef FILL
    fill_val: f32,
#endif
#ifdef XIELU
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,
#endif

};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

#ifdef GELU_ERF
// Port of libm::erff (musl / FreeBSD s_erff.c) so GELU_ERF matches the CPU
// reference GeluErf::f32 (candle-core/src/op.rs). WGSL has no built-in erf.
const ERX: f32 = 8.4506291151e-01;
const EFX8: f32 = 1.0270333290e+00;
const PP0: f32 = 1.2837916613e-01;
const PP1: f32 = -3.2504209876e-01;
const PP2: f32 = -2.8481749818e-02;
const PP3: f32 = -5.7702702470e-03;
const PP4: f32 = -2.3763017452e-05;
const QQ1: f32 = 3.9791721106e-01;
const QQ2: f32 = 6.5022252500e-02;
const QQ3: f32 = 5.0813062117e-03;
const QQ4: f32 = 1.3249473704e-04;
const QQ5: f32 = -3.9602282413e-06;
const PA0: f32 = -2.3621185683e-03;
const PA1: f32 = 4.1485610604e-01;
const PA2: f32 = -3.7220788002e-01;
const PA3: f32 = 3.1834661961e-01;
const PA4: f32 = -1.1089469492e-01;
const PA5: f32 = 3.5478305072e-02;
const PA6: f32 = -2.1663755178e-03;
const QA1: f32 = 1.0642088205e-01;
const QA2: f32 = 5.4039794207e-01;
const QA3: f32 = 7.1828655899e-02;
const QA4: f32 = 1.2617121637e-01;
const QA5: f32 = 1.3637083583e-02;
const QA6: f32 = 1.1984500103e-02;
const RA0: f32 = -9.8649440333e-03;
const RA1: f32 = -6.9385856390e-01;
const RA2: f32 = -1.0558626175e+01;
const RA3: f32 = -6.2375331879e+01;
const RA4: f32 = -1.6239666748e+02;
const RA5: f32 = -1.8460508728e+02;
const RA6: f32 = -8.1287437439e+01;
const RA7: f32 = -9.8143291473e+00;
const SA1: f32 = 1.9651271820e+01;
const SA2: f32 = 1.3765776062e+02;
const SA3: f32 = 4.3456588745e+02;
const SA4: f32 = 6.4538726807e+02;
const SA5: f32 = 4.2900814819e+02;
const SA6: f32 = 1.0863500214e+02;
const SA7: f32 = 6.5702495575e+00;
const SA8: f32 = -6.0424413532e-02;
const RB0: f32 = -9.8649431020e-03;
const RB1: f32 = -7.9928326607e-01;
const RB2: f32 = -1.7757955551e+01;
const RB3: f32 = -1.6063638306e+02;
const RB4: f32 = -6.3756646729e+02;
const RB5: f32 = -1.0250950928e+03;
const RB6: f32 = -4.8351919556e+02;
const SB1: f32 = 3.0338060379e+01;
const SB2: f32 = 3.2579251099e+02;
const SB3: f32 = 1.5367296143e+03;
const SB4: f32 = 3.1998581543e+03;
const SB5: f32 = 2.5530502930e+03;
const SB6: f32 = 4.7452853394e+02;
const SB7: f32 = -2.2440952301e+01;

fn erfc1(x: f32) -> f32 {
    let s = abs(x) - 1.0;
    let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    return 1.0 - ERX - p / q;
}

fn erfc2(ix_abs: u32, x: f32) -> f32 {
    if ix_abs < 0x3fa00000u {
        return erfc1(x);
    }
    let xa = abs(x);
    let s = 1.0 / (xa * xa);
    var r: f32;
    var big_s: f32;
    if ix_abs < 0x4036db6du {
        r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        big_s = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
    } else {
        r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        big_s = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
    }
    let z = bitcast<f32>(ix_abs & 0xffffe000u);
    return exp(-z * z - 0.5625) * exp((z - xa) * (z + xa) + r / big_s) / xa;
}

fn erf(x: f32) -> f32 {
    let ix = bitcast<u32>(x);
    let neg = (ix >> 31u) != 0u;
    let ix_abs = ix & 0x7fffffffu;
    if ix_abs >= 0x7f800000u {
        // erf(nan)=nan, erf(+inf)=+1, erf(-inf)=-1
        if neg {
            return -1.0 + 1.0 / x;
        }
        return 1.0 + 1.0 / x;
    }
    if ix_abs < 0x3f580000u {
        // |x| < 0.84375
        if ix_abs < 0x31800000u {
            // |x| < 2**-28, avoid underflow
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    var y: f32;
    if ix_abs < 0x40c00000u {
        // |x| < 6
        y = 1.0 - erfc2(ix_abs, x);
    } else {
        // 1.0 - 2^-120 (== 1.0 in f32)
        y = 1.0 - bitcast<f32>(0x03800000u);
    }
    if neg {
        return -y;
    }
    return y;
}
#endif

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let linear = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    if (linear >= params.ne) {
      return;
    }
    var i = linear;
    let ne2 = params.ne2;
#ifdef DIAG
    let ne1 = params.ne0;
#else
    let ne1 = params.ne1;
#endif
    let ne0 = params.ne0;

    let i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    let i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    let i1 = i / ne0;
    let i0 = i % ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

#ifdef ABS
    let res = abs(src[params.offset_src + src_idx]);
#endif
#ifdef SGN
    let res = select(TYPE(select(0.0, -1.0, src[params.offset_src + src_idx] < 0.0)), TYPE(1.0),
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef NEG
    let res = -src[params.offset_src + src_idx];
#endif
#ifdef STEP
    let res = TYPE(select(0.0, 1.0, src[params.offset_src + src_idx] > 0.0));
#endif
#ifdef TANH
    let res = tanh(clamp(src[params.offset_src + src_idx], -9.010913, 9.010913));
#endif
#ifdef RELU
    let res = select(0.0, src[params.offset_src + src_idx], src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef ELU
    let res = select(exp(src[params.offset_src + src_idx]) - 1.0, src[params.offset_src + src_idx],
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef HARDSIGMOID
    let res = min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef SIGMOID
    let res = 1.0 / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef SILU
    let res = src[params.offset_src + src_idx] / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef EXP
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32));
#endif
#ifdef LOG
    let res = TYPE(log(f32(src[params.offset_src + src_idx])));
#endif
#ifdef CLAMP
    let res = clamp(src[params.offset_src + src_idx], TYPE(params.clamp_min), TYPE(params.clamp_max));
#endif
#ifdef FILL
    let res = TYPE(params.fill_val);
#endif
#ifdef HARDSWISH
    let res = src[params.offset_src + src_idx] *
              min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef GELU
    let res = 0.5 * src[params.offset_src + src_idx] *
              (1.0 + tanh(clamp(sqrt(2.0 / 3.14159265) *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] * src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_QUICK
    let res = src[params.offset_src + src_idx] * 0.5 *
              (1.0 + tanh(clamp(0.79788456 *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] *
                                    src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_ERF
    // Exact erf-GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Computed in f32 (see
    // module-scope `erf` below, a port of libm::erff) so it matches the CPU
    // reference GeluErf::f32 (candle-core/src/op.rs).
    let src_val = f32(src[params.offset_src + src_idx]);
    let erf_x = erf(src_val * 0.7071067811865476); // FRAC_1_SQRT_2
    let res = TYPE(0.5 * src_val * (1.0 + erf_x));
#endif
#ifdef XIELU
    let val = f32(src[params.offset_src + src_idx]);
    let res =
        TYPE(select(
            ((exp(min(val, params.eps)) - 1.0) - val) * params.alpha_n + params.beta * val,
            params.alpha_p * val * val + params.beta * val,
            val > 0.0));
#endif
#ifdef SOFTPLUS
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(select(log(1.0 + exp(src_f32)), src_f32, src_f32 > 20.0));
#endif
#ifdef EXPM1
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32) - 1.0);
#endif
#ifdef FLOOR
    let res = floor(src[params.offset_src + src_idx]);
#endif
#ifdef CEIL
    let res = ceil(src[params.offset_src + src_idx]);
#endif
#ifdef ROUND
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let result = select(ceil(src_f32 - 0.5), floor(src_f32 + 0.5), src_f32 >= 0.0);
    let res = TYPE(result);
#endif
#ifdef TRUNC
    let res = trunc(src[params.offset_src + src_idx]);
#endif
#ifdef SQR
    let res = src[params.offset_src + src_idx] * src[params.offset_src + src_idx];
#endif
#ifdef SQRT
    let res = TYPE(sqrt(f32(src[params.offset_src + src_idx])));
#endif
#ifdef SIN
    let res_f32 = sin(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif
#ifdef COS
    let res_f32 = cos(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif
#ifdef DIAG
    let res = select(0.0, src[params.offset_src + i0 + i2 * params.stride_src2 + i3 * params.stride_src3], i0 == i1);
#endif
#ifdef TRI
#ifdef TRI_TYPE_LOWER
    let res = select(0.0, src[params.offset_src + src_idx], i0 < i1);
#elif TRI_TYPE_LOWER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 <= i1);
#elif TRI_TYPE_UPPER
    let res = select(0.0, src[params.offset_src + src_idx], i0 > i1);
#elif TRI_TYPE_UPPER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 >= i1);
#endif
#endif

#ifdef INPLACE
    src[params.offset_src + src_idx] = res;
#else
    dst[params.offset_dst + linear] = res;
#endif
}
