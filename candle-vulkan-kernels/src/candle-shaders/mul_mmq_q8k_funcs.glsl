// A/B-side helpers for the fused Q8K GEMM (`mul_mmq_q8k.comp`).
//
// The A-side (weight) unpacking is the llama.cpp `mul_mmq_funcs.glsl` code for
// the k-quants, copied verbatim: the quantized weight layout is unchanged. The
// B-side is redefined for the packed Q8K activation (`block_q8_k`, one f32
// scale per 256-element block, int8 values) instead of the per-32 f16
// `block_q8_1_x4` grid, and the dot products apply the f32 scale / exact
// integer min correction the CPU `QMatMul` contract requires.

#include "types.glsl"

#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    // Repack 4x4 quants into one int
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 2) | (vals2 << 4) | (vals3 << 6);

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
        buf_a[buf_ib].scales = unpack8(uint32_t(data_a_packed16[ib_k].scales[iqs_k / 8])).xy; // vec4 used due to #12147
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].scales = buf_a[buf_ib].scales;

    [[unroll]] for (uint iqs = 0; iqs < 2; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const uint8_t scale = cache_a[ib_a].scales[iqs / 4];
        const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x03030303);

        sum_d += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]) * (scale & 0xF);
        sum_m += dotPacked4x8EXT(scale_m, cache_b.qs[iqs]);
    }

    // CPU: dall * sum_d - dmin * sum_m with dall = x.d * y.d, dmin = x.dmin * y.d
    // (the activation f32 scale `y.d` factors out).
    return ACC_TYPE(float(cache_b.d) * (float(cache_a[ib_a].dm.x) * float(sum_d) - float(cache_a[ib_a].dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint hm_idx = iqs * QUANT_R_MMQ;
    const uint iqs_k = (ib % 8) * 8 + hm_idx;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // Repack 2x4 quants into one int
    // Add the 3rd bit instead of subtracting it to allow packing the quants
    // vec4 for unpack8 used due to #12147
    const i8vec2 vals00 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2        ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2    ] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals01 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 1    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 1] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals10 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 2    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 2] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals11 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 3    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 3] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    buf_a[buf_ib].qs[iqs] = pack32(u8vec4(vals00.x, vals00.y, vals01.x, vals01.y)) |
                           (pack32(u8vec4(vals10.x, vals10.y, vals11.x, vals11.y)) << 4);

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = i8vec2(unpack8(uint32_t(((data_a_packed16[ib_k].scales[(is % 8      ) / 2] >> (4 * (is / 8))) & 0x0F0F) |
                                                     (((data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2] >> (2 * (is / 4))) & 0x0303) << 4))).xy); // vec4 used due to #12147

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales - 32));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        // Subtract 4 from the quants to correct the 3rd bit offset
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.d) * result);
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    // Repack 2x4 quants into one int
#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 4);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs * QUANT_R_MMQ;
    const uint qh_shift = iqs_k / 8;

    buf_a[buf_ib].qs[iqs] = int32_t(((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
                                   (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4));
#endif

    if (iqs == 0) {
        // Scale index
        const uint is = iqs_k / 8;
        u8vec2 scale_dm;
        if (is < 4) {
            scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
        } else {
            scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                              (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
        }

        buf_a[buf_ib].dm = FLOAT_TYPEV2(vec2(data_a_packed32[ib_k].dm) * vec2(scale_dm));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8 / QUANT_R_MMQ; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
#if defined(DATA_A_Q4_K)
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
#else // defined(DATA_A_Q5_K)
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
#endif

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    // CPU min term: -dmin * sum_j(bsums[j] * mins[j/2]) with dmin = x.dmin * y.d;
    // `sum32` is the exact integer sum of this window's activation values, so
    // `dm.y * sum32` == the CPU bsums/min product for the window.
    return ACC_TYPE(float(cache_b.d) * (float(cache_a[ib_a].dm.x) * float(q_sum) - float(cache_a[ib_a].dm.y) * float(cache_b.sum32)));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    buf_a[buf_ib].qs[iqs] = pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y));

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = unpack8(int32_t(data_a_packed16[ib_k].scales[is / 2])).xy;

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.d) * result);
}
#endif

// B side: packed Q8K activation. `ib` is a 32-element window index, so the
// 256-element block is `ib / 8` and the window within it is `ib % 8`.
void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        if (iqs == 0) {
            buf_b[buf_ib].d = data_b[ib / 8].d;
        }

        const ivec4 values = data_b[ib / 8].qs[(ib % 8) * 2 + iqs];
        buf_b[buf_ib].qs[iqs * 4    ] = values.x;
        buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
        buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
        buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
    } else {
        if (iqs == 0) {
            buf_b[buf_ib].d = 0.0f;
        }

        buf_b[buf_ib].qs[iqs * 4    ] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 1] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 2] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 3] = 0;
    }
}

void block_b_to_registers(const uint ib) {
    cache_b.d = buf_b[ib].d;

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    // Exact integer sum of the window's 32 activation values, computed once per
    // window (it is reused by every A row of the tile).
    int32_t sum32 = 0;
#endif

    [[unroll]] for (uint iqs = 0; iqs < BK / 4; iqs++) {
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
        sum32 += dotPacked4x8EXT(int32_t(0x01010101), cache_b.qs[iqs]);
#endif
    }

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    cache_b.sum32 = sum32;
#endif
}
