// A/B-side helpers for the fused Q8K matvec (`mul_mat_vec_q8k.comp`).
//
// The A-side (weight) unpacking is the llama.cpp `mul_mat_vecq_funcs.glsl`
// code for the k-quants (Q2_K, Q3_K, Q4_K/Q5_K, Q6_K), copied verbatim: the
// quantized weight layout is unchanged. Legacy, MXFP4 and IQ parts are
// dropped. The B-side dot products are redefined for the packed Q8K
// activation (`block_q8_k`, one f32 scale per 256-element block) instead of
// the per-32 f16 `block_q8_1_x4` grid, and apply the f32 scale / exact
// integer min correction the CPU `QMatMul` contract requires. The B-side
// caches `cache_b_d` / `cache_b_qs` are globals of `mul_mat_vec_q8k.comp`.
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

#if defined(DATA_A_Q2_K)
FLOAT_TYPEV2 get_dm(uint ib) {
    const uint ib_k = ib / 8;
    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
}
#endif

#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return i32vec4((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303);
}

uint8_t get_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const uint8_t scale = get_scale(ib_a, iqs * 4);
    const vec2 dm = vec2(get_dm(ib_a));
    const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    sum_d += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[0]);

    sum_d += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[1]);

    sum_d += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[2]);

    sum_d += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_d) * (float(dm.x) * float(sum_d) - float(dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // bitwise OR to add 4 if hmask is set, subtract later
    const i8vec2 vals00 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2    ] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals01 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 1] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 1] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals10 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 2] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 2] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals11 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 3] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 3] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals20 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 4] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 4] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals21 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 5] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 5] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals30 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 6] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 6] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals31 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 7] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 7] >> hm_shift) & uint16_t(0x0101)) << 2));

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y) - int8_t(4)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y) - int8_t(4)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y) - int8_t(4)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y) - int8_t(4)));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const int8_t scale = int8_t(((data_a[ib_k].scales[is % 8      ] >> (4 * (is / 8))) & 0x0F0F) |
                               (((data_a[ib_k].scales[8 + (is % 4)] >> (2 * (is / 4))) & 0x0303) << 4));
    return float(data_a[ib_k].d) * float(scale - 32);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_d) * d_scale * float(q_sum));
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F;

    return i32vec4(vals0, vals1, vals2, vals3);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs;
    const uint qh_shift = iqs_k / 8;

    return i32vec4(((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx    ] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 1] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 2] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 3] >> qh_shift) & 0x01010101) << 4));
#endif
}

vec2 get_dm_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    const uvec3 scales = uvec3(data_a_packed32[ib_k].scales[0],
                               data_a_packed32[ib_k].scales[1],
                               data_a_packed32[ib_k].scales[2]);
    const uint scalesoffs = (is & 3) * 8;

    const uint scidx0 = (is < 4) ? 0 : 2;
    const uint scidxshift0 = scalesoffs;
    const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
    const uint mbidx0 = (is < 4) ? 1 : 2;
    const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
    const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

    const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
    const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
    u8vec2 scale_dm = u8vec2(sc, mbyte);

    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm) * FLOAT_TYPEV2(scale_dm);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const vec2 dm_scale = get_dm_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    // Exact integer sum of this thread's 16 activation values. Each thread
    // covers 16 of the window's 32 values and the reduction sums both
    // threads, so the total min term equals the CPU `min_g * (bsums[2g] +
    // bsums[2g+1])` exactly: no /2, no f16.
    int32_t sum16 = dotPacked4x8EXT(int32_t(0x01010101), cache_b_qs[0]);
    sum16 += dotPacked4x8EXT(int32_t(0x01010101), cache_b_qs[1]);
    sum16 += dotPacked4x8EXT(int32_t(0x01010101), cache_b_qs[2]);
    sum16 += dotPacked4x8EXT(int32_t(0x01010101), cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_d) * (float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(sum16)));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals10 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 2] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 2] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals11 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 3] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 3] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals20 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 4] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 4] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals21 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 5] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 5] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals30 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 6] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 6] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals31 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 7] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 7] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y)));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[iqs_k / 4]);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_d) * float(d_scale) * float(q_sum));
}
#endif
