#ifndef CANDLE_Q8K_LAYOUT_GLSL
#define CANDLE_Q8K_LAYOUT_GLSL

// Packed Q8K activation block (256 elements) for the fused k-quant dp4a
// kernels. Mirrors the CPU `BlockQ8K` contract that `QMatMul` uses for
// Q2_K..Q6_K: ONE f32 scale per 256-element block taken from the signed
// extreme (`iscale = -127/max`, `d = 1/iscale`), int8 quantized values, and
// the dequantized value is `d * q`. The per-32 f16 `block_q8_1` grid cannot
// represent this (f16 scale + f16 integer sum), which is why the fused q8_1
// kernels are numerically wrong for the k-quants.
//
// std430 layout, 16-byte aligned with no padding (272 bytes per block):
//   offset 0:   float  d
//   offset 16:  int8   qs[256]  (viewed as ivec4[16])
//
// The per-16-element integer sums (`bsums` in the CPU block) are NOT stored:
// the min correction of the Q4_K/Q5_K dot products needs only the sum over a
// 32-element window, which the kernels derive from the int8 values with
// `dotPacked4x8EXT(0x01010101, qs)` (exact integer, same as the CPU bsums).
struct block_q8_k {
    float d;
    ivec4 qs[16];
};

#define QUANT_K_Q8_K 256
#define Q8_K_BYTES 272

#endif
