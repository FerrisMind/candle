use super::Cpu;
use super::{CpuBF16, CpuF16};
use core::arch::wasm32::*;
use half::{bf16, f16};

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl Cpu for CurrentCpu {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        v128_load(mem_addr as *mut v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        v128_store(mem_addr as *mut v128, a);
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        for i in 0..ARR / 2 {
            x[2 * i] = f32x4_add(x[2 * i], x[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            x[4 * i] = f32x4_add(x[4 * i], x[4 * i + 2]);
        }
        for i in 0..ARR / 8 {
            x[8 * i] = f32x4_add(x[8 * i], x[8 * i + 4]);
        }
        *y = f32x4_extract_lane::<0>(x[0])
            + f32x4_extract_lane::<1>(x[0])
            + f32x4_extract_lane::<2>(x[0])
            + f32x4_extract_lane::<3>(x[0]);
    }
}

/// wasm32+simd128 f16 path.
///
/// wasm SIMD128 gives us fast f32 lanes, so we widen f16→f32 on load,
/// execute vector ops in f32, and narrow f32→f16 on store.
pub struct CurrentCpuF16 {}

impl CpuF16 for CurrentCpuF16 {
    // Keep the same geometry as `CurrentCpu` for simplicity:
    // 4 lanes × 4-bit values => 4 f16 per v128.
    type Unit = v128;
    type Array = [v128; 4];

    const STEP: usize = 16;
    const EPR: usize = 4;
    const ARR: usize = 4;

    // No need to periodically flush because we always operate in f32.
    const FLUSH_INTERVAL: usize = usize::MAX;

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); Self::ARR]
    }

    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        // Widen 4 f16 values into 4 f32 lanes.
        let mut tmp = [0f32; 4];
        for i in 0..4 {
            tmp[i] = (*mem_addr.add(i)).to_f32();
        }
        v128_load(tmp.as_ptr() as *const v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // a + b*c (lane-wise)
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
        // Sum all lanes across all v128 blocks into a single scalar.
        let mut sum = 0f32;
        for i in 0..Self::ARR {
            sum += f32x4_extract_lane::<0>(x[i])
                + f32x4_extract_lane::<1>(x[i])
                + f32x4_extract_lane::<2>(x[i])
                + f32x4_extract_lane::<3>(x[i]);
        }
        *y = sum;
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        // Narrow 4 f32 lanes back to 4 f16 values.
        *mem_addr.add(0) = f16::from_f32(f32x4_extract_lane::<0>(a));
        *mem_addr.add(1) = f16::from_f32(f32x4_extract_lane::<1>(a));
        *mem_addr.add(2) = f16::from_f32(f32x4_extract_lane::<2>(a));
        *mem_addr.add(3) = f16::from_f32(f32x4_extract_lane::<3>(a));
    }
}

/// wasm32+simd128 bf16 path.
///
/// Same strategy as f16: widen bf16→f32 on load, compute in f32 lanes,
/// narrow f32→bf16 on store.
pub struct CurrentCpuBF16 {}

impl CpuBF16 for CurrentCpuBF16 {
    type Unit = v128;
    type Array = [v128; 4];

    const STEP: usize = 16;
    const EPR: usize = 4;
    const ARR: usize = 4;

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); Self::ARR]
    }

    unsafe fn load(mem_addr: *const bf16) -> Self::Unit {
        let mut tmp = [0f32; 4];
        for i in 0..4 {
            tmp[i] = (*mem_addr.add(i)).to_f32();
        }
        v128_load(tmp.as_ptr() as *const v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
        let mut sum = 0f32;
        for i in 0..Self::ARR {
            sum += f32x4_extract_lane::<0>(x[i])
                + f32x4_extract_lane::<1>(x[i])
                + f32x4_extract_lane::<2>(x[i])
                + f32x4_extract_lane::<3>(x[i]);
        }
        *y = sum;
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit) {
        *mem_addr.add(0) = bf16::from_f32(f32x4_extract_lane::<0>(a));
        *mem_addr.add(1) = bf16::from_f32(f32x4_extract_lane::<1>(a));
        *mem_addr.add(2) = bf16::from_f32(f32x4_extract_lane::<2>(a));
        *mem_addr.add(3) = bf16::from_f32(f32x4_extract_lane::<3>(a));
    }
}
