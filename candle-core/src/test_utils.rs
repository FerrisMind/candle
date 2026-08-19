use crate::{Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?)
        }
    };
}

pub fn assert_tensor_eq(t1: &Tensor, t2: &Tensor) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    // Default U8 may not be large enough to hold the sum (`t.sum_all` defaults to the dtype of `t`)
    let eq_tensor = t1.eq(t2)?.to_dtype(crate::DType::U32)?;
    let all_equal = eq_tensor.sum_all()?;
    assert_eq!(all_equal.to_scalar::<u32>()?, eq_tensor.elem_count() as u32);
    Ok(())
}

pub fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

pub fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

pub fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

pub fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

// ---------------------------------------------------------------------------
// Differential parity test helpers (additive to existing test_utils)
// ---------------------------------------------------------------------------

/// Per-dtype tolerance pair (absolute, relative) for CPU-vs-GPU comparison.
/// Integer types require exact match (0.0, 0.0).
pub fn diff_tolerance(dtype: crate::DType) -> (f64, f64) {
    match dtype {
        crate::DType::F64 => (1e-6, 1e-5),
        crate::DType::F32 => (1e-4, 1e-4),
        crate::DType::F16 => (1e-2, 1e-3),
        crate::DType::BF16 => (2e-2, 1e-2),
        crate::DType::F8E4M3 => (1e-1, 5e-2),
        // Integer types: exact
        crate::DType::U8
        | crate::DType::U32
        | crate::DType::I16
        | crate::DType::I32
        | crate::DType::I64 => (0.0, 0.0),
        _ => (1e-3, 1e-3),
    }
}

/// Compute ULP difference between two f32 values.
/// Returns 0 when both are NaN, u64::MAX/4 for sign mismatches or infinities.
pub fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return if a.to_bits() == b.to_bits() {
            0
        } else {
            u64::MAX / 4
        };
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    let ai = if ai < 0 {
        0x8000_0000u32 as i32 - ai
    } else {
        ai
    };
    let bi = if bi < 0 {
        0x8000_0000u32 as i32 - bi
    } else {
        bi
    };
    (ai as i64 - bi as i64).unsigned_abs()
}

/// Compare two f32 slices element-wise. Returns (max_abs_err, max_rel_err, max_ulp, first_mismatch_idx).
pub fn compare_f32_slices(got: &[f32], expected: &[f32]) -> (f64, f64, u64, Option<usize>) {
    let n = got.len().min(expected.len());
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_ulp = 0u64;
    let mut first_bad = None;
    for i in 0..n {
        let g = got[i];
        let e = expected[i];
        if g.is_nan() && e.is_nan() {
            continue;
        }
        if g.is_nan() || e.is_nan() || g.is_infinite() || e.is_infinite() {
            if g.to_bits() != e.to_bits() {
                max_abs = max_abs.max(1e30);
                if first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
            continue;
        }
        let abs = (g as f64 - e as f64).abs();
        let rel = abs / (1e-30 + e.abs() as f64);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        let ulp = ulp_diff_f32(g, e);
        max_ulp = max_ulp.max(ulp);
        if first_bad.is_none() && (abs > 0.0 || ulp > 0) {
            first_bad = Some(i);
        }
    }
    (max_abs, max_rel, max_ulp, first_bad)
}

/// Compare two f64 slices element-wise. Returns (max_abs_err, max_rel_err, first_mismatch_idx).
pub fn compare_f64_slices(got: &[f64], expected: &[f64]) -> (f64, f64, Option<usize>) {
    let n = got.len().min(expected.len());
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut first_bad = None;
    for i in 0..n {
        let g = got[i];
        let e = expected[i];
        if g.is_nan() && e.is_nan() {
            continue;
        }
        if g.is_nan() || e.is_nan() || g.is_infinite() || e.is_infinite() {
            if g.to_bits() != e.to_bits() {
                max_abs = max_abs.max(1e30);
                if first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
            continue;
        }
        let abs = (g - e).abs();
        let rel = abs / (1e-30 + e.abs());
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if first_bad.is_none() && abs > 0.0 {
            first_bad = Some(i);
        }
    }
    (max_abs, max_rel, first_bad)
}

/// Compare two integer slices element-wise. Returns (first_mismatch_idx, got_val, expected_val).
pub fn compare_int_slices<T: PartialEq + Copy + std::fmt::Debug>(
    got: &[T],
    expected: &[T],
) -> Option<(usize, T, T)> {
    for i in 0..got.len().min(expected.len()) {
        if got[i] != expected[i] {
            return Some((i, got[i], expected[i]));
        }
    }
    None
}

/// Check if a dtype is integer (exact comparison required).
pub fn is_integer_dtype(dtype: crate::DType) -> bool {
    matches!(
        dtype,
        crate::DType::U8
            | crate::DType::U32
            | crate::DType::I16
            | crate::DType::I32
            | crate::DType::I64
    )
}
