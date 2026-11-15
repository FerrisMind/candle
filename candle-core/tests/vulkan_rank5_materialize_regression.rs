//! Regression tests for the rank>4 materialization paths on Vulkan.
//!
//! `materialize_rank_gt4_compact` used to slice a rank>4 tensor per leading
//! index even when the layout was already contiguous, so casting a stored F16
//! `[O, I, 3, 3, 3]` conv weight to F32 issued `O*I` copies of 54 bytes each.
//! The contiguous shortcut replaces that with one bulk copy. These tests pin
//! both paths (contiguous shortcut and strided per-prefix fallback) to the CPU
//! result, plus the small host-readback path that now stays on the compute
//! queue.
#![cfg(feature = "vulkan")]

use candle_core::{DType, Device, Result, Tensor};
use std::sync::{Mutex, MutexGuard};

fn lock_vulkan() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn vulkan_device() -> Result<Device> {
    Device::new_vulkan(0)
}

fn pattern(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 13 + 7) % 251) as f32 * 0.125 - 15.0)
        .collect()
}

fn assert_bit_equal(got: &Tensor, expected: &Tensor) -> Result<()> {
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        if got.to_bits() != expected.to_bits() {
            candle_core::bail!(
                "rank>4 materialize mismatch at {i}: got {got} ({:#010x}), expected {expected} ({:#010x})",
                got.to_bits(),
                expected.to_bits()
            );
        }
    }
    Ok(())
}

/// Contiguous rank-5 F16 -> F32 cast: one bulk copy, bit-exact result.
#[test]
fn vulkan_rank5_contiguous_f16_cast_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let shape = (8usize, 16usize, 3usize, 3usize, 3usize);
    let elems = shape.0 * shape.1 * shape.2 * shape.3 * shape.4;
    let cpu_f16 = Tensor::from_vec(pattern(elems), shape, &Device::Cpu)?.to_dtype(DType::F16)?;
    let expected = cpu_f16.to_dtype(DType::F32)?;

    let got = cpu_f16.to_device(&device)?.to_dtype(DType::F32)?;
    assert_eq!(got.dims(), expected.dims());
    assert_bit_equal(&got, &expected)
}

/// A start-offset-0 but strided rank-5 view takes the per-prefix fallback.
/// It must produce the same values as the contiguous shortcut.
#[test]
fn vulkan_rank5_strided_view_f16_cast_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let base_shape = (4usize, 8usize, 4usize, 3usize, 3usize);
    let base_elems = base_shape.0 * base_shape.1 * base_shape.2 * base_shape.3 * base_shape.4;
    let cpu_base = Tensor::from_vec(pattern(base_elems), base_shape, &Device::Cpu)?;
    // `narrow` on the middle dim keeps the original strides and start offset 0,
    // so the view is non-contiguous but not offset.
    let cpu_view = cpu_base.narrow(2, 0, 3)?;
    assert!(!cpu_view.is_contiguous());
    let cpu_f16 = cpu_view.to_dtype(DType::F16)?;
    let expected = cpu_f16.to_dtype(DType::F32)?;

    let got = cpu_f16.to_device(&device)?.to_dtype(DType::F32)?;
    assert_eq!(got.dims(), expected.dims());
    assert_bit_equal(&got, &expected)
}

/// Sub-megabyte device -> host readback: the copy stays on the compute queue
/// and `read_buffer` must still flush both queues before mapping.
#[test]
fn vulkan_small_readback_roundtrip() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let values = pattern(4096);
    let x = Tensor::from_vec(values.clone(), 4096, &device)?;
    // Force a device-side result so the readback is not just an identity map.
    let y = (&x + 1.0)?;
    let got = y.to_vec1::<f32>()?;
    assert_eq!(got.len(), values.len());
    for (i, (got, value)) in got.iter().zip(values.iter()).enumerate() {
        assert_eq!(*got, value + 1.0, "readback mismatch at {i}");
    }
    Ok(())
}
