//! Regression tests for Vulkan gather/index_select dispatch limits.
//!
//! `get_rows.comp` and `get_rows_quant.comp` grid-stride over the Y and Z
//! workgroup dimensions, but the dispatch previously passed `ids_len` and the
//! product of the remaining (left) dimensions straight through. Vulkan only
//! guarantees 65535 workgroups per dimension, so a gather with 65536 indices
//! (e.g. a 256x256 deformable-conv sampling grid) or 65536 left slices silently
//! left part of the output unwritten.
#![cfg(feature = "vulkan")]

use candle_core::{Device, Result, Tensor};
use std::sync::{Mutex, MutexGuard};

/// Tests in this file create separate Vulkan devices; serializing them avoids
/// exercising the driver's multi-device paths, which are unrelated to the
/// dispatch-limit regression under test.
fn lock_vulkan() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn vulkan_device() -> Result<Device> {
    Device::new_vulkan(0)
}

fn assert_exact_f32(got: &Tensor, expected: &Tensor) -> Result<()> {
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        if got.to_bits() != expected.to_bits() {
            candle_core::bail!(
                "gather mismatch at {i}: got {got} ({:#010x}), expected {expected} ({:#010x})",
                got.to_bits(),
                expected.to_bits()
            );
        }
    }
    Ok(())
}

fn assert_close_f32(got: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        if (got - expected).abs() > tolerance {
            candle_core::bail!(
                "gather mismatch at {i}: got {got}, expected {expected} (tolerance {tolerance})"
            );
        }
    }
    Ok(())
}

/// `ids_len = 65536` exercises the Y workgroup dimension.
#[test]
fn vulkan_gather_ids_len_above_workgroup_limit_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let len = 65536usize;
    let values: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let indices: Vec<u32> = (0..len).map(|i| ((i * 7919 + 13) % len) as u32).collect();
    let x = Tensor::from_vec(values, (1, 1, len), &device)?;
    let ids = Tensor::from_vec(indices, (1, 1, len), &device)?;
    let got = x.gather(&ids, 2)?;

    let x_cpu = x.to_device(&Device::Cpu)?;
    let ids_cpu = ids.to_device(&Device::Cpu)?;
    let expected = x_cpu.gather(&ids_cpu, 2)?;
    assert_exact_f32(&got, &expected)
}

/// `left_size = c * ids_len = 65536` exercises the Z workgroup dimension with
/// an otherwise small Y dispatch, matching the deformable-conv row gather.
#[test]
fn vulkan_gather_left_size_above_workgroup_limit_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let (channels, len, ids_len) = (64usize, 65536usize, 1024usize);
    let values: Vec<f32> = (0..channels * len)
        .map(|i| ((i * 13 + 7) % 1009) as f32)
        .collect();
    let indices: Vec<u32> = (0..ids_len).map(|i| ((i * 31 + 5) % len) as u32).collect();
    let x = Tensor::from_vec(values, (1, channels, len), &device)?;
    let ids = Tensor::from_vec(indices, (1, 1, ids_len), &device)?
        .broadcast_as((1, channels, ids_len))?
        .contiguous()?;
    let got = x.gather(&ids, 2)?;

    let x_cpu = x.to_device(&Device::Cpu)?;
    let ids_cpu = ids.to_device(&Device::Cpu)?;
    let expected = x_cpu.gather(&ids_cpu, 2)?;
    assert_exact_f32(&got, &expected)
}

/// Non-last-dim `index_select` uses the same `get_rows` shader and can hit the
/// limit on either Y (`ids_len`) or Z (`left_size`).
#[test]
fn vulkan_index_select_above_workgroup_limits_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let total = 65536usize;
    let values: Vec<f32> = (0..total * 2)
        .map(|i| ((i * 17 + 3) % 997) as f32)
        .collect();
    let x = Tensor::from_vec(values, (total, 2), &device)?;

    // Large Y: 65536 selected rows, each with 2 columns.
    let row_ids: Vec<u32> = (0..total)
        .map(|i| ((i * 7919 + 13) % total) as u32)
        .collect();
    let row_ids = Tensor::from_vec(row_ids, total, &device)?;
    let got = x.index_select(&row_ids, 0)?;
    let x_cpu = x.to_device(&Device::Cpu)?;
    let row_ids_cpu = row_ids.to_device(&Device::Cpu)?;
    let expected = x_cpu.index_select(&row_ids_cpu, 0)?;
    assert_exact_f32(&got, &expected)?;

    // Large Z: one selected column, so `left_size` is 65536.
    let col_ids = Tensor::from_vec(vec![1u32], 1, &device)?;
    let got = x.index_select(&col_ids, 1)?;
    let col_ids_cpu = col_ids.to_device(&Device::Cpu)?;
    let expected = x_cpu.index_select(&col_ids_cpu, 1)?;
    assert_exact_f32(&got, &expected)
}

/// Quantized `index_select` has a separate shader with the same Y/Z dispatch
/// scheme. `QTensor::embedding` is the public path to its row gather.
#[test]
fn vulkan_quantized_index_select_above_workgroup_limit_matches_cpu() -> Result<()> {
    let _guard = lock_vulkan();
    let device = match vulkan_device() {
        Ok(device) => device,
        Err(err) => {
            eprintln!("vulkan unavailable: {err}; skipping");
            return Ok(());
        }
    };
    let total = 65536usize;
    let hidden = 32usize;
    let values: Vec<f32> = (0..total * hidden)
        .map(|i| ((i * 17 + 3) % 997) as f32)
        .collect();
    let x = Tensor::from_vec(values, (total, hidden), &device)?;
    let quantized =
        candle_core::quantized::QTensor::quantize(&x, candle_core::quantized::GgmlDType::Q8_0)?;
    let row_ids: Vec<u32> = (0..total)
        .map(|i| ((i * 7919 + 13) % total) as u32)
        .collect();
    let row_ids = Tensor::from_vec(row_ids, total, &device)?;
    let got = quantized.embedding(&row_ids)?;

    // Dequantizing the same QTensor on the CPU isolates the row-gather
    // dispatch from any GPU/CPU quantization-kernel differences. The Vulkan
    // bulk-dequant path materializes q8_0 through an f16 buffer, so compare
    // with the half-ULP tolerance of the source value range (< 1024 -> 0.25)
    // instead of bit equality.
    let expected = quantized
        .dequantize(&Device::Cpu)?
        .index_select(&row_ids.to_device(&Device::Cpu)?, 0)?;
    assert_close_f32(&got, &expected, 0.5)
}
