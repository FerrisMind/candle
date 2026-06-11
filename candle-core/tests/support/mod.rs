#![allow(dead_code)]

use candle_core::{DType, Device, Result, Tensor};
use std::sync::{Mutex, MutexGuard};

/// The CPU-fallback counters asserted by `native_required` are process-global,
/// so any test that reads or resets them must hold this lock for its full
/// duration. Otherwise a concurrently running `fallback_allowed` test can
/// increment the counter between another test's reset and assertion, causing
/// spurious `native_required` failures under the default parallel test runner.
static FALLBACK_COUNTER_LOCK: Mutex<()> = Mutex::new(());

pub fn fallback_counter_guard() -> MutexGuard<'static, ()> {
    FALLBACK_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestBackend {
    Wgpu,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackPolicy {
    NativeRequired,
    FallbackAllowed,
}

pub fn backend_name(backend: TestBackend) -> &'static str {
    match backend {
        TestBackend::Wgpu => "wgpu",
        TestBackend::Vulkan => "vulkan",
    }
}

pub fn backend_device_or_skip(test_name: &str, backend: TestBackend) -> Result<Option<Device>> {
    let require_device = std::env::var_os(required_device_env_var(backend)).is_some();
    let device = match backend {
        TestBackend::Wgpu => Device::new_wgpu(0),
        TestBackend::Vulkan => Device::new_vulkan(0),
    };
    match device {
        Ok(device) => Ok(Some(device)),
        Err(err) if require_device => Err(err),
        Err(err) => {
            eprintln!(
                "skipping {test_name}: {} device unavailable: {err}",
                backend_name(backend)
            );
            Ok(None)
        }
    }
}

pub fn fallback_allowed<F>(test_name: &str, backend: TestBackend, body: F) -> Result<()>
where
    F: FnOnce(&Device) -> Result<()>,
{
    run_backend_case(test_name, backend, FallbackPolicy::FallbackAllowed, body)
}

pub fn native_required<F>(test_name: &str, backend: TestBackend, body: F) -> Result<()>
where
    F: FnOnce(&Device) -> Result<()>,
{
    run_backend_case(test_name, backend, FallbackPolicy::NativeRequired, body)
}

pub fn backend_fallback_count(backend: TestBackend) -> usize {
    match backend {
        TestBackend::Wgpu => candle_core::wgpu_cpu_fallback_count(),
        TestBackend::Vulkan => candle_core::vulkan_cpu_fallback_count(),
    }
}

pub fn reset_backend_fallback_count(backend: TestBackend) {
    match backend {
        TestBackend::Wgpu => candle_core::reset_wgpu_cpu_fallback_count(),
        TestBackend::Vulkan => candle_core::reset_vulkan_cpu_fallback_count(),
    }
}

pub fn deterministic_f32_data(len: usize, seed: u64) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let mixed = (idx as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_mul(1442695040888963407));
            let bucket = ((mixed >> 17) % 257) as i64 - 128;
            bucket as f32 / 19.0
        })
        .collect()
}

pub fn shape_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn deterministic_f32_tensor(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        deterministic_f32_data(shape_len(shape), seed),
        shape.to_vec(),
        device,
    )
}

pub fn assert_close_vec(actual: &[f32], expected: &[f32], dtype: DType, label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        candle_core::bail!(
            "{label}: length mismatch, got {}, expected {}",
            actual.len(),
            expected.len()
        );
    }
    let (atol, rtol) = tolerance(dtype);
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tol = atol + rtol * actual.abs().max(expected.abs());
        if diff > tol {
            candle_core::bail!(
                "{label}: mismatch at idx {idx}: got {actual}, expected {expected}, diff {diff}, tol {tol}"
            );
        }
    }
    Ok(())
}

pub fn assert_tensors_close(
    actual: &Tensor,
    expected: &Tensor,
    dtype: DType,
    label: &str,
) -> Result<()> {
    let actual = actual.to_dtype(DType::F32)?;
    let expected = expected.to_dtype(DType::F32)?;
    if actual.dims() != expected.dims() {
        candle_core::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            actual.dims(),
            expected.dims()
        );
    }
    let actual_vals = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected_vals = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_close_vec(&actual_vals, &expected_vals, dtype, label)
}

fn run_backend_case<F>(
    test_name: &str,
    backend: TestBackend,
    policy: FallbackPolicy,
    body: F,
) -> Result<()>
where
    F: FnOnce(&Device) -> Result<()>,
{
    // Every backend case serializes on the global fallback-counter lock so
    // `native_required` assertions cannot observe fallbacks recorded by other
    // concurrently running tests.
    let _guard = fallback_counter_guard();
    let Some(device) = backend_device_or_skip(test_name, backend)? else {
        return Ok(());
    };
    reset_backend_fallback_count(backend);
    body(&device)?;
    device.synchronize()?;
    if policy == FallbackPolicy::NativeRequired {
        let fallback_count = backend_fallback_count(backend);
        assert_eq!(
            fallback_count,
            0,
            "{test_name}: {} path triggered {} CPU fallbacks",
            backend_name(backend),
            fallback_count
        );
    }
    Ok(())
}

fn required_device_env_var(backend: TestBackend) -> &'static str {
    match backend {
        TestBackend::Wgpu => "CANDLE_REQUIRE_WGPU_TEST_DEVICE",
        TestBackend::Vulkan => "CANDLE_REQUIRE_VULKAN_TEST_DEVICE",
    }
}

fn tolerance(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::F16 => (1e-2, 1e-3),
        DType::BF16 => (1e-2, 1e-2),
        _ => (1e-4, 1e-4),
    }
}
