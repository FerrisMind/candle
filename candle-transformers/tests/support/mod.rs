#![allow(dead_code)]

use candle::{DType, Device, Result, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestBackend {
    Cuda,
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
        TestBackend::Cuda => "cuda",
        TestBackend::Wgpu => "wgpu",
        TestBackend::Vulkan => "vulkan",
    }
}

pub fn backend_device_or_skip(test_name: &str, backend: TestBackend) -> Result<Option<Device>> {
    let require_device = std::env::var_os(required_device_env_var(backend)).is_some();
    match backend_device(backend) {
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

pub fn native_required<F>(test_name: &str, backend: TestBackend, body: F) -> Result<()>
where
    F: FnOnce(&Device) -> Result<()>,
{
    run_backend_case(test_name, backend, FallbackPolicy::NativeRequired, body)
}

pub fn fallback_allowed<F>(test_name: &str, backend: TestBackend, body: F) -> Result<()>
where
    F: FnOnce(&Device) -> Result<()>,
{
    run_backend_case(test_name, backend, FallbackPolicy::FallbackAllowed, body)
}

pub fn backend_fallback_count(backend: TestBackend) -> usize {
    match backend {
        TestBackend::Cuda => 0,
        TestBackend::Wgpu => candle::wgpu_cpu_fallback_count(),
        TestBackend::Vulkan => candle::vulkan_cpu_fallback_count(),
    }
}

pub fn reset_backend_fallback_count(backend: TestBackend) {
    match backend {
        TestBackend::Cuda => {}
        TestBackend::Wgpu => candle::reset_wgpu_cpu_fallback_count(),
        TestBackend::Vulkan => candle::reset_vulkan_cpu_fallback_count(),
    }
}

pub fn assert_close_tensors(
    actual: &Tensor,
    expected: &Tensor,
    atol: f32,
    rtol: f32,
    label: &str,
) -> Result<()> {
    let actual = actual.to_dtype(DType::F32)?;
    let expected = expected.to_dtype(DType::F32)?;
    if actual.dims() != expected.dims() {
        candle::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            actual.dims(),
            expected.dims()
        );
    }
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tol = atol + rtol * actual.abs().max(expected.abs());
        if diff > tol {
            candle::bail!(
                "{label}: mismatch at idx {idx}: got {actual}, expected {expected}, diff {diff}, tol {tol}"
            );
        }
    }
    Ok(())
}

pub fn deterministic_f32_data(len: usize, seed: u64) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let mixed = (idx as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_mul(1442695040888963407));
            let bucket = ((mixed >> 17) % 257) as i64 - 128;
            bucket as f32 / 64.0
        })
        .collect()
}

pub fn mean_pool(sequence_output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let masked = sequence_output.broadcast_mul(&mask)?;
    let sum = masked.sum(1)?;
    let denom = mask.sum(1)?;
    sum.broadcast_div(&denom)
}

pub fn download_model_artifact(model_id: &str, revision: &str, filename: &str) -> Result<PathBuf> {
    let api = Api::new().map_err(|err| {
        candle::Error::msg(format!(
            "failed to create hf-hub client for {model_id}: {err}"
        ))
    })?;
    let repo = Repo::with_revision(model_id.to_owned(), RepoType::Model, revision.to_owned());
    api.repo(repo).get(filename).map_err(|err| {
        candle::Error::msg(format!(
            "failed to download {filename} from {model_id}@{revision}: {err}"
        ))
    })
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

fn backend_device(backend: TestBackend) -> Result<Device> {
    match backend {
        TestBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle::bail!("cuda backend feature not enabled")
            }
        }
        TestBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Device::new_wgpu(0)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                candle::bail!("wgpu backend feature not enabled")
            }
        }
        TestBackend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Device::new_vulkan(0)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                candle::bail!("vulkan backend feature not enabled")
            }
        }
    }
}

fn required_device_env_var(backend: TestBackend) -> &'static str {
    match backend {
        TestBackend::Cuda => "CANDLE_REQUIRE_CUDA_TEST_DEVICE",
        TestBackend::Wgpu => "CANDLE_REQUIRE_WGPU_TEST_DEVICE",
        TestBackend::Vulkan => "CANDLE_REQUIRE_VULKAN_TEST_DEVICE",
    }
}
