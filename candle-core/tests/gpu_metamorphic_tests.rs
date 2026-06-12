mod support;

use candle_core::{DType, Result, Tensor};
use support::{assert_tensors_close, native_required, TestBackend};

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
fn gpu_metamorphic_wgpu() -> Result<()> {
    native_required(
        "gpu_metamorphic_wgpu",
        TestBackend::Wgpu,
        run_metamorphic_suite,
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_metamorphic_vulkan() -> Result<()> {
    native_required(
        "gpu_metamorphic_vulkan",
        TestBackend::Vulkan,
        run_metamorphic_suite,
    )
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_metamorphic_suite(device: &candle_core::Device) -> Result<()> {
    let cpu = candle_core::Device::Cpu;
    let xs_vals = support::deterministic_f32_data(24, 1337);
    let xs_cpu = Tensor::from_vec(xs_vals.clone(), (2, 3, 4), &cpu)?;
    let xs_dev = Tensor::from_vec(xs_vals, (2, 3, 4), device)?;

    let zeros_cpu = Tensor::zeros((2, 3, 4), DType::F32, &cpu)?;
    let zeros_dev = Tensor::zeros((2, 3, 4), DType::F32, device)?;
    assert_tensors_close(
        &xs_dev.add(&zeros_dev)?,
        &xs_cpu.add(&zeros_cpu)?,
        DType::F32,
        "metamorphic x_plus_zero",
    )?;

    let ones_cpu = Tensor::ones((2, 3, 4), DType::F32, &cpu)?;
    let ones_dev = Tensor::ones((2, 3, 4), DType::F32, device)?;
    assert_tensors_close(
        &xs_dev.mul(&ones_dev)?,
        &xs_cpu.mul(&ones_cpu)?,
        DType::F32,
        "metamorphic x_times_one",
    )?;

    let zeros_expected = Tensor::zeros((2, 3, 4), DType::F32, &cpu)?;
    assert_tensors_close(
        &xs_dev.sub(&xs_dev)?,
        &zeros_expected,
        DType::F32,
        "metamorphic x_minus_x",
    )?;

    assert_tensors_close(
        &xs_dev.transpose(1, 2)?.transpose(1, 2)?,
        &xs_cpu,
        DType::F32,
        "metamorphic transpose_transpose",
    )?;

    assert_tensors_close(
        &xs_dev.reshape((6, 4))?.flatten_all()?,
        &xs_cpu.flatten_all()?,
        DType::F32,
        "metamorphic reshape_flatten",
    )?;

    let addend_cpu = Tensor::full(17.5f32, (2, 3, 4), &cpu)?;
    let addend_dev = Tensor::full(17.5f32, (2, 3, 4), device)?;
    let base_argmax = xs_cpu.argmax_keepdim(2)?.flatten_all()?.to_vec1::<u32>()?;
    let shifted_argmax = xs_dev
        .add(&addend_dev)?
        .argmax_keepdim(2)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    assert_eq!(shifted_argmax, base_argmax, "metamorphic argmax_shift");
    assert_tensors_close(
        &xs_dev.add(&addend_dev)?,
        &xs_cpu.add(&addend_cpu)?,
        DType::F32,
        "metamorphic additive_shift",
    )?;

    let lhs_vals = support::deterministic_f32_data(12, 21);
    let lhs_cpu = Tensor::from_vec(lhs_vals.clone(), (3, 4), &cpu)?;
    let lhs_dev = Tensor::from_vec(lhs_vals, (3, 4), device)?;
    let eye_cpu = Tensor::eye(4, DType::F32, &cpu)?;
    let eye_dev = Tensor::eye(4, DType::F32, device)?;
    assert_tensors_close(
        &lhs_dev.matmul(&eye_dev)?,
        &lhs_cpu.matmul(&eye_cpu)?,
        DType::F32,
        "metamorphic matmul_identity",
    )?;

    Ok(())
}
