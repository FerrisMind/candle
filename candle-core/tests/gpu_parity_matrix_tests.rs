//! Structured CUDA-parity matrix: op × dtype × rank × layout with
//! `native_required` and zero CPU-fallback count.

mod support;

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use support::{assert_tensors_close, deterministic_f32_tensor, native_required, TestBackend};

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_unary_dtype_matrix(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let vals = vec![0.0f32, 0.25, 1.0, -0.5, 2.0, -1.25];
    let shape = (2, 3);

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let dev_x = Tensor::from_vec(vals.clone(), shape, device)?.to_dtype(dtype)?;
        let cpu_x = Tensor::from_vec(vals.clone(), shape, &cpu)?.to_dtype(dtype)?;

        assert_tensors_close(&dev_x.erf()?, &cpu_x.erf()?, dtype, &format!("erf {dtype:?}"))?;
        assert_tensors_close(
            &dev_x.recip()?,
            &cpu_x.recip()?,
            dtype,
            &format!("recip {dtype:?}"),
        )?;
        assert_tensors_close(&dev_x.neg()?, &cpu_x.neg()?, dtype, &format!("neg {dtype:?}"))?;
        assert_tensors_close(&dev_x.abs()?, &cpu_x.abs()?, dtype, &format!("abs {dtype:?}"))?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_binary_dtype_matrix(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
    let shape = (2, 3);

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let dev_a = Tensor::from_vec(a_vals.clone(), shape, device)?.to_dtype(dtype)?;
        let dev_b = Tensor::from_vec(b_vals.clone(), shape, device)?.to_dtype(dtype)?;
        let cpu_a = Tensor::from_vec(a_vals.clone(), shape, &cpu)?.to_dtype(dtype)?;
        let cpu_b = Tensor::from_vec(b_vals.clone(), shape, &cpu)?.to_dtype(dtype)?;

        assert_tensors_close(
            &dev_a.add(&dev_b)?,
            &cpu_a.add(&cpu_b)?,
            dtype,
            &format!("add {dtype:?}"),
        )?;
        assert_tensors_close(
            &dev_a.sub(&dev_b)?,
            &cpu_a.sub(&cpu_b)?,
            dtype,
            &format!("sub {dtype:?}"),
        )?;
        assert_tensors_close(
            &dev_a.mul(&dev_b)?,
            &cpu_a.mul(&cpu_b)?,
            dtype,
            &format!("mul {dtype:?}"),
        )?;
        assert_tensors_close(
            &dev_a.div(&dev_b)?,
            &cpu_a.div(&cpu_b)?,
            dtype,
            &format!("div {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_rank_layout_matrix(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;

    let scalar = Tensor::from_slice(&[3.5f32], (1,), device)?;
    let scalar_cpu = Tensor::from_slice(&[3.5f32], (1,), &cpu)?;
    assert_tensors_close(
        &scalar.sum_all()?,
        &scalar_cpu.sum_all()?,
        DType::F32,
        "scalar sum_all",
    )?;

    let dev = deterministic_f32_tensor(&[1, 2, 3, 2, 2], 7, device)?
        .transpose(0, 4)?
        .to_dtype(DType::BF16)?;
    let reference = deterministic_f32_tensor(&[1, 2, 3, 2, 2], 7, &cpu)?
        .transpose(0, 4)?
        .to_dtype(DType::BF16)?;
    assert_tensors_close(
        &dev.relu()?,
        &reference.relu()?,
        DType::BF16,
        "rank5 strided bf16 relu",
    )?;

    let u8_a = Tensor::from_vec((0..24u8).collect::<Vec<_>>(), (2, 3, 4), device)?
        .transpose(0, 2)?;
    let u8_b = Tensor::from_vec((1..=24u8).collect::<Vec<_>>(), (2, 3, 4), device)?
        .transpose(0, 2)?;
    let u8_a_cpu = Tensor::from_vec((0..24u8).collect::<Vec<_>>(), (2, 3, 4), &cpu)?
        .transpose(0, 2)?;
    let u8_b_cpu = Tensor::from_vec((1..=24u8).collect::<Vec<_>>(), (2, 3, 4), &cpu)?
        .transpose(0, 2)?;
    assert_eq!(
        u8_a.add(&u8_b)?.flatten_all()?.to_vec1::<u8>()?,
        u8_a_cpu.add(&u8_b_cpu)?.flatten_all()?.to_vec1::<u8>()?
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_conv_transpose(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let input_vals = vec![1.0f32, 2.0, 3.0];
    let kernel_vals = vec![0.5f32, 1.0, 0.5];
    for dtype in [DType::F32, DType::F16] {
        let input_gpu =
            Tensor::from_vec(input_vals.clone(), (1, 1, 3), device)?.to_dtype(dtype)?;
        let kernel_gpu =
            Tensor::from_vec(kernel_vals.clone(), (1, 1, 3), device)?.to_dtype(dtype)?;
        let input_cpu = Tensor::from_vec(input_vals.clone(), (1, 1, 3), &cpu)?.to_dtype(dtype)?;
        let kernel_cpu = Tensor::from_vec(kernel_vals.clone(), (1, 1, 3), &cpu)?.to_dtype(dtype)?;
        let got = input_gpu.conv_transpose1d(&kernel_gpu, 0, 0, 1, 1, 1)?;
        let want = input_cpu.conv_transpose1d(&kernel_cpu, 0, 0, 1, 1, 1)?;
        assert_tensors_close(
            &got,
            &want,
            dtype,
            &format!("conv_transpose1d {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_rng_determinism(device: &Device) -> Result<()> {
    device.set_seed(0xC0FFEE)?;
    let a = Tensor::rand(0.0f32, 1.0f32, (4, 8), device)?;
    device.set_seed(0xC0FFEE)?;
    let b = Tensor::rand(0.0f32, 1.0f32, (4, 8), device)?;
    assert_tensors_close(&a, &b, DType::F32, "rand seed")?;

    device.set_seed(0xBEEF)?;
    let n1 = Tensor::randn(0.0f32, 1.0f32, (3, 5), device)?;
    device.set_seed(0xBEEF)?;
    let n2 = Tensor::randn(0.0f32, 1.0f32, (3, 5), device)?;
    assert_tensors_close(&n1, &n2, DType::F32, "randn seed")?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_parity_matrix(device: &Device) -> Result<()> {
    parity_unary_dtype_matrix(device)?;
    parity_binary_dtype_matrix(device)?;
    parity_rank_layout_matrix(device)?;
    parity_conv_transpose(device)?;
    parity_rng_determinism(device)?;
    Ok(())
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn gpu_parity_matrix_wgpu() -> Result<()> {
    native_required("gpu_parity_matrix_wgpu", TestBackend::Wgpu, run_parity_matrix)
}

#[test]
#[cfg(feature = "vulkan")]
fn gpu_parity_matrix_vulkan() -> Result<()> {
    native_required(
        "gpu_parity_matrix_vulkan",
        TestBackend::Vulkan,
        run_parity_matrix,
    )
}
