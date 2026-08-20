#![cfg(any(feature = "wgpu", feature = "vulkan"))]

use candle_core::{DType, Device, Tensor};

fn gpu_device() -> candle_core::Result<Device> {
    #[cfg(feature = "wgpu")]
    if std::env::var("CANDLE_DEVICE").as_deref() != Ok("vulkan") {
        return Device::new_wgpu(0);
    }
    #[cfg(feature = "vulkan")]
    return Device::new_vulkan(0);
    #[cfg(not(feature = "vulkan"))]
    candle_core::bail!("no gpu backend")
}

#[test]
fn gpu_f32_to_bf16_and_matmul_dtype() -> candle_core::Result<()> {
    let dev = gpu_device()?;
    let bf16 = Tensor::new(&[1.0f32, 2.0], &dev)?.to_dtype(DType::BF16)?;
    assert_eq!(bf16.dtype(), DType::BF16, "f32->bf16 cast dtype");

    let lhs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev)?.to_dtype(DType::BF16)?;
    let rhs = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &dev)?.to_dtype(DType::BF16)?;
    let out = lhs.matmul(&rhs)?;
    assert_eq!(out.dtype(), DType::BF16, "bf16 matmul output dtype");
    // Value check (odd output width exercises packed-half word handling).
    let lhs3 = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 4),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let rhs3 = Tensor::from_vec(
        vec![1.0f32, 0.5, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0],
        (4, 3),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let gout = lhs3
        .matmul(&rhs3)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cout = lhs3
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .matmul(&rhs3.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (i, (g, c)) in gout.iter().zip(cout.iter()).enumerate() {
        assert!(
            (g - c).abs() <= 0.02 * c.abs().max(1.0),
            "bf16 matmul mismatch at {i}: gpu={g} cpu={c}"
        );
    }
    Ok(())
}

#[test]
fn gpu_index_select_bf16() -> candle_core::Result<()> {
    let dev = gpu_device()?;
    let emb = Tensor::randn(0f32, 1f32, (128, 64), &dev)?.to_dtype(DType::BF16)?;
    let ids = Tensor::new(&[0u32, 3u32, 7u32], &dev)?;
    let out = emb.index_select(&ids, 0)?;
    assert_eq!(out.dtype(), DType::BF16, "bf16 index_select dtype");
    Ok(())
}
