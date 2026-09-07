use candle_core::{DType, Device, Tensor};
fn main() -> anyhow::Result<()> {
    let vk = Device::new_vulkan(0)?;
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let cpu = Tensor::randn(0f32, 1f32, (64, 96), &Device::Cpu)?
            .abs()?.to_dtype(dtype)?; // strictly positive for sqrt
        let gpu = cpu.to_device(&vk)?;
        let r_cpu = cpu.affine(2.5f64, -1.25f64)?;
        let r_gpu = gpu.affine(2.5f64, -1.25f64)?.to_device(&Device::Cpu)?;
        let d = (r_cpu.to_dtype(DType::F32)? - r_gpu.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("affine {dtype:?} maxdiff {d}");
        let s_cpu = cpu.sqrt()?;
        let s_gpu = gpu.sqrt()?.to_device(&Device::Cpu)?;
        let d = (s_cpu.to_dtype(DType::F32)? - s_gpu.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("sqrt   {dtype:?} maxdiff {d}");
    }
    Ok(())
}
