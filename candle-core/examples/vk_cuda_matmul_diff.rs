//! Differential check: CUDA / Vulkan / CPU dense F32 matmul.
//! Validates virtual B^T (tall-skinny) and materialize path (squares).
use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let shapes = [(64usize, 4096, 4096), (1024, 1024, 1024), (256, 256, 256)];
    let cuda = Device::new_cuda(0)?;
    let vk = Device::new_vulkan(0)?;
    for (m, n, k) in shapes {
        let a = Tensor::randn(0f32, 1.0, (m, k), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1.0, (k, n), &Device::Cpu)?;
        let cpu = a.matmul(&b)?;
        let cc = a
            .to_device(&cuda)?
            .matmul(&b.to_device(&cuda)?)?
            .to_device(&Device::Cpu)?;
        let cv = a
            .to_device(&vk)?
            .matmul(&b.to_device(&vk)?)?
            .to_device(&Device::Cpu)?;
        let d_cuda = cpu.sub(&cc)?.abs()?.flatten_all()?;
        let d_vk = cpu.sub(&cv)?.abs()?.flatten_all()?;
        println!(
            "cpu-cuda {m}x{n}x{k}: max={:.6e} mean={:.6e}",
            d_cuda.max(0)?.to_scalar::<f32>()?,
            d_cuda.mean_all()?.to_scalar::<f32>()?
        );
        println!(
            "cpu-vk   {m}x{n}x{k}: max={:.6e} mean={:.6e}",
            d_vk.max(0)?.to_scalar::<f32>()?,
            d_vk.mean_all()?.to_scalar::<f32>()?
        );
    }
    Ok(())
}
