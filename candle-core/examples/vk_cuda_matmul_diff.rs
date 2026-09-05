//! Differential check: CUDA / Vulkan / CPU dense F32 matmul.
//! Validates virtual B^T (tall-skinny), aligned coopmat, and the edge-tolerant
//! unaligned coopmat (attention-shaped M=1025) paths against the CPU result.
use candle_core::{Device, Result, Tensor};

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = a.sub(b)?.abs()?.flatten_all()?;
    d.max(0)?.to_scalar::<f32>()
}

fn main() -> Result<()> {
    let shapes2 = [
        (64usize, 4096, 4096),
        (1024, 1024, 1024),
        (256, 256, 256),
        // unaligned attention-like shapes (edge-tolerant coopmat)
        (1025, 1025, 64),
        (1025, 64, 1025),
        (1025, 3072, 768),
        (1025, 768, 768),
        (1025, 768, 3072),
        // K-edge shapes (K % 8 != 0) previously forced onto the row-looped
        // matvec path: Pi3X projective attention (N,4)@(4,4) and Whisper w@v.
        (16384, 4, 4),
        (4096, 64, 12),
        (1025, 64, 1500),
        (1025, 96, 1500),
        (1374, 3072, 1500),
        // cm1 failure isolation sweep (p.M = candle n, K edges)
        (1025, 64, 1024),
        (1025, 64, 64),
        (1024, 64, 1025),
        (2048, 64, 1025),
        (1025, 96, 1025),
        (1025, 128, 1025),
        (1025, 160, 1025),
        (1025, 192, 1025),
        (1025, 3072, 1025),
        (64, 64, 1025),
        (1025, 65, 1025),
        (1025, 64, 512),
        (1025, 64, 769),
        // n sweep at m=1025, k=1025 (fp32 scalar, cm1 off)
        (1025, 128, 1025),
        (1025, 256, 1025),
        (1025, 512, 1025),
        (1025, 768, 1025),
        (1025, 1024, 1025),
        (1025, 2048, 1025),
        // m sweep at n=64, k=1025
        (128, 64, 1025),
        (512, 64, 1025),
        (4096, 64, 1025),
    ];
    let shapes3 = [
        (24usize, 1025, 1025, 64usize),
        (24, 1025, 64, 1025),
    ];
    let cuda = Device::new_cuda(0)?;
    let vk = Device::new_vulkan(0)?;
    for (m, n, k) in shapes2 {
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
        println!(
            "mm_{m}x{n}x{k}: cpu-cuda max={:.4e} mean={:.4e} | cpu-vk max={:.4e} mean={:.4e}",
            max_abs(&cpu, &cc)?,
            cpu.sub(&cc)?.abs()?.mean_all()?.to_scalar::<f32>()?,
            max_abs(&cpu, &cv)?,
            cpu.sub(&cv)?.abs()?.mean_all()?.to_scalar::<f32>()?,
        );
    }
    for (b_, m, n, k) in shapes3 {
        let a = Tensor::randn(0f32, 1.0, (b_, m, k), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1.0, (b_, k, n), &Device::Cpu)?;
        let cpu = a.matmul(&b)?;
        let cc = a
            .to_device(&cuda)?
            .matmul(&b.to_device(&cuda)?)?
            .to_device(&Device::Cpu)?;
        let cv = a
            .to_device(&vk)?
            .matmul(&b.to_device(&vk)?)?
            .to_device(&Device::Cpu)?;
        println!(
            "bmm_{b_}x{m}x{n}x{k}: cpu-cuda max={:.4e} mean={:.4e} | cpu-vk max={:.4e} mean={:.4e}",
            max_abs(&cpu, &cc)?,
            cpu.sub(&cc)?.abs()?.mean_all()?.to_scalar::<f32>()?,
            max_abs(&cpu, &cv)?,
            cpu.sub(&cv)?.abs()?.mean_all()?.to_scalar::<f32>()?,
        );
    }
    Ok(())
}
