use candle_core::{Device, DType, Result, Tensor};
use std::time::Instant;

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_matmul(dev: &Device, m: usize, n: usize, k: usize, iters: usize) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (m, k), dev)?;
    let b = Tensor::randn(0f32, 1.0, (k, n), dev)?;
    // warmup
    for _ in 0..5 {
        let _ = a.matmul(&b)?;
        dev.synchronize()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let c = a.matmul(&b)?;
        dev.synchronize()?;
        let _ = c.dtype();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_ms(times))
}

fn main() -> Result<()> {
    let shapes = [(256usize, 256, 256), (1024, 1024, 1024), (64, 4096, 4096)];
    let iters = 20usize;
    println!("backend,m,n,k,median_ms");
    #[cfg(feature = "cuda")]
    {
        let cuda = Device::new_cuda(0)?;
        for (m, n, k) in shapes {
            let ms = bench_matmul(&cuda, m, n, k, iters)?;
            println!("cuda,{m},{n},{k},{ms:.4}");
        }
    }
    #[cfg(feature = "vulkan")]
    {
        let vk = Device::new_vulkan(0)?;
        for (m, n, k) in shapes {
            let ms = bench_matmul(&vk, m, n, k, iters)?;
            println!("vulkan,{m},{n},{k},{ms:.4}");
        }
    }
    #[cfg(feature = "wgpu")]
    {
        let wg = Device::new_wgpu(0)?;
        for (m, n, k) in shapes {
            let ms = bench_matmul(&wg, m, n, k, iters)?;
            println!("wgpu,{m},{n},{k},{ms:.4}");
        }
    }
    let _ = DType::F32;
    Ok(())
}
