//! Microbench: dense F32 matmul latency for CUDA / Vulkan / WGPU.
//!
//! Reports:
//! - `sync`: one matmul + device.synchronize() per sample (steady-state, cold host path)
//! - `batch`: N matmuls then one synchronize; per-matmul = total/N (amortized host)
use candle_core::{Device, DType, Result, Tensor};
use std::time::Instant;

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_matmul_sync(dev: &Device, m: usize, n: usize, k: usize, iters: usize) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (m, k), dev)?;
    let b = Tensor::randn(0f32, 1.0, (k, n), dev)?;
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

fn bench_matmul_batch(
    dev: &Device,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    batch: usize,
) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (m, k), dev)?;
    let b = Tensor::randn(0f32, 1.0, (k, n), dev)?;
    for _ in 0..3 {
        for _ in 0..batch {
            let _ = a.matmul(&b)?;
        }
        dev.synchronize()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        // Drop each result immediately so we measure kernel+dispatch, not
        // allocator pressure from retaining `batch` full outputs.
        for _ in 0..batch {
            let c = a.matmul(&b)?;
            std::mem::drop(c);
        }
        dev.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0 / batch as f64);
    }
    Ok(median_ms(times))
}

fn run_backend(name: &str, dev: &Device, shapes: &[(usize, usize, usize)], iters: usize) -> Result<()> {
    let batch = 20usize;
    for &(m, n, k) in shapes {
        let sync_ms = bench_matmul_sync(dev, m, n, k, iters)?;
        let batch_ms = bench_matmul_batch(dev, m, n, k, iters, batch)?;
        println!("{name},{m},{n},{k},sync,{sync_ms:.4}");
        println!("{name},{m},{n},{k},batch{batch},{batch_ms:.4}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let shapes = [(256usize, 256, 256), (1024, 1024, 1024), (64, 4096, 4096)];
    let iters = 20usize;
    println!("backend,m,n,k,mode,median_ms");
    #[cfg(feature = "cuda")]
    {
        let cuda = Device::new_cuda(0)?;
        run_backend("cuda", &cuda, &shapes, iters)?;
    }
    #[cfg(feature = "vulkan")]
    {
        let vk = Device::new_vulkan(0)?;
        run_backend("vulkan", &vk, &shapes, iters)?;
    }
    #[cfg(feature = "wgpu")]
    {
        let wg = Device::new_wgpu(0)?;
        run_backend("wgpu", &wg, &shapes, iters)?;
    }
    let _ = DType::F32;
    Ok(())
}
