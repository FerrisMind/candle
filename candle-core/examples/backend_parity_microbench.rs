//! Microbench: CUDA / Vulkan / WGPU steady-state latency.
//!
//! Reports:
//! - `sync`: one op + device.synchronize() per sample
//! - `batch`: N ops then one synchronize; per-op = total/N
//!
//! Default: dense F32 matmul shapes. Pass `--suite` for unary/binary/softmax too.
use candle_core::{DType, Device, Result, Tensor, D};
use std::time::Instant;

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_op_sync<F>(dev: &Device, iters: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..5 {
        let _ = f()?;
        dev.synchronize()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let c = f()?;
        dev.synchronize()?;
        let _ = c.dtype();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_ms(times))
}

fn bench_op_batch<F>(dev: &Device, iters: usize, batch: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..3 {
        for _ in 0..batch {
            let _ = f()?;
        }
        dev.synchronize()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..batch {
            let c = f()?;
            std::mem::drop(c);
        }
        dev.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0 / batch as f64);
    }
    Ok(median_ms(times))
}

fn emit(name: &str, op: &str, mode: &str, ms: f64) {
    println!("{name},{op},{mode},{ms:.4}");
}

fn run_matmul(
    name: &str,
    dev: &Device,
    shapes: &[(usize, usize, usize)],
    iters: usize,
) -> Result<()> {
    let batch = 20usize;
    for &(m, n, k) in shapes {
        let a = Tensor::randn(0f32, 1.0, (m, k), dev)?;
        let b = Tensor::randn(0f32, 1.0, (k, n), dev)?;
        let tag = format!("matmul_{m}x{n}x{k}");
        let sync_ms = bench_op_sync(dev, iters, || a.matmul(&b))?;
        let batch_ms = bench_op_batch(dev, iters, batch, || a.matmul(&b))?;
        emit(name, &tag, "sync", sync_ms);
        emit(name, &tag, &format!("batch{batch}"), batch_ms);
    }
    Ok(())
}

fn run_suite_ops(name: &str, dev: &Device, iters: usize) -> Result<()> {
    let batch = 20usize;
    let x = Tensor::randn(0f32, 1.0, (1024, 1024), dev)?;
    let y = Tensor::randn(0f32, 1.0, (1024, 1024), dev)?;
    let sync_ms = bench_op_sync(dev, iters, || x.relu())?;
    let batch_ms = bench_op_batch(dev, iters, batch, || x.relu())?;
    emit(name, "relu_1024", "sync", sync_ms);
    emit(name, "relu_1024", &format!("batch{batch}"), batch_ms);

    let sync_ms = bench_op_sync(dev, iters, || x.mul(&y))?;
    let batch_ms = bench_op_batch(dev, iters, batch, || x.mul(&y))?;
    emit(name, "mul_1024", "sync", sync_ms);
    emit(name, "mul_1024", &format!("batch{batch}"), batch_ms);

    let sync_ms = bench_op_sync(dev, iters, || x.sum_keepdim(D::Minus1))?;
    let batch_ms = bench_op_batch(dev, iters, batch, || x.sum_keepdim(D::Minus1))?;
    emit(name, "sum_last_1024", "sync", sync_ms);
    emit(name, "sum_last_1024", &format!("batch{batch}"), batch_ms);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let suite = args.iter().any(|a| a == "--suite");
    let matmul_only = args.iter().any(|a| a == "--matmul-only") || !suite;
    let shapes = [(256usize, 256, 256), (1024, 1024, 1024), (64, 4096, 4096)];
    let iters = 20usize;
    println!("backend,op,mode,median_ms");
    let run = |name: &str, dev: &Device| -> Result<()> {
        if matmul_only || suite {
            run_matmul(name, dev, &shapes, iters)?;
        }
        if suite {
            run_suite_ops(name, dev, iters)?;
        }
        Ok(())
    };
    #[cfg(feature = "cuda")]
    {
        let cuda = Device::new_cuda(0)?;
        run("cuda", &cuda)?;
    }
    #[cfg(feature = "vulkan")]
    {
        let vk = Device::new_vulkan(0)?;
        run("vulkan", &vk)?;
    }
    #[cfg(feature = "wgpu")]
    {
        let wg = Device::new_wgpu(0)?;
        run("wgpu", &wg)?;
    }
    let _ = DType::F32;
    Ok(())
}
