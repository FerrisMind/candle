//! Perf check on the real model GEMM shapes: CUDA vs Vulkan (and wgpu).
//!
//! Plain 2D matmul plus batched (bmm) attention shapes. Reports median ms and
//! effective TFLOP/s so the scalar-fp32 vs coopmat gap is visible per shape.
//!
//! Run:
//!   cargo run --release --features cuda,vulkan --example perf_model_shapes
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    v[v.len() / 2]
}

fn bench<F>(dev: &Device, iters: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..3 {
        drop(f()?);
        dev.synchronize()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        drop(f()?);
        dev.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_ms(times))
}

fn emit(tag: &str, backend: &str, ms: f64, flops: f64) {
    let tflops = flops / (ms / 1000.0) / 1e12;
    println!("{tag},{backend},{ms:.3},{tflops:.2}");
}

fn main() -> Result<()> {
    println!("shape,backend,median_ms,tflops");

    // (m, n, k) — plain dense GEMM shapes from pi3/pi3x/triposr + aligned twins.
    let shapes: &[(usize, usize, usize)] = &[
        (1025, 1025, 64),  // attn q@kT (M unaligned)
        (1025, 64, 1025),  // attn @vT (M unaligned)
        (1025, 768, 768),  // qkv proj
        (1025, 3072, 768), // MLP up
        (1025, 768, 3072), // MLP down
        (1088, 3072, 768), // aligned-M twin of MLP up
        (1024, 1024, 64),  // aligned twin of q@kT
        (2048, 2048, 2048),
        (4096, 4096, 1024),
    ];

    // Batched attention: 12 heads over M=1025 (b*heads=24 typical for two frames).
    let bshapes: &[(usize, usize, usize, usize)] = &[
        (24, 1025, 1025, 64), // q@kT batched
        (24, 1025, 64, 1025), // @vT batched
    ];

    let cuda = Device::new_cuda(0).ok();
    let vk = Device::new_vulkan(0).ok();
    let wgpu = Device::new_wgpu(0).ok();

    for &(m, n, k) in shapes {
        let a = Tensor::randn(0f32, 1.0, (m, k), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1.0, (k, n), &Device::Cpu)?;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tag = format!("mm_{m}x{n}x{k}");
        for (name, dev) in [("cuda", &cuda), ("vulkan", &vk), ("wgpu", &wgpu)] {
            let Some(dev) = dev else { continue };
            let a = a.to_device(dev)?;
            let b = b.to_device(dev)?;
            let ms = bench(dev, 10, || a.matmul(&b))?;
            emit(&tag, name, ms, flops);
        }
    }

    for &(bh, m, n, k) in bshapes {
        let a = Tensor::randn(0f32, 1.0, (bh, m, k), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1.0, (bh, k, n), &Device::Cpu)?;
        let flops = 2.0 * bh as f64 * m as f64 * n as f64 * k as f64;
        let tag = format!("bmm_{bh}x{m}x{n}x{k}");
        for (name, dev) in [("cuda", &cuda), ("vulkan", &vk), ("wgpu", &wgpu)] {
            let Some(dev) = dev else { continue };
            let a = a.to_device(dev)?;
            let b = b.to_device(dev)?;
            let ms = bench(dev, 8, || a.matmul(&b))?;
            emit(&tag, name, ms, flops);
        }
    }
    print_vulkan_profile();
    Ok(())
}

#[allow(dead_code)]
fn print_vulkan_profile() {
    if let Some((wall, rows)) = candle_core::vulkan_gpu_profile_report() {
        println!("== vulkan gpu profile (aggregation wall {wall:.2}s)");
        for (name, count, total_ms) in rows.iter().take(20) {
            println!("  {name:40} {count:>6}x {total_ms:>9.2}ms");
        }
    }
}
