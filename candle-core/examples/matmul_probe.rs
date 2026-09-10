//! Scratch probe: time specific F32 matmul shapes on the vulkan device to
//! quantify the unaligned-cm1 (TF32) vs matmul_f32_f32_fp32 (scalar tile)
//! dispatch choice. Not part of the test suite.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench_shape(device: &Device, name: &str, shape: (usize, usize, usize), rank: usize, iters: usize) -> candle_core::Result<()> {
    let (m, n, k) = shape;
    let b = if rank > 2 { 1 } else { 0 };
    let a_shape: Vec<usize> = if rank > 2 { vec![1, m, k] } else { vec![m, k] };
    let b_shape: Vec<usize> = if rank > 2 { vec![1, k, n] } else { vec![k, n] };
    let a_vals: Vec<f32> = (0..m * k).map(|i| ((i % 71) as f32 - 35.0) / 17.0).collect();
    let b_vals: Vec<f32> = (0..k * n).map(|i| ((i % 53) as f32 - 26.0) / 13.0).collect();
    let a = Tensor::from_vec(a_vals, a_shape, device)?;
    let bm = Tensor::from_vec(b_vals, b_shape, device)?;
    let _ = b;
    // warmup
    for _ in 0..3 {
        let c = a.matmul(&bm)?;
        device.synchronize()?;
        drop(c);
    }
    let mut times = Vec::new();
    for _ in 0..iters {
        let start = Instant::now();
        let c = a.matmul(&bm)?;
        device.synchronize()?;
        let el = start.elapsed();
        drop(c);
        times.push(el);
    }
    times.sort();
    let med = times[times.len() / 2];
    println!("rank{rank} {name} m={m} n={n} k={k}: median {:?} ({} iters)", med, iters);
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let device = Device::new_vulkan(0)?;
    let shapes: &[(&str, (usize, usize, usize), usize)] = &[
        ("sweep_fail", (128, 196, 256), 2),
        ("sweep_fail2", (256, 196, 1152), 2),
        ("attn_ctx", (1025, 1101, 960), 3),
        ("pi3_linear", (3903, 3072, 1536), 2),
        ("pi3x_linear", (1301, 1536, 1152), 2),
        ("aligned_ref", (4096, 4096, 4096), 2),
    ];
    for (name, shape, rank) in shapes {
        let iters = if shape.0 * shape.1 * shape.2 > 1_000_000_000 { 5 } else { 20 };
        bench_shape(&device, name, *shape, *rank, iters)?;
    }
    Ok(())
}
