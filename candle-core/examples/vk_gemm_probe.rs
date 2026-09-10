use candle_core::{Device, Tensor};
use std::time::Instant;

fn timed(name: &str, m: usize, k: usize, n: usize) -> candle_core::Result<()> {
    let dev = Device::new_vulkan(0)?;
    let mk = m * k;
    let kn = k * n;
    let a: Vec<f32> = (0..mk).map(|v| (v as f32 / mk as f32) * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..kn).map(|v| (v as f32 / kn as f32) * 2.0 - 1.0).collect();
    let ta = Tensor::from_vec(a.clone(), (m, k), &dev)?;
    let tb = Tensor::from_vec(b.clone(), (k, n), &dev)?;
    let out = ta.matmul(&tb)?;
    let out_cpu = ta
        .to_device(&Device::Cpu)?
        .matmul(&tb.to_device(&Device::Cpu)?)?;
    let g = out.flatten_all()?.to_vec1::<f32>()?;
    let c = out_cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0f32;
    let mut rel = 0f32;
    for (gv, cv) in g.iter().zip(c.iter()) {
        max = max.max((gv - cv).abs());
        if cv.abs() > 1e-3 {
            rel = rel.max(((gv - cv) / cv).abs());
        }
    }
    // Timed (exclude first-call pipeline compile).
    for _ in 0..3 {
        ta.matmul(&tb)?;
    }
    let iters = 20;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ta.matmul(&tb)?;
    }
    dev.synchronize()?;
    let per = start.elapsed() / iters;
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    println!(
        "{name} ({m}x{k})@({k}x{n}): {:?} {:.1} GFLOP/s  max_abs={max:e} max_rel={rel:e}",
        per,
        gflop / per.as_secs_f64() / 1e9
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    timed("linear_large", 17424, 768, 3072)?;
    timed("square_1024", 1024, 1024, 1024)?;
    timed("attn_4d", 121 * 24, 32, 144)?;
    Ok(())
}
