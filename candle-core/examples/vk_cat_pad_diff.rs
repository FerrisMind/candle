//! Differential check: Vulkan vs CPU for cat / replication-pad copy paths
//! (the `copy2d` strided-shader rewrite).
use candle_core::{Device, Result, Tensor};

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = a.sub(b)?.abs()?.flatten_all()?;
    d.max(0)?.to_scalar::<f32>()
}

fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    let (_b, _c, h, w) = xs.dims4()?;
    let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
    let xs = Tensor::cat(&[&first, xs, &last], 3)?;
    let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
    Tensor::cat(&[&first, &xs, &last], 2)
}

fn main() -> Result<()> {
    let vk = Device::new_vulkan(0)?;
    // W-pad killer shape: (2,64,518,518) cat along W with 1-elem strided args
    for (b, c, h, w) in [(2usize, 64usize, 37usize, 37usize), (2, 64, 518, 518), (1, 3, 8, 8)] {
        let xs = Tensor::randn(0f32, 1.0, (b, c, h, w), &Device::Cpu)?;
        let cpu = replication_pad2d(&xs, 1)?;
        let xsv = xs.to_device(&vk)?;
        // warm + timed isolated runs (sync via to_cpu of result)
        let t0 = std::time::Instant::now();
        let v = replication_pad2d(&xsv, 1)?;
        let _ = v.to_device(&Device::Cpu)?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let v2 = replication_pad2d(&xsv, 1)?;
            let _ = v2.to_device(&Device::Cpu)?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        let t1 = t0.elapsed().as_secs_f64();
        eprintln!("  pad timing: first={t1:.3}s best={best:.3}s");
        let v = v.to_device(&Device::Cpu)?;
        println!(
            "pad_{b}x{c}x{h}x{w}: shape {:?} vs {:?} max={:.4e}",
            cpu.shape(),
            v.shape(),
            max_abs(&cpu, &v)?
        );
        // plain cat along dim 3 with strided and contiguous args
        let a = xs.narrow(3, 0, w / 2)?;
        let b_t = xs.narrow(3, w / 2, w - w / 2)?;
        let cpu_cat = Tensor::cat(&[&a, &b_t], 3)?;
        let vk_cat = Tensor::cat(
            &[&a.to_device(&vk)?, &b_t.to_device(&vk)?],
            3,
        )?
        .to_device(&Device::Cpu)?;
        println!("cat_{b}x{c}x{h}x{w}: max={:.4e}", max_abs(&cpu_cat, &vk_cat)?);
    }
    Ok(())
}
