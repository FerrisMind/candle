#[cfg(any(feature = "wgpu", feature = "vulkan"))]
mod bench {
    use candle::{DType, Device, Result, Tensor};
    use std::time::Instant;

    fn rand_tensor(shape: (usize, usize, usize, usize), dev: &Device, seed: u64) -> Result<Tensor> {
        let mut vs = Vec::new();
        let mut s = seed;
        for _ in 0..shape.0 * shape.1 * shape.2 * shape.3 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            vs.push((s as f32 / u32::MAX as f32) - 0.5);
        }
        Tensor::from_vec(vs, shape, dev)
    }

    fn bench_sdpa(
        dev: &Device,
        q: &Tensor, k: &Tensor, v: &Tensor,
        scale: f32, causal: bool,
        iterations: usize,
    ) -> Result<(f64, Tensor)> {
        for _ in 0..3 {
            let _ = candle_nn::ops::sdpa(q, k, v, None, causal, scale, 1.0)?;
        }
        let start = Instant::now();
        let mut out = q.clone();
        for _ in 0..iterations {
            out = candle_nn::ops::sdpa(q, k, v, None, causal, scale, 1.0)?;
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
        Ok((elapsed, out))
    }

    fn bench_unfused(
        dev: &Device,
        q: &Tensor, k: &Tensor, v: &Tensor,
        scale: f32, causal: bool,
        iterations: usize,
    ) -> Result<(f64, Tensor)> {
        for _ in 0..3 {
            let _ = unfused_sdpa(dev, q, k, v, scale, causal)?;
        }
        let start = Instant::now();
        let mut out = q.clone();
        for _ in 0..iterations {
            out = unfused_sdpa(dev, q, k, v, scale, causal)?;
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
        Ok((elapsed, out))
    }

    fn unfused_sdpa(
        dev: &Device,
        q: &Tensor, k: &Tensor, v: &Tensor,
        scale: f32, causal: bool,
    ) -> Result<Tensor> {
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // att = Q @ K^T * scale  →  (B, H, S_q, S_kv)
        let att = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let att = (att * scale as f64)?;

        let q_seq = att.dim(2)?;
        let k_seq = att.dim(3)?;

        if causal {
            let offset = k_seq as isize - q_seq as isize;
            let mut mask_data = Vec::new();
            for i in 0..q_seq {
                for j in 0..k_seq {
                    if j as isize > (i as isize + offset) {
                        mask_data.push(f32::NEG_INFINITY);
                    } else {
                        mask_data.push(0.0f32);
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, q_seq, k_seq), dev)?;
            let att = att.broadcast_add(&mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&att)?;
            probs.matmul(&v)
        } else {
            let probs = candle_nn::ops::softmax_last_dim(&att)?;
            probs.matmul(&v)
        }
    }

    fn compare(label: &str, fused: &Tensor, unfused: &Tensor) -> Result<f32> {
        let f = fused.to_dtype(DType::F32)?;
        let u = unfused.to_dtype(DType::F32)?;
        let abs_diff = (&f - &u)?.abs()?;
        let max_diff = abs_diff.max_all()?.to_scalar::<f32>()?;
        let mean_diff = abs_diff.mean_all()?.to_scalar::<f32>()?;
        println!("  [{label}] max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}");
        Ok(max_diff)
    }

    #[test]
    #[ignore = "bench: fused vs unfused SDPA timing + correctness"]
    #[cfg(feature = "vulkan")]
    fn bench_vulkan_sdpa() -> Result<()> {
        if std::env::var("CANDLE_REQUIRE_VULKAN_TEST_DEVICE").is_err() {
            println!("Skipping: set CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1");
            return Ok(());
        }
        let dev = Device::new_vulkan(0)?;
        run_benchmarks(&dev, "vulkan")
    }

    #[test]
    #[ignore = "bench: fused vs unfused SDPA timing + correctness"]
    #[cfg(feature = "wgpu")]
    fn bench_wgpu_sdpa() -> Result<()> {
        if std::env::var("CANDLE_REQUIRE_WGPU_TEST_DEVICE").is_err() {
            println!("Skipping: set CANDLE_REQUIRE_WGPU_TEST_DEVICE=1");
            return Ok(());
        }
        let dev = Device::new_wgpu(0)?;
        run_benchmarks(&dev, "wgpu")
    }

    fn run_benchmarks(dev: &Device, backend: &str) -> Result<()> {
        let iters = 20;
        let scale = 1.0f32 / (64.0f32).sqrt();

        struct Case {
            name: &'static str,
            b: usize, h: usize, sq: usize, skv: usize, hd: usize, causal: bool,
        }

        let cases = vec![
            Case { name: "tiny_32x64",    b: 1, h: 2, sq: 32,  skv: 64,  hd: 64, causal: true },
            Case { name: "small_128x128",  b: 1, h: 8, sq: 128, skv: 128, hd: 64, causal: true },
            Case { name: "mid_256x256",    b: 1, h: 8, sq: 256, skv: 256, hd: 64, causal: true },
            Case { name: "long_512x512",   b: 1, h: 8, sq: 512, skv: 512, hd: 64, causal: true },
            Case { name: "cross_64x256",   b: 1, h: 8, sq: 64,  skv: 256, hd: 64, causal: false },
            Case { name: "batch4_128",     b: 4, h: 8, sq: 128, skv: 128, hd: 64, causal: true },
            Case { name: "wide_128x128",   b: 1, h: 4, sq: 128, skv: 128, hd: 128, causal: true },
        ];

        println!("\n=== {} SDPA: fused vs unfused ({} iters each) ===", backend.to_uppercase(), iters);
        println!("{:<18} {:>10} {:>10} {:>8} {:>14} {:>8}",
            "Config", "Fused(ms)", "Unfused(ms)", "Speedup", "MaxDiff", "OK?");
        println!("{}", "-".repeat(78));

        for c in &cases {
            let q = rand_tensor((c.b, c.h, c.sq, c.hd), dev, 42)?;
            let k = rand_tensor((c.b, c.h, c.skv, c.hd), dev, 43)?;
            let v = rand_tensor((c.b, c.h, c.skv, c.hd), dev, 44)?;

            let (fused_ms, fused_out) = bench_sdpa(dev, &q, &k, &v, scale, c.causal, iters)?;
            let (unfused_ms, unfused_out) = bench_unfused(dev, &q, &k, &v, scale, c.causal, iters)?;

            let speedup = if fused_ms > 0.0001 { unfused_ms / fused_ms } else { 0.0 };
            let max_diff = compare(c.name, &fused_out, &unfused_out)?;
            let ok = max_diff < 0.01;

            println!("{:<18} {:>10.2} {:>10.2} {:>7.2}x {:>14.6e} {:>8}",
                c.name, fused_ms * 1000.0, unfused_ms * 1000.0, speedup, max_diff,
                if ok { "OK" } else { "FAIL" });
        }
        println!();
        Ok(())
    }
}
