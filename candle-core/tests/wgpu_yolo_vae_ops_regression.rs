//! Regression: the wgpu kernels for every yolo-v8 class-path op (conv2d,
//! silu, channel-slice view, sigmoid) and SD-VAE decode op (group-norm
//! last-dim reduction) must match the CPU reference (relative err < 2e-3) and
//! be DETERMINISTIC across repeated runs. This guards the base-op correctness
//! that the Worker-F1 wave validated: the yolo "motorbike zeroed" /
//! nondeterminism and the SD "green channel ≈ 0" defects are NOT produced by
//! the kernels here, so a divergence in these ops is a new regression.
#![cfg(feature = "wgpu")]

use candle_core::{Device, IndexOp, Result, Tensor};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 13) % 4093) as f32 / 61.0 - 32.0
        })
        .collect()
}

fn device() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

fn relerr(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (((x - y).abs()) as f64) / (x.hypot(*y).max(1e-6) as f64))
        .fold(0.0, f64::max)
}

fn sigmoid(t: &Tensor) -> Result<Tensor> {
    let e = t.neg()?.exp()?;
    let one = Tensor::full(1f32, t.shape(), t.device())?;
    &one / &(&one + &e)?
}

fn silu(t: &Tensor) -> Result<Tensor> {
    t * sigmoid(t)?
}

fn group_norm_lastdim(t: &Tensor, groups: usize) -> Result<Tensor> {
    let (b, c, h, w) = t.dims4()?;
    let hidden = h * w * c / groups;
    let x = t.reshape((b, groups, hidden))?;
    let eps = 1e-6f64;
    let mean = (x.sum_keepdim(2)? / hidden as f64)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = (xc.sqr()?.sum_keepdim(2)? / hidden as f64)?;
    let norm = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    norm.reshape((b, c, h, w))
}

fn check_det<F>(name: &str, data: &[f32], shape: &[usize], iters: usize, op: F) -> Result<()>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let Some(dev) = device() else { return Ok(()) };
    let mut first: Option<Vec<f32>> = None;
    for k in 0..iters {
        let cpu = op(&Tensor::from_slice(data, shape, &Device::Cpu)?)?;
        let wgpu = op(&Tensor::from_slice(data, shape, &dev)?)?;
        let c = cpu.flatten_all()?.to_vec1::<f32>()?;
        let w = wgpu.flatten_all()?.to_vec1::<f32>()?;
        let r = relerr(&c, &w);
        assert!(r < 2e-3, "[{name}] iter{k} cpu-vs-wgpu rel={r:.4}");
        if k == 0 {
            first = Some(w.clone());
        } else if let Some(f) = &first {
            let r2 = relerr(f, &w);
            assert!(r2 < 2e-3, "[{name}] NONdeterminism iter{k} rel={r2:.4}");
        }
    }
    Ok(())
}

#[test]
fn wgpu_yolo_vae_ops_correct_deterministic() -> Result<()> {
    if device().is_none() {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    }
    const IT: usize = 20;

    // yolo head conv: silu -> conv1x1 256->84 -> sigmoid.
    let w84: Vec<f32> = lcg(1, 84 * 256);
    check_det("head_conv1x1_84", &lcg(2, 1 * 256 * 8 * 8), &[1, 256, 8, 8], IT, move |t| {
        let w = Tensor::from_slice(&w84, (84usize, 256, 1, 1), t.device())?;
        sigmoid(&silu(t)?.conv2d(&w, 0, 1, 1, 1)?)
    })?;

    // cv3 chain: conv3x3+silu x2 -> conv1x1 -> sigmoid (84 classes).
    let w3 = lcg(3, 128 * 64 * 3 * 3);
    let w31 = lcg(4, 128 * 128 * 3 * 3);
    let head = lcg(5, 84 * 128);
    check_det("cv3_chain_84", &lcg(6, 1 * 64 * 16 * 16), &[1, 64, 16, 16], IT, move |t| {
        let wa = Tensor::from_slice(&w3, (128usize, 64, 3, 3), t.device())?;
        let a = silu(&t.conv2d(&wa, 1, 1, 1, 1)?)?;
        let wb = Tensor::from_slice(&w31, (128usize, 128, 3, 3), t.device())?;
        let b = silu(&a.conv2d(&wb, 1, 1, 1, 1)?)?;
        let wh = Tensor::from_slice(&head, (84usize, 128, 1, 1), t.device())?;
        sigmoid(&b.conv2d(&wh, 0, 1, 1, 1)?)
    })?;

    // channel-slice view with offset -> sigmoid (the yolo CLS path).
    check_det("slice_view_sigmoid", &lcg(7, 1 * 144 * 8400), &[1, 144, 8400], IT, |t| {
        sigmoid(&t.i((.., 64..))?)
    })?;

    // mid-dim reduction (DFL softmax components).
    check_det("sum_keepdim_d1", &lcg(8, 1 * 16 * 4 * 32), &[1, 16, 4, 32], IT, |t| {
        t.sum_keepdim(1)
    })?;

    // SD VAE group-norm last-dim reduction (per-channel), green (channel 1).
    check_det("group_norm_lastdim", &lcg(9, 1 * 3 * 8 * 8), &[1, 3, 8, 8], IT, |t| {
        group_norm_lastdim(t, 3)
    })?;

    // neck upsample + cat, and chunk + cat (C2f).
    let ub = lcg(10, 1 * 256 * 16 * 16);
    check_det("upsample_cat", &lcg(11, 1 * 512 * 8 * 8), &[1, 512, 8, 8], IT, move |t| {
        let u = t.upsample_nearest2d(16, 16)?;
        let b = Tensor::from_slice(&ub, (1usize, 256, 16, 16), t.device())?;
        Tensor::cat(&[&u, &b], 1)
    })?;
    check_det("chunk_cat", &lcg(12, 1 * 128 * 8 * 8), &[1, 128, 8, 8], IT, |t| {
        let parts = t.chunk(2, 1)?;
        Tensor::cat(&parts, 1)
    })?;

    Ok(())
}
