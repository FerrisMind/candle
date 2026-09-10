//! Composite per-step benchmark mirroring the pi3 encoder/decoder attention
//! and MLP blocks (same strided views, contiguous copies, softmax, linears),
//! sync-timed step by step on CUDA vs Vulkan to attribute the per-op gap.
//!
//! Encoder shapes: seq=1374 (37*37 + cls + 4 reg), d=1024, heads=16, mlp=4096,
//! batch=2 frames. Decoder shapes: seq=1025, d=768, heads=12, mlp=3072.
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::time::Instant;

fn bench<F>(dev: &Device, iters: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..2 {
        drop(f()?);
        dev.synchronize()?;
    }
    let mut total = 0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        drop(f()?);
        dev.synchronize()?;
        total += t0.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(total / iters as f64)
}

fn report(dev_name: &str, tag: &str, ms: f64) {
    println!("{tag:36} {dev_name:8} {ms:9.3} ms");
}

struct SoftmaxLastDimFused;

impl candle_core::CustomOp1 for SoftmaxLastDimFused {
    fn name(&self) -> &'static str {
        "softmax-last-dim-fused"
    }
    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // reference (single-threaded) softmax for the cpu path
        let dims = l.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let src = match s {
            candle_core::CpuStorage::F32(slice) => slice,
            other => candle_core::bail!("unsupported {other:?}"),
        };
        let src = match l.contiguous_offsets() {
            None => candle_core::bail!("input has to be contiguous"),
            Some((o1, o2)) => &src[o1..o2],
        };
        let mut dst = vec![0f32; src.len()];
        for (src_row, dst_row) in src.chunks(dim_m1).zip(dst.chunks_mut(dim_m1)) {
            let max = src_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                let e = (s - max).exp();
                sum += e;
                *d = e;
            }
            for d in dst_row.iter_mut() {
                *d /= sum;
            }
        }
        use candle_core::WithDType;
        Ok((
            candle_core::WithDType::to_cpu_storage_owned(dst),
            l.shape().clone(),
        ))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s: &candle_core::VulkanStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::VulkanStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        let storage = s.softmax_last_dim(l)?;
        Ok((storage, l.shape().clone()))
    }
}

fn main() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let vk = Device::new_vulkan(0)?;
    let iters = 8;

    for (name, (seq, d, heads, mlp, b)) in [
        ("enc", (1374usize, 1024usize, 16usize, 4096usize, 2usize)),
        ("dec", (1025, 768, 12, 3072, 2)),
    ] {
        let hd = d / heads;
        println!("== {name} block: seq={seq} d={d} heads={heads} mlp={mlp} batch={b}");
        for (dn, dev) in [("cuda", &cuda), ("vulkan", &vk)] {
            // -- attention inputs (as produced by the decoder/encoder pattern)
            let xs = Tensor::randn(0f32, 1.0, (b, seq, d), dev)?;

            // qkv linear on contiguous input
            let wqkv = Tensor::randn(0f32, 1.0, (d, 3 * d), dev)?;
            let ms = bench(dev, iters, || xs.reshape((b * seq, d))?.matmul(&wqkv))?;
            report(dn, "qkv_linear (contig lhs)", ms);

            // qkv linear on a STRIDED lhs (norm output view pattern:
            // narrow a (b, seq+5, d) then use last seq rows — strided)
            let xs_wide = Tensor::randn(0f32, 1.0, (b, seq + 5, d), dev)?;
            let xs_strided = xs_wide.i((.., 5..))?;
            let ms = bench(dev, iters, || xs_strided.reshape((b * seq, d))?.matmul(&wqkv))?;
            report(dn, "qkv_linear (strided lhs)", ms);

            // split heads exactly like the models
            let qkv = xs
                .reshape((b * seq, d))?
                .matmul(&wqkv)?
                .reshape((b, seq, 3, heads, hd))?
                .transpose(1, 3)?;
            let q = qkv.i((.., .., 0))?.contiguous()?;
            let k = qkv.i((.., .., 1))?.contiguous()?;
            let v = qkv.i((.., .., 2))?.contiguous()?;
            let ms = {
                // time the three contiguous() copies directly
                let mut total = 0f64;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let _q = qkv.i((.., .., 0))?.contiguous()?;
                    let _k = qkv.i((.., .., 1))?.contiguous()?;
                    let _v = qkv.i((.., .., 2))?.contiguous()?;
                    dev.synchronize()?;
                    total += t0.elapsed().as_secs_f64() * 1000.0;
                }
                total / iters as f64
            };
            report(dn, "head split + 3x contiguous", ms);

            // scores with the transposed-rhs VIEW (materializes B^T on vulkan)
            let ms = bench(dev, iters, || q.matmul(&k.transpose(2, 3)?))?;
            report(dn, "scores = q @ kT(view)", ms);

            let scores = q.matmul(&k.transpose(2, 3)?)?;
            let ms = bench(dev, iters, || {
                let max = scores.max_keepdim(D::Minus1)?;
                let diff = scores.broadcast_sub(&max)?;
                let num = diff.exp()?;
                let den = num.sum_keepdim(D::Minus1)?;
                num.broadcast_div(&den)
            })?;
            report(dn, "softmax_last_dim", ms);
            if dn == "vulkan" {
                let ms = bench(dev, iters, || scores.apply_op1_no_bwd(&SoftmaxLastDimFused))?;
                report(dn, "softmax FUSED (soft_max_f32)", ms);
            }

            let attn = {
                let max = scores.max_keepdim(D::Minus1)?;
                let diff = scores.broadcast_sub(&max)?;
                let num = diff.exp()?;
                let den = num.sum_keepdim(D::Minus1)?;
                num.broadcast_div(&den)
            }?;
            let ms = bench(dev, iters, || attn.matmul(&v))?;
            report(dn, "attn @ v", ms);

            // out merge + proj linear
            let merged = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?;
            println!("  merge shape {:?}", merged.shape());
            let wo = Tensor::randn(0f32, 1.0, (d, d), dev)?;
            let ms = bench(dev, iters, || {
                attn.matmul(&v)?
                    .transpose(1, 2)?
                    .reshape((b * seq, d))?
                    .matmul(&wo)
            })?;
            report(dn, "merge + proj linear", ms);

            // mlp
            let w_up = Tensor::randn(0f32, 1.0, (d, mlp), dev)?;
            let w_down = Tensor::randn(0f32, 1.0, (mlp, d), dev)?;
            let ms = bench(dev, iters, || {
                let h = xs.reshape((b * seq, d))?.matmul(&w_up)?;
                let h = h.gelu_erf()?;
                h.reshape((b * seq, mlp))?.matmul(&w_down)
            })?;
            report(dn, "mlp (up+gelu+down)", ms);

            // layernorm (mean/var + affine)
            let ms = bench(dev, iters, || {
                let mean = xs.mean_keepdim(D::Minus1)?;
                let centered = xs.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
                let norm = centered.broadcast_div(&(var + 1e-6)?.sqrt()?)?;
                norm.affine(1.0, 0.0)
            })?;
            report(dn, "layernorm", ms);

            let _ = (q, k, v, scores, attn, D::Minus1, DType::F32);
        }
    }
    Ok(())
}
