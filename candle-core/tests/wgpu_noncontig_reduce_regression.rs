//! Regression test for the wgpu last-dim reduce unit-stride bug (P5r / MobileSAM).
//! The last-dim sum/extrema kernels index `src[i_src_row + col]` assuming unit
//! stride along the reduced dim. A non-contiguous view (e.g. reshape((b,c,l))->
//! transpose(1,2)) has a reduced-dim stride != 1, so the kernel read consecutive
//! memory instead of strided elements -> silently wrong result on GPU (CPU correct).
//! The fix routes such views to `run_reduce_non_last_dim`, which materializes via a
//! strides-aware copy. This test compares the wgpu result against the CPU reference.
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn wgpu_noncontig_last_dim_sum_matches_cpu() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    let (b, c, l) = (2usize, 6usize, 4usize);
    // Deterministic values; sine/scale so sums are non-trivial and non-symmetric.
    let vals: Vec<f32> = (0..b * c * l)
        .map(|i| ((i as f32) * 1.7f32).sin() * 3.1f32 + (i as f32) * 0.13f32)
        .collect();
    // Contiguous [B, C, L] on wgpu, then transpose to [B, L, C]: last-dim stride = L.
    #[allow(clippy::redundant_clone)]
    let x = Tensor::from_slice(&vals, (b, c, l), &device)?.to_dtype(DType::F32)?;
    let xt = x.transpose(1, 2)?;
    assert_ne!(xt.stride()[2], 1, "test requires a non-contiguous last dim");
    let got = xt.sum(2)?;

    let ref_x = Tensor::from_slice(&vals, (b, c, l), &Device::Cpu)?.to_dtype(DType::F32)?;
    let ref_xt = ref_x.transpose(1, 2)?;
    let ref_sum = ref_xt.sum(2)?;

    let got_v = got.flatten_all()?.to_vec1::<f32>()?;
    let ref_v = ref_sum.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got_v.len(), ref_v.len());
    for (i, (g, r)) in got_v.iter().zip(ref_v.iter()).enumerate() {
        if (g - r).abs() > 1e-3 {
            candle_core::bail!(
                "non-contiguous last-dim sum mismatch at {i}: got {g}, ref {r}"
            );
        }
    }
    Ok(())
}

/// Also validate that sum over the reduced dim is stable across the identity and a
/// non-contiguous-permuted layout for argmax (integer indices must be exact).
#[test]
fn wgpu_noncontig_last_dim_argmax_matches_contiguous() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    let (b, c, l) = (2usize, 7usize, 5usize);
    let vals: Vec<f32> = (0..b * c * l)
        .map(|i| (i as f32 * 0.77).fract() * 1000.0)
        .collect();
    let x = Tensor::from_slice(&vals, (b, c, l), &device)?;
    // argmax over the LAST dim of a transposed (non-contiguous) view.
    let xt = x.transpose(1, 2)?;
    let got = xt.argmax_keepdim(2)?.flatten_all()?.to_vec1::<u32>()?;
    // CPU reference on the same non-contiguous view.
    let ref_x = Tensor::from_slice(&vals, (b, c, l), &Device::Cpu)?;
    let ref_got = ref_x
        .transpose(1, 2)?
        .argmax_keepdim(2)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    assert_eq!(got, ref_got);
    Ok(())
}
