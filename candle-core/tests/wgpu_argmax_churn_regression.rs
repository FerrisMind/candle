//! Regression test for the wgpu elementwise bind-group cache ABA race
//! (fixed in 8bb2754e). The cache was keyed by the ADDRESS of the
//! `wgpu::Buffer` handle inside each Arc slot: when a short-lived tensor's
//! Arc died and a new buffer was allocated at the same address, a cache hit
//! replayed a stale bind group referencing the old (pinned) buffer, so
//! argmax/argmin intermittently returned wrong indices under allocation
//! churn. Churning small tensors through argmax/argmin with drops in between
//! maximizes address reuse; results are compared against the CPU reference at
//! chunk checkpoints (readback happens only there, so work stays in flight
//! within a chunk, which the race required).
#![cfg(feature = "wgpu")]

use candle_core::{Device, Result, Tensor};

fn cpu_argmax_row(vals: &[f32], row: usize, ne0: usize) -> u32 {
    let mut best = vals[row * ne0];
    let mut best_idx = 0usize;
    for j in 1..ne0 {
        let v = vals[row * ne0 + j];
        if v > best {
            best = v;
            best_idx = j;
        }
    }
    best_idx as u32
}

fn cpu_argmin_row(vals: &[f32], row: usize, ne0: usize) -> u32 {
    let mut best = vals[row * ne0];
    let mut best_idx = 0usize;
    for j in 1..ne0 {
        let v = vals[row * ne0 + j];
        if v < best {
            best = v;
            best_idx = j;
        }
    }
    best_idx as u32
}

fn make_vals(it: u64) -> Vec<f32> {
    (0..6)
        .map(|j| {
            let mixed = it
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add((j as u64).wrapping_mul(1442695040888963407));
            ((mixed >> 13) % 4093) as f32 / 61.0
        })
        .collect()
}

fn expected_rows(vals: &[f32], ne0: usize, argmax: bool) -> (u32, u32) {
    if argmax {
        (cpu_argmax_row(vals, 0, ne0), cpu_argmax_row(vals, 1, ne0))
    } else {
        (cpu_argmin_row(vals, 0, ne0), cpu_argmin_row(vals, 1, ne0))
    }
}

#[test]
fn wgpu_argmax_churn_matches_cpu() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    const NE0: usize = 3;
    const CHUNKS: usize = 8;
    const CHUNK: usize = 200; // ops per checkpoint (no readback in between)

    for c in 0..CHUNKS {
        let mut expected_cpu: Vec<(u32, u32)> = Vec::with_capacity(CHUNK);
        let mut results: Vec<Tensor> = Vec::with_capacity(CHUNK);
        let mut inputs: Vec<Tensor> = Vec::with_capacity(CHUNK);
        for i in 0..CHUNK {
            let it = (c * CHUNK + i) as u64;
            let is_argmax = it.is_multiple_of(2); // mixed argmax/argmin
            let vals = make_vals(it);
            let xs = Tensor::from_slice(&vals, (2, NE0), &device)?;
            inputs.push(xs.clone());
            let r = if is_argmax {
                xs.argmax_keepdim(1)?
            } else {
                xs.argmin_keepdim(1)?
            };
            expected_cpu.push(expected_rows(&vals, NE0, is_argmax));
            results.push(r);
            // xs drops here -> src buffer recycle-eligible while the active
            // batch (or a submitted batch) may still reference it.
        }
        // Checkpoint: read back everything since the last checkpoint and
        // compare to the CPU reference.
        for (idx, r) in results.iter().enumerate() {
            let got = r.flatten_all()?.to_vec1::<u32>()?;
            let got_again = r.flatten_all()?.to_vec1::<u32>()?;
            let (e0, e1) = expected_cpu[idx];
            if got[0] != e0 || got[1] != e1 {
                let it = (c * CHUNK + idx) as u64;
                let vals = make_vals(it);
                // Is the GPU-side source intact?
                let src_gpu = match inputs[idx].flatten_all()?.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(e) => vec![f32::NAN, e.to_string().len() as f32],
                };
                // Recompute with the SAME op on the SAME retained input
                // (fully drained at checkpoint time): does a fresh dispatch
                // match?
                let argmax_original = ((c * CHUNK + idx) as u64).is_multiple_of(2);
                let recomputed = if argmax_original {
                    inputs[idx]
                        .argmax_keepdim(1)?
                        .flatten_all()?
                        .to_vec1::<u32>()?
                } else {
                    inputs[idx]
                        .argmin_keepdim(1)?
                        .flatten_all()?
                        .to_vec1::<u32>()?
                };
                candle_core::bail!(
                    "argmax/argmin churn mismatch chunk={c} op={idx}: got {got:?} \
                     got_again={got_again:?}, expected [{e0}, {e1}], \
                     recomputed={recomputed:?}, src_gpu={src_gpu:?}, vals={vals:?}"
                );
            }
        }
        eprintln!("chunk {c} clean");
    }
    Ok(())
}
