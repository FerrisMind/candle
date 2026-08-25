//! Isolation test for the Qwen3-16B-A3B MoE routing nondeterminism.
//!
//! The 16B wgpu fused-MoE path is bit-identical WITHIN a process but the
//! generated greedy output differs ACROSS separate processes (observed:
//! run1 numeric loop, run2 "capital of France is Paris" loop, run3 variant).
//! M2 pointed at the MoE-forward routing: `arg_sort_last_dim(false)` on the
//! [tokens, n_experts] routing weights followed by a `gather` + `sort_last_dim`
//! on the flattened top-k ids.
//!
//! This test drives the EXACT routing sort chain from a FIXED deterministic
//! input (no model weights) and prints a routing signature so the test binary
//! can be executed as SEPARATE PROCESSES and diffed. If the wgpu
//! arg-sort / sort ops are cross-process deterministic, the printed signature
//! is stable across process invocations. If they read uninitialized / recycled
//! buffer contents, the signature drifts.
//!
//! It also asserts the wgpu routing matches a CPU reference computed by the
//! exact same comparator, so an in-process correctness drift is caught too.
#![cfg(feature = "wgpu")]

use candle_core::{D, Device, Result, Tensor};

const N_EXPERTS: usize = 64;
const TOP_K: usize = 8;
const N_TOKENS: usize = 13;

/// Deterministic orderable key transform matching argsort.wgsl (CUDA sort.cu
/// trick): raw float bits mapped to a u32 key preserving float order under
/// unsigned comparison.
fn orderable_f32(v: f32) -> u32 {
    let b = v.to_bits();
    let mask = if (b & 0x8000_0000) == 0 {
        0x8000_0000
    } else {
        0xFFFF_FFFF
    };
    b ^ mask
}

/// CPU arg-sort (descending) with the exact wgpu comparator: primary key, then
/// ascending source index (stable). Returns the sorted indices (desc).
fn cpu_argsort_desc(vals: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..vals.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let ka = orderable_f32(vals[a as usize]);
        let kb = orderable_f32(vals[b as usize]);
        // desc: larger key first; ties -> smaller original index first
        kb.cmp(&ka).then(a.cmp(&b))
    });
    idx
}

/// Build a deterministic [N_TOKENS, N_EXPERTS] f32 router-logits matrix with
/// realistic softmax-concentrating scores plus a few deliberate exact ties so
/// that stable tie-breaking is exercised.
fn make_logits() -> Vec<f32> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut v = Vec::with_capacity(N_TOKENS * N_EXPERTS);
    for t in 0..N_TOKENS {
        let base = (t as f32) * 0.001; // tiny per-token drift
        for e in 0..N_EXPERTS {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((s >> 13) % 10_000) as f32 / 10_000.0;
            // a couple of dominant experts per token
            let dominant = if e % 17 == 0 { 3.0 } else { 0.0 };
            let mut val = base + r * 0.4 + dominant;
            // exact tie between two adjacent experts in some tokens
            if e == 5 && t % 3 == 0 {
                val = 1.2345;
            }
            if e == 6 && t % 3 == 0 {
                val = 1.2345; // exact tie with expert 5
            }
            v.push(val);
        }
    }
    v
}

fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let max = t.max_keepdim(D::Minus1)?;
    let e = t.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(D::Minus1)?;
    e.broadcast_div(&sum)
}

#[test]
fn wgpu_moe_routing_determinism() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };

    let logits = make_logits();
    let logits_t = Tensor::from_slice(&logits, (N_TOKENS, N_EXPERTS), &device)?;
    let routing = softmax_last_dim(&logits_t)?;

    // --- the exact model routing chain ---
    let topk_ids = routing
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, TOP_K)?
        .contiguous()?;
    let mut topk_w = routing.gather(&topk_ids, D::Minus1)?;
    topk_w = topk_w.broadcast_div(&topk_w.sum_keepdim(D::Minus1)?)?;
    let (sorted_ids, sort_idx) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    // --- readback for the printed cross-process signature ---
    let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
    let sorted_vec = sorted_ids.flatten_all()?.to_vec1::<u32>()?;
    let sort_idx_vec = sort_idx.flatten_all()?.to_vec1::<u32>()?;
    let w_vec = topk_w.flatten_all()?.to_vec1::<f32>()?;

    // One lynch-pin row: make the signature compact & diff-able.
    let row0: Vec<u32> = ids_vec[0..TOP_K].to_vec();
    let sig = format!(
        "ROUTING_SIG ids0={row0:?} sorted={sorted_vec:?} sidx={sort_idx_vec:?} w0={:.6?}",
        row0_w(&w_vec)
    );
    println!("{sig}");

    // --- in-process correctness check against the CPU comparator ---
    // Compare per-token top-k ids to CPU desc arg-sort (stable) of routing w.
    let cpu_routing: Vec<f32> = routing.flatten_all()?.to_vec1::<f32>()?;
    for t in 0..N_TOKENS {
        let row = &cpu_routing[t * N_EXPERTS..(t + 1) * N_EXPERTS];
        let cpu_idx = cpu_argsort_desc(row);
        let gpu_row: Vec<u32> = ids_vec[t * TOP_K..(t + 1) * TOP_K].to_vec();
        if gpu_row != cpu_idx[0..TOP_K] {
            candle_core::bail!(
                "token {t}: wgpu topk {gpu_row:?} != cpu topk {:?}\nfull cpu order: {cpu_idx:?}",
                &cpu_idx[0..TOP_K]
            );
        }
    }

    Ok(())
}

fn row0_w(w: &[f32]) -> Vec<f32> {
    // just first TOP_K entries (row 0)
    w[0..TOP_K].to_vec()
}
