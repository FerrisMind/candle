//! Regression test for the wgpu raw-float GGML dequantize 2D-dispatch corruption
//! (OmniVoice gguf-bf16). `quantized_raw_float_dequantize_f32` dispatched its
//! inline F16/BF16 decode shader via `run_compute`, which splits workgroup
//! counts above `max_compute_workgroups_per_dimension` (65535) into
//! (wg_x, wg_y) — but the shader indexed only by `global_invocation_id.x`, so
//! every wg_y row re-decoded the same first wg_x*WG_SIZE elements and the rest
//! of the `alloc_uninit` destination stayed uninitialized garbage. A GGUF BF16
//! `llm.embed_tokens.weight` (151676x1024 = 155,316,224 elements) decoded only
//! its first 65535*256 = 16,776,960 elements; the rest of the embedding table
//! was uninitialized bytes -> off-text generation. The fix folds
//! `(wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x` like every other wgpu kernel.
//!
//! The test dequantizes a GGML BF16 tensor just above the 65535*256 element
//! cap and requires EVERY element to match the CPU reference (pre-fix, the
//! elements past the cap were uninitialized recycled-buffer bytes).
#![cfg(feature = "wgpu")]

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};

/// Just over the dispatch cap: 65535 * 256 = 16,776,960 usable elements when
/// the shader lacks the y-fold. 16,809,984 elements exercise rows wg_y >= 1.
const LARGE_ROWS: usize = 16416;
const LARGE_ELEMS: usize = LARGE_ROWS * 1024;

fn test_bf16_bytes(len: usize) -> Vec<u8> {
    // Alternating bf16 values so the half-word ordering is also exercised:
    // 1.5 = 0x3fc0, -2.25 = 0xc010 (little-endian half words per u32).
    let mut bytes = Vec::with_capacity(len * 2);
    for i in 0..len {
        let v: f32 = if i % 2 == 0 { 1.5 } else { -2.25 };
        let b = half::bf16::from_f32(v);
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    bytes
}

fn assert_f32_eq(got: &Tensor, want: &Tensor) -> Result<()> {
    let got_v = got.flatten_all()?.to_vec1::<f32>()?;
    let want_v = want.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got_v.len(), want_v.len());
    let mismatches: Vec<usize> = got_v
        .iter()
        .zip(want_v.iter())
        .enumerate()
        .filter_map(|(i, (g, w))| (g.to_bits() != w.to_bits()).then_some(i))
        .collect();
    assert!(
        mismatches.is_empty(),
        "wgpu dequantize has {} mismatched values, first at {mismatches:?}",
        mismatches.len()
    );
    Ok(())
}

#[test]
fn wgpu_ggml_bf16_dequantize_beyond_dispatch_cap_matches_cpu() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    let bytes = test_bf16_bytes(LARGE_ELEMS);
    let shape = (LARGE_ROWS, 1024);

    let qcpu = QStorage::from_data(
        std::borrow::Cow::Borrowed(&bytes),
        &Device::Cpu,
        GgmlDType::BF16,
    )?;
    let qt_cpu = QTensor::new(qcpu, shape)?;
    let want = qt_cpu.dequantize(&Device::Cpu)?;

    let qwgpu = QStorage::from_data(
        std::borrow::Cow::Owned(bytes),
        &device,
        GgmlDType::BF16,
    )?;
    let qt_wgpu = QTensor::new(qwgpu, shape)?;
    let got = qt_wgpu.dequantize(&device)?;
    assert_eq!(got.dtype(), DType::F32);
    assert_f32_eq(&got, &want)?;
    Ok(())
}