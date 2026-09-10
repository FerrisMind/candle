//! Regression test for the wgpu `copy2d` unaligned-copy corruption (OmniVoice
//! stage0). `copy2d`'s raw command path issued `copy_buffer_to_buffer` commands
//! whose offsets and size are not multiples of COPY_BUFFER_ALIGNMENT (4) — e.g.
//! a U8 `(1,114)+(1,94)` cat copies 114 and 94 bytes at byte offset 114 — which
//! wgpu VALIDATES AND DROPS, silently leaving stale recycled buffer bytes in the
//! destination (`Copy size 94 does not respect COPY_BUFFER_ALIGNMENT` /
//! `Buffer offset 114 is not aligned`). The fix routes unaligned `copy2d` calls
//! through per-row `copy_strided_src`, which already guards alignment and has a
//! copy/emulated shader fallback for every dtype. U8 is the minimal case because
//! its byte size equals the element count.
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Result, Tensor};

fn test_values_u8(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u8).wrapping_mul(37).wrapping_add(salt)) % 251)
        .collect()
}

fn assert_u8_eq(got: &Tensor, want: &Tensor) -> Result<()> {
    let got_v = got.flatten_all()?.to_vec1::<u8>()?;
    let want_v = want.flatten_all()?.to_vec1::<u8>()?;
    assert_eq!(got_v.len(), want_v.len());
    let mismatches: Vec<usize> = got_v
        .iter()
        .zip(want_v.iter())
        .enumerate()
        .filter_map(|(i, (g, w))| (g != w).then_some(i))
        .collect();
    assert!(
        mismatches.is_empty(),
        "wgpu result has {} mismatched values, first at {mismatches:?}",
        mismatches.len()
    );
    Ok(())
}

/// Minimal unaligned case: cat along dim 1 of U8 (1,114) + (1,94). Both pieces
/// violate COPY_BUFFER_ALIGNMENT (size 114/94, dst offset 114) and were dropped
/// as no-ops before the guard, leaving stale bytes at index >= 114.
#[test]
fn wgpu_cat_u8_odd_pieces_matches_cpu() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    let a = Tensor::from_slice(&test_values_u8(114, 1), (1, 114), &device)?;
    let b = Tensor::from_slice(&test_values_u8(94, 2), (1, 94), &device)?;
    let got = Tensor::cat(&[&a, &b], 1)?;

    let ref_a = Tensor::from_slice(&test_values_u8(114, 1), (1, 114), &Device::Cpu)?;
    let ref_b = Tensor::from_slice(&test_values_u8(94, 2), (1, 94), &Device::Cpu)?;
    let want = Tensor::cat(&[&ref_a, &ref_b], 1)?;
    assert_eq!(got.dtype(), DType::U8);
    assert_u8_eq(&got, &want)?;
    Ok(())
}

/// Production shape (OmniVoice stage0 `slice_assign`/`pad_with_zeros`): cat
/// along dim 2 of rank-3 (1,8,·) U8 mask tensors — d1=8 strided rows of d2=114
/// and d2=94 bytes, every row command unaligned.
#[test]
fn wgpu_cat_u8_rank3_strided_rows_matches_cpu() -> Result<()> {
    let device = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("wgpu unavailable: {e}; skipping");
            return Ok(());
        }
    };
    let a = Tensor::from_slice(&test_values_u8(8 * 114, 3), (1, 8, 114), &device)?;
    let b = Tensor::from_slice(&test_values_u8(8 * 94, 4), (1, 8, 94), &device)?;
    let got = Tensor::cat(&[&a, &b], 2)?;

    let ref_a = Tensor::from_slice(&test_values_u8(8 * 114, 3), (1, 8, 114), &Device::Cpu)?;
    let ref_b = Tensor::from_slice(&test_values_u8(8 * 94, 4), (1, 8, 94), &Device::Cpu)?;
    let want = Tensor::cat(&[&ref_a, &ref_b], 2)?;
    assert_u8_eq(&got, &want)?;

    // Also exercise the slice_assign path that pads with zeros through the same
    // cat machinery (mask tensor assignment at the failing 114 offset).
    let src = Tensor::from_slice(&test_values_u8(94, 5), (1, 94), &device)?;
    let mut base = Tensor::from_slice(&test_values_u8(114, 6), (1, 114), &device)?;
    let updated = base.slice_assign(&[(0..1), (20..114)], &src)?;
    let ref_base = Tensor::from_slice(&test_values_u8(114, 6), (1, 114), &Device::Cpu)?;
    let ref_src = Tensor::from_slice(&test_values_u8(94, 5), (1, 94), &Device::Cpu)?;
    let ref_updated = ref_base.slice_assign(&[(0..1), (20..114)], &ref_src)?;
    assert_u8_eq(&updated, &ref_updated)?;
    Ok(())
}
