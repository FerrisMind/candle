//! WASM resolve-mode tests for `candle-wasm-device-select`.
//!
//! Differential F32 tolerances for slice-1 comparisons (CPU ↔ portable WebGPU when
//! evidence exists): rtol `1e-5`, atol `1e-6`. Do **not** treat a green resolve
//! smoke as `portable_webgpu_status: Verified`.

use candle_wasm_device_select::{DeviceMode, ResolvedKind};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// F32 relative tolerance for slice-1 differential checks (spec §8).
#[allow(dead_code)]
const F32_RTOL: f64 = 1e-5;
/// F32 absolute tolerance for slice-1 differential checks (spec §8).
#[allow(dead_code)]
const F32_ATOL: f64 = 1e-6;

#[wasm_bindgen_test]
async fn resolve_cpu() {
    let r = DeviceMode::Cpu
        .resolve()
        .await
        .expect("Cpu resolve must succeed");
    assert_eq!(r.resolved, ResolvedKind::Cpu);
    assert!(r.adapter_name.is_none());
    assert!(r.device.is_cpu());
}

#[wasm_bindgen_test]
async fn resolve_auto_never_panics() {
    let r = DeviceMode::Auto
        .resolve()
        .await
        .expect("Auto resolve must return Ok (Cpu or Wgpu), never panic");
    assert!(matches!(
        r.resolved,
        ResolvedKind::Cpu | ResolvedKind::Wgpu
    ));
}

/// Explicit `wgpu`: Ok if adapter works, Err if unavailable / feature off.
/// Must not panic either way.
#[wasm_bindgen_test]
async fn resolve_wgpu() {
    let result = DeviceMode::Wgpu.resolve().await;
    match result {
        Ok(r) => {
            assert_eq!(r.resolved, ResolvedKind::Wgpu);
            assert!(r.device.is_wgpu());
        }
        Err(err) => {
            // Expected when feature off, no adapter, or create path unavailable.
            let msg = format!("{err}");
            assert!(
                !msg.is_empty(),
                "wgpu resolve Err should carry a non-empty message"
            );
        }
    }
}

/// Minimal CPU-only golden path (slice 1) when browser WebGPU differential
/// cannot run — exact integer-like F32 matmul on CPU, not a Verified claim.
#[wasm_bindgen_test]
fn cpu_f32_matmul_golden() {
    use candle::{Device, Tensor};

    let cpu = Device::Cpu;
    let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &cpu).expect("a");
    let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &cpu).expect("b");
    let c = a.matmul(&b).expect("matmul");
    let got = c.to_vec2::<f32>().expect("to_vec2");
    let expected = [[19f32, 22.], [43., 50.]];
    for (row_g, row_e) in got.iter().zip(expected.iter()) {
        for (&g, &e) in row_g.iter().zip(row_e.iter()) {
            let diff = (g as f64 - e as f64).abs();
            let tol = F32_ATOL + F32_RTOL * (e as f64).abs();
            assert!(
                diff <= tol,
                "cpu golden matmul out of tolerance: got {g}, expected {e}, diff {diff}, tol {tol}"
            );
        }
    }
}
