//! Runtime CPU-fallback counter audit for Vulkan / native WebGPU.
//!
//! Resets counters, runs dense + quantized GPU workloads on the real shipped
//! paths, then asserts counters remain zero. Intentional host APIs (to_cpu /
//! quantize-from-float host encode matching CUDA) are not counted as compute
//! fallbacks by the production counters.
//!
//! ```text
//! cargo run -p candle-core --release --features "vulkan,wgpu" --example fallback_runtime_audit
//! ```
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

fn audit_backend(name: &str, device: &Device) -> Result<()> {
    match name {
        "vulkan" => candle_core::reset_vulkan_cpu_fallback_count(),
        "wgpu" => candle_core::reset_wgpu_cpu_fallback_count(),
        _ => {}
    }
    let before = match name {
        "vulkan" => candle_core::vulkan_cpu_fallback_count(),
        "wgpu" => candle_core::wgpu_cpu_fallback_count(),
        _ => 0,
    };
    println!("[{name}] fallback_count after reset: {before}");

    // Dense hot path
    let a = Tensor::randn(0f32, 1.0, (256, 512), device)?;
    let b = Tensor::randn(0f32, 1.0, (512, 128), device)?;
    let c = a.matmul(&b)?;
    let d = (c.relu()? + &a.narrow(1, 0, 128)?)?;
    let _ = d.sum_all()?;
    let e = a.transpose(0, 1)?.contiguous()?;
    let _ = e.matmul(&Tensor::randn(0f32, 1.0, (256, 64), device)?)?;

    // Quantized path (GGUF-style): host-side pack matches CUDA baseline, then GPU qmatmul
    let weights = Tensor::randn(0f32, 1.0, (256, 128), device)?;
    let q = QTensor::quantize(&weights, GgmlDType::Q4_0)?;
    let deq = q.dequantize(device)?;
    let act = Tensor::randn(0f32, 1.0, (4, 128), device)?;
    let qmm = QMatMul::from_qtensor(q)?;
    let out = qmm.forward(&act)?;
    device.synchronize()?;
    let _ = (deq.dtype(), out.dtype());

    let after = match name {
        "vulkan" => candle_core::vulkan_cpu_fallback_count(),
        "wgpu" => candle_core::wgpu_cpu_fallback_count(),
        _ => 0,
    };
    println!("[{name}] fallback_count after dense+quant workload: {after}");
    if after != 0 {
        candle_core::bail!("{name}: expected 0 CPU fallbacks, got {after}");
    }
    println!("[{name}] PASS: zero CPU fallbacks on dense+quant GPU path");
    Ok(())
}

fn main() -> Result<()> {
    println!("fallback_runtime_audit: start");
    #[cfg(feature = "vulkan")]
    {
        let vk = Device::new_vulkan(0)?;
        println!("[vulkan] device ok: {:?}", vk);
        audit_backend("vulkan", &vk)?;
    }
    #[cfg(feature = "wgpu")]
    {
        let wg = Device::new_wgpu(0)?;
        println!("[wgpu] device ok: {:?}", wg);
        audit_backend("wgpu", &wg)?;
    }
    println!("fallback_runtime_audit: ALL PASS");
    Ok(())
}
