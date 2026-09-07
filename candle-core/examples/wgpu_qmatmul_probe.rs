use candle_core::quantized::{self, GgmlDType, QMatMul};
use candle_core::{DType, Device, Tensor, Module};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_wgpu(0)?;
    let (m, n, k) = (1usize, 1024usize, 1024usize);
    let lhs_v: Vec<f32> = (0..m * k).map(|v| v as f32 / (m * k) as f32).collect();
    let rhs_v: Vec<f32> = (0..k * n).map(|v| v as f32 / (n * k) as f32).collect();
    let lhs = Tensor::from_slice(&lhs_v, (m, k), &dev)?;
    let rhs = Tensor::from_slice(&rhs_v, (k, n), &dev)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, GgmlDType::F32)?;
    let matmul = QMatMul::from_qtensor(qtensor)?;
    for qdtype in [GgmlDType::F32, GgmlDType::F16, GgmlDType::Q8_0] {
        let qt = quantized::QTensor::quantize(&rhs.t()?, qdtype)?;
        let mm = QMatMul::from_qtensor(qt)?;
        let _ = mm.forward(&lhs)?; // warm
        let t0 = Instant::now();
        let iters = 50;
        for _ in 0..iters {
            let _ = mm.forward(&lhs)?;
        }
        dev.synchronize()?;
        let dt = t0.elapsed() / iters as u32;
        println!("qmatmul {qdtype:?}: {dt:?}");
    }
    // Break down: dequantize + dense matmul (the fallback path).
    let qt = quantized::QTensor::quantize(&rhs.t()?, GgmlDType::F32)?;
    let dense = qt.dequantize(&dev)?;
    let w = dense.t()?.to_dtype(DType::F32)?;
    let _ = lhs.matmul(&w)?;
    for mm_ in [1usize, 16, 64, 1024] {
        let lhs_t = Tensor::from_vec((0..mm_ * k).map(|v| v as f32 / 1024.0).collect(), (mm_, k), &dev)?;
        let _ = lhs_t.matmul(&w)?;
        let t0 = Instant::now();
        for _ in 0..30 {
            let _ = lhs_t.matmul(&w)?;
        }
        dev.synchronize()?;
        println!("dense matmul m={mm_}: {:?}", t0.elapsed() / 30);
    }
    let _ = (&matmul, &n, DType::F32);
    Ok(())
}
