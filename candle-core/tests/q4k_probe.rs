//! Q4_K GPU-vs-CPU parity probe (campaign: "fix q4k").
//!
//! The model-level `gpu_model_matrix` quantized cases compare whole-model
//! logits and only report a composite metric, so they cannot tell whether the
//! deviation comes from the dequant kernel, the quantized matmul kernel, or the
//! activation-quantization contract (CPU quantizes the LHS to Q8K, the GPU
//! backends emulate it with Q8_1). This probe isolates each stage on real
//! Qwen3-0.6B GGUF tensors:
//!
//!   1. weight dequant: CPU vs GPU elementwise.
//!   2. matmul (m>1): GPU vs CPU vs an f32 reference built from CPU-dequantized
//!      weights; plus per-side comparisons against the two candidate activation
//!      contracts (Q8K and Q8_1) applied on the CPU.
//!   3. matvec (m=1): same comparisons for the decode path.
//!
//! Run (GPU lock mandatory):
//!   CANDLE_QWEN3_GGUF_PATH=G:/LM_STUD/.../Qwen3-0.6B-Q4_K_M.gguf \
//!   cargo test --release -p candle-core --features wgpu --test q4k_probe -- --nocapture

mod support;

use candle_core::quantized::gguf_file;
use candle_core::quantized::k_quants::{BlockQ8K, GgmlType};
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use support::{backend_device_or_skip, TestBackend};

fn gguf_path() -> PathBuf {
    std::env::var_os("CANDLE_QWEN3_GGUF_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("G:/LM_STUD/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
        })
}

/// Optional DC offset for the synthetic activation, so the per-block integer
/// sum (the k-quant min term) is non-zero and can be compared against the CPU.
fn lhs_bias() -> f32 {
    std::env::var("Q4K_PROBE_LHS_BIAS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
}

/// Optional outlier: value written at the first element of every 256-element
/// block, mimicking the massive activations real models show.
fn lhs_spike() -> f32 {
    std::env::var("Q4K_PROBE_LHS_SPIKE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
}

fn lhs_value(i: usize) -> f32 {
    let base = ((i * 37 % 101) as f32 - 50.0) / 50.0 + lhs_bias();
    if i % 256 == 0 {
        lhs_spike()
    } else {
        base
    }
}

fn metrics(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?;
    let diff = (&a - &b)?;
    let max_abs = diff.abs()?.max_all()?.to_scalar::<f32>()?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let ref_pow = a.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let nmse = mse / ref_pow.max(1e-30);
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let cos = dot / (na * nb).max(1e-30);
    println!("    {label}: max_abs={max_abs:.4e} nmse={nmse:.4e} cos={cos:.7}");
    Ok(())
}

/// Quantize an f32 LHS row-block to `T` and dequantize back to f32 — the
/// activation contract the CPU reference (Q8K) and the GPU emulation (Q8_1)
/// apply before the dot.
fn lhs_roundtrip<T: GgmlType>(xs: &[f32]) -> Result<Vec<f32>> {
    let mut quant = vec![T::zeros(); xs.len() / T::BLCK_SIZE];
    T::from_float(xs, &mut quant);
    let mut out = vec![0f32; xs.len()];
    T::to_float(&quant, &mut out);
    Ok(out)
}

fn f32_ref(weights: &Tensor, lhs: &Tensor) -> Result<Tensor> {
    // QMatMul::forward(xs) computes xs @ W.t() with W of shape [out, in].
    lhs.matmul(&weights.t()?.contiguous()?)
}

fn probe_tensor(
    dev: &Device,
    content: &gguf_file::Content,
    file: &mut File,
    name: &str,
) -> Result<()> {
    let cpu = Device::Cpu;
    let info = content.tensor_infos.get(name).unwrap();
    println!(
        "\n== {name} {:?} {:?} ==",
        info.ggml_dtype,
        info.shape.dims()
    );
    let qt_cpu = Arc::new(content.tensor(file, name, &cpu)?);
    let qt_dev = Arc::new(content.tensor(file, name, dev)?);
    let (out_dim, in_dim) = qt_cpu.shape().dims2()?;

    // 1. dequant parity.
    let w_cpu = qt_cpu.dequantize(&cpu)?.to_dtype(DType::F32)?;
    let w_dev = qt_dev.dequantize(dev)?.to_dtype(DType::F32)?;
    let w_dev_cpu = w_dev.to_device(&cpu)?;
    metrics("dequant GPU-vs-CPU", &w_dev_cpu, &w_cpu)?;

    // deterministic lhs
    let m = 4usize;
    let lhs_vals: Vec<f32> = (0..m * in_dim).map(lhs_value).collect();
    let lhs_cpu = Tensor::from_slice(&lhs_vals, (m, in_dim), &cpu)?;
    let lhs_dev = Tensor::from_slice(&lhs_vals, (m, in_dim), dev)?;

    // 2. matmul (prefill path, m=4).
    let mm_cpu = QMatMul::from_arc(qt_cpu.clone())?.forward(&lhs_cpu)?;
    let mm_dev = QMatMul::from_arc(qt_dev.clone())?
        .forward(&lhs_dev)?
        .to_device(&cpu)?;
    let ref_f32 = f32_ref(&w_cpu, &lhs_cpu)?;
    println!("  -- matmul m={m} (out={out_dim}, k={in_dim}) --");
    metrics("CPU-vs-f32ref", &mm_cpu, &ref_f32)?;
    metrics("GPU-vs-f32ref", &mm_dev, &ref_f32)?;
    metrics("GPU-vs-CPU  ", &mm_dev, &mm_cpu)?;

    // activation-contract comparison: apply each candidate quantizer to the
    // lhs on the CPU, then f32-matmul with the CPU-dequantized weights.
    let q8k_vals = lhs_roundtrip::<BlockQ8K>(&lhs_vals)?;
    let q8k_ref = f32_ref(&w_cpu, &Tensor::from_slice(&q8k_vals, (m, in_dim), &cpu)?)?;
    metrics("CPU-vs-f32ref@Q8K", &mm_cpu, &q8k_ref)?;
    metrics("GPU-vs-f32ref@Q8K", &mm_dev, &q8k_ref)?;
    if info.ggml_dtype == GgmlDType::Q4K
        || info.ggml_dtype == GgmlDType::Q5K
        || info.ggml_dtype == GgmlDType::Q6K
    {
        use candle_core::quantized::k_quants::BlockQ8_1;
        let q81_vals = lhs_roundtrip::<BlockQ8_1>(&lhs_vals)?;
        let q81_ref = f32_ref(&w_cpu, &Tensor::from_slice(&q81_vals, (m, in_dim), &cpu)?)?;
        metrics("CPU-vs-f32ref@Q8_1", &mm_cpu, &q81_ref)?;
        metrics("GPU-vs-f32ref@Q8_1", &mm_dev, &q81_ref)?;
    }

    // 3. matvec (decode path, m=1).
    let lhs1_cpu = lhs_cpu.narrow(0, 0, 1)?.contiguous()?;
    let lhs1_dev = lhs_dev.narrow(0, 0, 1)?.contiguous()?;
    let mv_cpu = QMatMul::from_arc(qt_cpu.clone())?.forward(&lhs1_cpu)?;
    let mv_dev = QMatMul::from_arc(qt_dev.clone())?
        .forward(&lhs1_dev)?
        .to_device(&cpu)?;
    let ref1_f32 = f32_ref(&w_cpu, &lhs1_cpu)?;
    println!("  -- matvec m=1 --");
    metrics("CPU-vs-f32ref", &mv_cpu, &ref1_f32)?;
    metrics("GPU-vs-f32ref", &mv_dev, &ref1_f32)?;
    metrics("GPU-vs-CPU  ", &mv_dev, &mv_cpu)?;

    Ok(())
}

fn nmse_only(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?;
    let mse = (&a - &b)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let pw = b.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok((mse as f64) / (pw as f64).max(1e-30))
}

/// Sweep every 2-D quantized tensor in the file and report the CPU-vs-GPU nmse
/// for both the multi-column (prefill) and the single-column (decode) shapes.
fn probe_all(dev: &Device, content: &gguf_file::Content, file: &mut File) -> Result<()> {
    let cpu = Device::Cpu;
    let mut names: Vec<String> = content
        .tensor_infos
        .iter()
        .filter(|(_, i)| i.shape.dims().len() == 2)
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    let mut worst_mm = (String::new(), 0f64);
    let mut worst_mv = (String::new(), 0f64);
    for name in names {
        let info = content.tensor_infos.get(&name).unwrap();
        let qt_cpu = Arc::new(content.tensor(file, &name, &cpu)?);
        let qt_dev = Arc::new(content.tensor(file, &name, dev)?);
        let (_out_dim, in_dim) = qt_cpu.shape().dims2()?;
        let m = 4usize;
        let lhs_vals: Vec<f32> = (0..m * in_dim).map(lhs_value).collect();
        let lhs_cpu = Tensor::from_slice(&lhs_vals, (m, in_dim), &cpu)?;
        let lhs_dev = Tensor::from_slice(&lhs_vals, (m, in_dim), dev)?;
        let mm_cpu = QMatMul::from_arc(qt_cpu.clone())?.forward(&lhs_cpu)?;
        let mm_dev = QMatMul::from_arc(qt_dev.clone())?
            .forward(&lhs_dev)?
            .to_device(&cpu)?;
        let mm = nmse_only(&mm_dev, &mm_cpu)?;
        let l1_cpu = lhs_cpu.narrow(0, 0, 1)?.contiguous()?;
        let l1_dev = lhs_dev.narrow(0, 0, 1)?.contiguous()?;
        let mv_cpu = QMatMul::from_arc(qt_cpu.clone())?.forward(&l1_cpu)?;
        let mv_dev = QMatMul::from_arc(qt_dev.clone())?
            .forward(&l1_dev)?
            .to_device(&cpu)?;
        let mv = nmse_only(&mv_dev, &mv_cpu)?;
        println!(
            "    {name} {:?} {:?} mm={mm:.3e} mv={mv:.3e}",
            info.ggml_dtype,
            info.shape.dims()
        );
        if mm > worst_mm.1 {
            worst_mm = (name.clone(), mm);
        }
        if mv > worst_mv.1 {
            worst_mv = (name.clone(), mv);
        }
    }
    println!("  worst mm: {} {:.3e}", worst_mm.0, worst_mm.1);
    println!("  worst mv: {} {:.3e}", worst_mv.0, worst_mv.1);
    Ok(())
}

fn run(dev: &Device, tensors: &[String]) -> Result<()> {
    let path = gguf_path();
    println!("GGUF {path:?} on {:?}", dev);
    let mut file = File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    if std::env::var_os("Q4K_PROBE_ALL").is_some() {
        return probe_all(dev, &content, &mut file);
    }
    // Default set: one tensor per quant family present in the Q4_K_M file.
    let mut names: Vec<String> = Vec::new();
    if tensors.is_empty() {
        // One 2-D tensor per quant family, plus the first two Q4K tensors
        // (the failing family) so shape-dependence is visible.
        let mut seen = std::collections::BTreeSet::new();
        let mut q4k_count = 0usize;
        for (name, info) in content.tensor_infos.iter() {
            if info.shape.dims().len() != 2 {
                continue;
            }
            let key = format!("{:?}", info.ggml_dtype);
            let first_of_family = seen.insert(key);
            let q4k_extra = info.ggml_dtype == GgmlDType::Q4K && q4k_count < 3;
            if q4k_extra {
                q4k_count += 1;
            }
            if first_of_family || q4k_extra {
                names.push(name.clone());
            }
        }
        names.sort();
        names.dedup();
    } else {
        names = tensors.to_vec();
    }
    for name in &names {
        probe_tensor(dev, &content, &mut file, name)?;
    }
    Ok(())
}

fn names_from_env() -> Vec<String> {
    std::env::var("Q4K_PROBE_TENSORS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default()
}

#[cfg(feature = "wgpu")]
#[test]
fn q4k_probe_wgpu() -> Result<()> {
    let Some(dev) = backend_device_or_skip("q4k_probe_wgpu", TestBackend::Wgpu)? else {
        return Ok(());
    };
    run(&dev, &names_from_env())
}

#[cfg(feature = "vulkan")]
#[test]
fn q4k_probe_vulkan() -> Result<()> {
    let Some(dev) = backend_device_or_skip("q4k_probe_vulkan", TestBackend::Vulkan)? else {
        return Ok(());
    };
    run(&dev, &names_from_env())
}
