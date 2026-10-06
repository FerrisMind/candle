mod support;

use candle::quantized::gguf_file;
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    quantized_qwen3_moe::GGUFQWenMoE as Qwen3Moe,
    quantized_recurrent_gemma::Model as QuantRecurrentGemma,
    recurrent_gemma::{Config as RecurrentGemmaConfig, TemporalBlockType},
    rwkv_v7::{Config as RwkvV7Config, Model as RwkvV7, ModelVersion, State as RwkvV7State},
};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{backend_fallback_count, native_required, TestBackend};

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const CASE_FILTER_ENV: &str = "CANDLE_GPU_SEQ_QUANT_CASE_FILTER";

// Tolerances: dense F32 sequential model uses the same 5e-2 convention as the
// base gpu_model_matrix dense decoder cases; quantized (GGUF) cases use the
// campaign-approved 1e-2/1e-2 with argmax + top-5 recorded in the log.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const DENSE_ATOL: f32 = 5e-2;
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const DENSE_RTOL: f32 = 5e-2;
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const QUANT_ATOL: f32 = 1e-2;
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const QUANT_RTOL: f32 = 1e-2;

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
type ModelCaseFn = fn(&Device) -> Result<()>;

#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_seq_quant_ext_cuda() -> Result<()> {
    native_required(
        "gpu_model_matrix_seq_quant_ext_cuda",
        TestBackend::Cuda,
        run_seq_quant_matrix,
    )
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_seq_quant_ext_wgpu() -> Result<()> {
    native_required(
        "gpu_model_matrix_seq_quant_ext_wgpu",
        TestBackend::Wgpu,
        run_seq_quant_matrix,
    )
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_seq_quant_ext_vulkan() -> Result<()> {
    native_required(
        "gpu_model_matrix_seq_quant_ext_vulkan",
        TestBackend::Vulkan,
        run_seq_quant_matrix,
    )
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn run_seq_quant_matrix(device: &Device) -> Result<()> {
    if cfg!(debug_assertions) {
        println!(
            "seq/quant gpu model matrix is running in the debug test profile; for certification runtime use `cargo test --release`"
        );
    }

    let requested_cases = requested_case_names();
    let cases: [(&str, ModelCaseFn); 3] = [
        ("rwkv7_case", rwkv7_case),
        (
            "quantized_recurrent_gemma_case",
            quantized_recurrent_gemma_case,
        ),
        ("quantized_qwen3_moe_case", quantized_qwen3_moe_case),
    ];
    let mut failed = Vec::new();
    let mut ran_any = false;
    for (name, case_fn) in cases {
        if !case_is_requested(name, requested_cases.as_deref()) {
            println!(
                "skipping {name} on {} due to {CASE_FILTER_ENV}",
                backend_name(device)
            );
            continue;
        }
        ran_any = true;
        println!("running {name} on {}", backend_name(device));
        let start = Instant::now();
        match case_fn(device) {
            Ok(()) => println!(
                "{name} PASS in {:.2?}; cumulative fallback count after {name}: {}",
                start.elapsed(),
                backend_fallback_count(backend(device)),
            ),
            Err(err) => {
                println!(
                    "{name} FAIL in {:.2?}; cumulative fallback count after {name}: {}",
                    start.elapsed(),
                    backend_fallback_count(backend(device)),
                );
                println!("{name} error: {err}");
                failed.push(name);
            }
        }
    }
    if !ran_any {
        candle::bail!(
            "{CASE_FILTER_ENV} did not match any seq/quant GPU model case: rwkv7_case, quantized_recurrent_gemma_case, quantized_qwen3_moe_case"
        );
    }
    if !failed.is_empty() {
        candle::bail!(
            "seq/quant gpu model matrix: {} case(s) failed on {}: {:?} (see per-case output above)",
            failed.len(),
            backend_name(device),
            failed
        );
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn backend(device: &Device) -> TestBackend {
    if device.is_cuda() {
        TestBackend::Cuda
    } else if device.is_wgpu() {
        TestBackend::Wgpu
    } else if device.is_vulkan() {
        TestBackend::Vulkan
    } else {
        TestBackend::Cuda
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn backend_name(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_wgpu() {
        "wgpu"
    } else if device.is_vulkan() {
        "vulkan"
    } else {
        "cpu"
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn requested_case_names() -> Option<Vec<String>> {
    let value = std::env::var(CASE_FILTER_ENV).ok()?;
    let cases = value
        .split(',')
        .map(str::trim)
        .filter(|case_name| !case_name.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        None
    } else {
        Some(cases)
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn case_is_requested(name: &str, requested_cases: Option<&[String]>) -> bool {
    match requested_cases {
        None => true,
        Some(requested_cases) => requested_cases.iter().any(|requested| requested == name),
    }
}

// ─── Parity helper ───────────────────────────────────────────────────────────

/// Summary of a CPU-vs-GPU logits comparison. Printed for every comparison so
/// the certification log keeps argmax / top-5 evidence even when the strict
/// elementwise gate passes or fails.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
struct LogitsReport {
    elems: usize,
    max_diff: f32,
    max_diff_idx: usize,
    max_rel: f32,
    cosine: f64,
    nmse: f64,
    argmax_dev: usize,
    argmax_cpu: usize,
    top5_overlap: usize,
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
impl std::fmt::Display for LogitsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "elems={} max_diff={:.6} (idx {}) max_rel={:.6} cosine={:.6} nmse={:.6} argmax_dev={} argmax_cpu={} argmax_equal={} top5_overlap={}/5",
            self.elems,
            self.max_diff,
            self.max_diff_idx,
            self.max_rel,
            self.cosine,
            self.nmse,
            self.argmax_dev,
            self.argmax_cpu,
            self.argmax_dev == self.argmax_cpu,
            self.top5_overlap
        )
    }
}

/// Compare CPU-vs-GPU logits, always print the full report, and record a
/// violation message instead of failing immediately so that later comparisons
/// in the same case still produce evidence. The caller bails at case end if
/// any violation was recorded.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn compare_and_record(
    dev_logits: &Tensor,
    cpu_logits: &Tensor,
    atol: f32,
    rtol: f32,
    label: &str,
    violations: &mut Vec<String>,
) -> Result<()> {
    if dev_logits.dims() != cpu_logits.dims() {
        candle::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            dev_logits.dims(),
            cpu_logits.dims()
        );
    }
    let dev = dev_logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cpu = cpu_logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut max_diff = 0f32;
    let mut max_diff_idx = 0usize;
    let mut max_rel = 0f32;
    let mut mse_diff = 0f64;
    let mut mse_ref = 0f64;
    let mut dot = 0f64;
    let mut dev_norm = 0f64;
    let mut cpu_norm = 0f64;
    let mut first_violation: Option<(usize, f32, f32, f32)> = None;
    for (idx, (&d, &c)) in dev.iter().zip(cpu.iter()).enumerate() {
        let diff = (d - c).abs();
        let rel = diff / d.abs().max(c.abs()).max(1.0);
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = idx;
        }
        max_rel = max_rel.max(rel);
        let diff64 = d as f64 - c as f64;
        mse_diff += diff64 * diff64;
        mse_ref += c as f64 * c as f64;
        dot += d as f64 * c as f64;
        dev_norm += d as f64 * d as f64;
        cpu_norm += c as f64 * c as f64;
        let tol = atol + rtol * d.abs().max(c.abs());
        if diff > tol && first_violation.is_none() {
            first_violation = Some((idx, d, c, diff));
        }
    }
    let nmse = if mse_ref > 0.0 { mse_diff / mse_ref } else { 0.0 };
    let cosine = if dev_norm > 0.0 && cpu_norm > 0.0 {
        dot / (dev_norm.sqrt() * cpu_norm.sqrt())
    } else {
        1.0
    };
    let argmax_dev = argmax_index(&dev);
    let argmax_cpu = argmax_index(&cpu);
    let top5_overlap = topk_overlap(&dev, &cpu, 5);
    let report = LogitsReport {
        elems: dev.len(),
        max_diff,
        max_diff_idx,
        max_rel,
        cosine,
        nmse,
        argmax_dev,
        argmax_cpu,
        top5_overlap,
    };
    println!("{label}: {report}");
    if let Some((idx, d, c, diff)) = first_violation {
        violations.push(format!(
            "{label}: elementwise parity gate failed (atol={atol} rtol={rtol}): first violation at idx {idx}: dev={d} cpu={c} diff={diff}; report: {report}"
        ));
    }
    Ok(())
}

/// Fail the case with every recorded violation, so a single case still yields
/// complete evidence when multiple comparisons diverge.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn finish_case(case_name: &str, violations: &[String]) -> Result<()> {
    if violations.is_empty() {
        return Ok(());
    }
    candle::bail!(
        "{case_name}: {} comparison(s) failed:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn argmax_index(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn topk_overlap(dev: &[f32], cpu: &[f32], k: usize) -> usize {
    fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
    let dev_topk = topk_indices(dev, k);
    let cpu_topk = topk_indices(cpu, k);
    dev_topk
        .iter()
        .filter(|idx| cpu_topk.contains(idx))
        .count()
}

// ─── Case 1: RWKV v7 (safetensors, dense, sequential state) ─────────────────

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn rwkv7_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let weights_path = rwkv7_weights_path()?;
    println!("rwkv7 weights {weights_path:?} on {}", backend_name(device));
    let config = rwkv7_g1d_0_1b_config();

    let cpu_model = load_rwkv7_model(&weights_path, &config, &cpu)?;
    let dev_model = load_rwkv7_model(&weights_path, &config, device)?;
    let mut violations = Vec::new();

    // Prefill the fixed prompt [1, 8] through the sequential forward_seq path.
    let token_ids = [1u32, 8];
    let mut cpu_state = RwkvV7State::new(&config, &cpu)?;
    let mut dev_state = RwkvV7State::new(&config, device)?;
    let seq_cpu = cpu_model.forward_seq(&token_ids, &mut cpu_state)?;
    let seq_dev = dev_model.forward_seq(&token_ids, &mut dev_state)?;
    compare_and_record(
        &seq_dev,
        &seq_cpu,
        DENSE_ATOL,
        DENSE_RTOL,
        "rwkv7_forward_seq_logits",
        &mut violations,
    )?;

    // Single-token recurrent decode step on top of the advanced state.
    let next_token = 8u32;
    let next_cpu = Tensor::new(&[[next_token]], &cpu)?;
    let next_dev = Tensor::new(&[[next_token]], device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, &mut cpu_state, &[next_token])?;
    let decode_dev = dev_model.forward(&next_dev, &mut dev_state, &[next_token])?;
    compare_and_record(
        &decode_dev,
        &decode_cpu,
        DENSE_ATOL,
        DENSE_RTOL,
        "rwkv7_decode_logits",
        &mut violations,
    )?;
    finish_case("rwkv7_case", &violations)
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn rwkv7_weights_path() -> Result<PathBuf> {
    let dir = match std::env::var_os("CANDLE_RWKV7_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(r"G:\models\rwkv7-g1d-0.1b"),
    };
    let path = dir.join("rwkv7-g1d-0.1b-20260129-ctx8192.safetensors");
    if path.is_file() {
        Ok(path)
    } else {
        candle::bail!("rwkv7_case: weights not found at {path:?} (set CANDLE_RWKV7_DIR)")
    }
}

/// Built-in config for rwkv7-g1d-0.1b, mirroring the rwkv example's
/// `Which::Rwkv7G1d0_1b.v7_config()` (v7 models need no config.json).
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn rwkv7_g1d_0_1b_config() -> RwkvV7Config {
    RwkvV7Config {
        version: ModelVersion::V7,
        vocab_size: 65536,
        hidden_size: 768,
        num_hidden_layers: 12,
        head_size: 64,
        intermediate_size: None, // defaults to hidden_size * 4
        rescale_every: 0,
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn load_rwkv7_model(path: &Path, config: &RwkvV7Config, device: &Device) -> Result<RwkvV7> {
    // Source weights are BF16 on disk; convert to F32 on load for CPU/GPU parity.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    RwkvV7::new(config, vb)
}

// ─── Case 2: quantized RecurrentGemma 2B (GGUF Q4_K) ────────────────────────

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn quantized_recurrent_gemma_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let gguf_path = recurrent_gemma_gguf_path()?;
    let config = load_recurrent_gemma_config()?;
    println!(
        "recurrent-gemma gguf {gguf_path:?} (layers={}) on {}",
        config.num_hidden_layers,
        backend_name(device)
    );

    let mut cpu_model = load_quant_recurrent_gemma(&gguf_path, &config, &cpu)?;
    let mut dev_model = load_quant_recurrent_gemma(&gguf_path, &config, device)?;
    let mut violations = Vec::new();

    // Fixed prompt tokens [1, 8], single prefill forward at pos 0.
    let ids = [1u32, 8];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
    let prefill_dev = dev_model.forward(&ids_dev, 0)?;
    compare_and_record(
        &prefill_dev,
        &prefill_cpu,
        QUANT_ATOL,
        QUANT_RTOL,
        "recurrent_gemma_quantized_prefill_logits",
        &mut violations,
    )?;

    // Single-token recurrent decode step at pos 2.
    let next_token = [8u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
    let decode_dev = dev_model.forward(&next_dev, ids.len())?;
    compare_and_record(
        &decode_dev,
        &decode_cpu,
        QUANT_ATOL,
        QUANT_RTOL,
        "recurrent_gemma_quantized_decode_logits",
        &mut violations,
    )?;
    finish_case("quantized_recurrent_gemma_case", &violations)
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn recurrent_gemma_gguf_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("CANDLE_RECURRENT_GEMMA_GGUF") {
        return Ok(PathBuf::from(path));
    }
    let path = PathBuf::from(r"G:\models\recurrent-gemma-2b-q4k\recurrent-gemma-2b-q4k.gguf");
    if path.is_file() {
        Ok(path)
    } else {
        candle::bail!(
            "quantized_recurrent_gemma_case: GGUF not found at {path:?} (set CANDLE_RECURRENT_GEMMA_GGUF)"
        )
    }
}

/// Load the google/recurrentgemma-2b config: env-provided file first
/// (CANDLE_RECURRENT_GEMMA_CONFIG), then the local HF hub cache snapshot,
/// then an in-code fallback mirroring that exact config.json (no network).
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn load_recurrent_gemma_config() -> Result<RecurrentGemmaConfig> {
    if let Some(path) = std::env::var_os("CANDLE_RECURRENT_GEMMA_CONFIG") {
        let raw = std::fs::read_to_string(PathBuf::from(path))?;
        return serde_json::from_str(&raw)
            .map_err(|err| candle::Error::msg(format!("failed to parse recurrent gemma config: {err}")));
    }
    let snapshot = PathBuf::from(concat!(
        r"C:\Users\PC\.cache\huggingface\hub",
        r"\models--google--recurrentgemma-2b",
        r"\snapshots\3620f4ca9c5d16ee56c00180474a3201ec7f734a\config.json"
    ));
    if snapshot.is_file() {
        let raw = std::fs::read_to_string(&snapshot)?;
        return serde_json::from_str(&raw)
            .map_err(|err| candle::Error::msg(format!("failed to parse recurrent gemma config: {err}")));
    }
    println!(
        "recurrent gemma config.json not found at {snapshot:?}; using in-code google/recurrentgemma-2b values"
    );
    Ok(RecurrentGemmaConfig {
        num_hidden_layers: 26,
        vocab_size: 256000,
        hidden_size: 2560,
        intermediate_size: 15360,
        num_attention_heads: 10,
        num_key_value_heads: 1,
        head_dim: 256,
        lru_width: Some(2560),
        attention_window_size: 2048,
        conv1d_width: 4,
        logits_soft_cap: 30.0,
        hidden_activation: candle_nn::Activation::GeluPytorchTanh,
        partial_rotary_factor: 0.5,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        block_types: vec![
            TemporalBlockType::Recurrent,
            TemporalBlockType::Recurrent,
            TemporalBlockType::Attention,
        ],
        attention_bias: false,
        max_seq_len: 8192,
    })
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn load_quant_recurrent_gemma(
    path: &Path,
    config: &RecurrentGemmaConfig,
    device: &Device,
) -> Result<QuantRecurrentGemma> {
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(path, device)?;
    QuantRecurrentGemma::new(config, vb.pp("model"))
}

// ─── Case 3: quantized Qwen3-16B-A3B MoE (GGUF Q4_K_M, ~9.3 GB) ─────────────

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn quantized_qwen3_moe_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let model_path = qwen3_moe_gguf_path()?;
    println!("qwen3 moe gguf {model_path:?} on {}", backend_name(device));

    let mut cpu_model = load_quantized_qwen3_moe(&model_path, &cpu)?;
    let mut dev_model = load_quantized_qwen3_moe(&model_path, device)?;
    let mut violations = Vec::new();

    // Fixed prompt tokens [1, 8], single prefill forward at pos 0.
    let ids = [1u32, 8];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
    let prefill_dev = dev_model.forward(&ids_dev, 0)?;
    compare_and_record(
        &prefill_dev,
        &prefill_cpu,
        QUANT_ATOL,
        QUANT_RTOL,
        "qwen3_moe_quantized_prefill_logits",
        &mut violations,
    )?;

    // Single-token decode step at pos 2.
    let next_token = [8u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
    let decode_dev = dev_model.forward(&next_dev, ids.len())?;
    compare_and_record(
        &decode_dev,
        &decode_cpu,
        QUANT_ATOL,
        QUANT_RTOL,
        "qwen3_moe_quantized_decode_logits",
        &mut violations,
    )?;
    finish_case("quantized_qwen3_moe_case", &violations)
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn qwen3_moe_gguf_path() -> Result<PathBuf> {
    let dir = match std::env::var_os("CANDLE_QWEN3_MOE_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(r"G:\models\Qwen3-16B-A3B-GGUF"),
    };
    let path = dir.join("Qwen3-16B-A3B-Q4_K_M.gguf");
    if path.is_file() {
        Ok(path)
    } else {
        candle::bail!(
            "quantized_qwen3_moe_case: GGUF not found at {path:?} (set CANDLE_QWEN3_MOE_DIR)"
        )
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn load_quantized_qwen3_moe(path: &Path, device: &Device) -> Result<Qwen3Moe> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| err.with_path(path))?;
    // F32 everywhere (rotary/mask/expert compute) for strict CPU/GPU parity.
    Qwen3Moe::from_gguf(content, &mut file, device, DType::F32)
}
