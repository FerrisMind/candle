mod support;

use candle::{quantized::gguf_file, DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::{
    bert, convmixer, llama2_c, llama2_c_weights, quantized_qwen3, whisper,
};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{
    assert_close_tensors, deterministic_f32_data, download_model_artifact, mean_pool,
    native_required, TestBackend,
};

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_wgpu() -> Result<()> {
    native_required(
        "gpu_model_matrix_wgpu",
        TestBackend::Wgpu,
        run_public_model_matrix,
    )
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_vulkan() -> Result<()> {
    native_required(
        "gpu_model_matrix_vulkan",
        TestBackend::Vulkan,
        run_public_model_matrix,
    )
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_public_model_matrix(device: &Device) -> Result<()> {
    if cfg!(debug_assertions) {
        println!(
            "gpu model matrix is running in the debug test profile; for certification runtime use `cargo test --release`"
        );
    }

    run_case(
        "dense_causal_decoder_case",
        device,
        dense_causal_decoder_case,
    )?;
    run_case(
        "quantized_causal_gguf_case",
        device,
        quantized_causal_gguf_case,
    )?;
    run_case("encoder_only_text_case", device, encoder_only_text_case)?;
    run_case("audio_seq2seq_case", device, audio_seq2seq_case)?;
    run_case("vision_convmixer_case", device, vision_convmixer_case)?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn fallback_count(device: &Device) -> usize {
    if device.is_wgpu() {
        candle::wgpu_cpu_fallback_count()
    } else if device.is_vulkan() {
        candle::vulkan_cpu_fallback_count()
    } else {
        0
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn backend_name(device: &Device) -> &'static str {
    if device.is_wgpu() {
        "wgpu"
    } else if device.is_vulkan() {
        "vulkan"
    } else {
        "cpu"
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_case(name: &str, device: &Device, f: fn(&Device) -> Result<()>) -> Result<()> {
    println!("running {name} on {}", backend_name(device));
    let start = Instant::now();
    f(device)?;
    println!(
        "{name} finished in {:.2?}; fallback count after {name}: {}",
        start.elapsed(),
        fallback_count(device)
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn dense_causal_decoder_case(device: &Device) -> Result<()> {
    let model_path = download_model_artifact("karpathy/tinyllamas", "main", "stories15M.bin")?;
    let cpu = Device::Cpu;

    let (cpu_model, mut cpu_cache, config) = load_llama2_c_model(&model_path, &cpu)?;
    let (dev_model, mut dev_cache, _) = load_llama2_c_model(&model_path, device)?;

    let ids = [1u32, 13, 42, 7, 19, 5];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let prefill_cpu = cpu_model.forward(&ids_cpu, 0, &mut cpu_cache)?;
    let prefill_dev = dev_model.forward(&ids_dev, 0, &mut dev_cache)?;
    assert_close_tensors(
        &prefill_dev,
        &prefill_cpu,
        5e-2,
        5e-2,
        "llama2_c_prefill_logits",
    )?;

    let next_token = [11u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, ids.len(), &mut cpu_cache)?;
    let decode_dev = dev_model.forward(&next_dev, ids.len(), &mut dev_cache)?;
    assert_close_tensors(
        &decode_dev,
        &decode_cpu,
        5e-2,
        5e-2,
        "llama2_c_decode_logits",
    )?;

    // The downloaded binary defines the config contract for this representative dense decoder.
    assert!(config.seq_len >= ids.len() + 1);
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn encoder_only_text_case(device: &Device) -> Result<()> {
    let config_path = download_model_artifact(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "config.json",
    )?;
    let weights_path = download_model_artifact(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "model.safetensors",
    )?;
    let config: bert::Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
        .map_err(|err| candle::Error::msg(format!("failed to parse MiniLM config: {err}")))?;
    let cpu = Device::Cpu;

    let cpu_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], bert::DTYPE, &cpu)? };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], bert::DTYPE, device)? };
    let cpu_model = bert::BertModel::load(cpu_vb, &config)?;
    let dev_model = bert::BertModel::load(dev_vb, &config)?;

    let ids = [101u32, 2023, 2003, 1037, 3231, 102, 0, 0];
    let mask = [1u32, 1, 1, 1, 1, 1, 0, 0];
    let token_type_ids = [0u32; 8];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let mask_cpu = Tensor::from_slice(&mask, (1, mask.len()), &cpu)?;
    let mask_dev = Tensor::from_slice(&mask, (1, mask.len()), device)?;
    let tt_cpu = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), &cpu)?;
    let tt_dev = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), device)?;

    let hidden_cpu = cpu_model.forward(&ids_cpu, &tt_cpu, Some(&mask_cpu))?;
    let hidden_dev = dev_model.forward(&ids_dev, &tt_dev, Some(&mask_dev))?;
    assert_close_tensors(
        &hidden_dev,
        &hidden_cpu,
        5e-2,
        5e-2,
        "bert_final_hidden_states",
    )?;

    let pooled_cpu = mean_pool(&hidden_cpu, &mask_cpu)?;
    let pooled_dev = mean_pool(&hidden_dev, &mask_dev)?;
    assert_close_tensors(
        &pooled_dev,
        &pooled_cpu,
        5e-2,
        5e-2,
        "bert_pooled_embedding",
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn quantized_causal_gguf_case(device: &Device) -> Result<()> {
    let model_path = qwen3_gguf_path()?;
    let cpu = Device::Cpu;

    let mut cpu_model = load_quantized_qwen3_model(&model_path, &cpu)?;
    let mut dev_model = load_quantized_qwen3_model(&model_path, device)?;

    let ids = [1u32, 2, 3, 4];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
    let prefill_dev = dev_model.forward(&ids_dev, 0)?;
    assert_quantized_qwen3_close(
        &prefill_dev,
        &prefill_cpu,
        5e-2,
        "qwen3_quantized_prefill_logits",
    )?;

    let next_token = [5u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
    let decode_dev = dev_model.forward(&next_dev, ids.len())?;
    assert_quantized_qwen3_close(
        &decode_dev,
        &decode_cpu,
        5e-2,
        "qwen3_quantized_decode_logits",
    )?;

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn audio_seq2seq_case(device: &Device) -> Result<()> {
    let config_path =
        download_model_artifact("openai/whisper-tiny.en", "refs/pr/15", "config.json")?;
    let weights_path =
        download_model_artifact("openai/whisper-tiny.en", "refs/pr/15", "model.safetensors")?;
    let config: whisper::Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
        .map_err(|err| candle::Error::msg(format!("failed to parse Whisper config: {err}")))?;
    let cpu = Device::Cpu;

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], whisper::DTYPE, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], whisper::DTYPE, device)? };
    let mut cpu_model = whisper::model::Whisper::load(&cpu_vb, config.clone())?;
    let mut dev_model = whisper::model::Whisper::load(&dev_vb, config.clone())?;

    let mel = deterministic_f32_data(config.num_mel_bins * whisper::N_FRAMES, 0x5151);
    let mel_cpu = Tensor::from_vec(
        mel.clone(),
        (1, config.num_mel_bins, whisper::N_FRAMES),
        &cpu,
    )?;
    let mel_dev = Tensor::from_vec(mel, (1, config.num_mel_bins, whisper::N_FRAMES), device)?;
    let encoder_cpu = cpu_model.encoder.forward(&mel_cpu, true)?;
    let encoder_dev = dev_model.encoder.forward(&mel_dev, true)?;
    assert_close_tensors(
        &encoder_dev,
        &encoder_cpu,
        1e-3,
        1e-3,
        "whisper_encoder_output",
    )?;

    let decoder_ids = [1u32];
    let decoder_ids_cpu = Tensor::from_slice(&decoder_ids, (1, decoder_ids.len()), &cpu)?;
    let decoder_ids_dev = Tensor::from_slice(&decoder_ids, (1, decoder_ids.len()), device)?;
    let decoder_hidden_cpu = cpu_model
        .decoder
        .forward(&decoder_ids_cpu, &encoder_cpu, true)?;
    let decoder_hidden_dev = dev_model
        .decoder
        .forward(&decoder_ids_dev, &encoder_dev, true)?;
    let logits_cpu = cpu_model.decoder.final_linear(&decoder_hidden_cpu)?;
    let logits_dev = dev_model.decoder.final_linear(&decoder_hidden_dev)?;
    assert_close_tensors(
        &logits_dev,
        &logits_cpu,
        5e-2,
        5e-2,
        "whisper_decode_logits",
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn vision_convmixer_case(device: &Device) -> Result<()> {
    let weights_path = download_model_artifact(
        "lmz/candle-convmixer",
        "main",
        "convmixer_1024_20_ks9_p14.safetensors",
    )?;
    let cpu = Device::Cpu;

    let cpu_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &cpu)? };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let cpu_model = convmixer::c1024_20(1000, cpu_vb)?;
    let dev_model = convmixer::c1024_20(1000, dev_vb)?;

    let image = deterministic_f32_data(3 * 224 * 224, 0xA11CE);
    let image_cpu = Tensor::from_vec(image.clone(), (1, 3, 224, 224), &cpu)?;
    let image_dev = Tensor::from_vec(image, (1, 3, 224, 224), device)?;
    let logits_cpu = cpu_model.forward(&image_cpu)?;
    let logits_dev = dev_model.forward(&image_dev)?;
    assert_close_tensors(&logits_dev, &logits_cpu, 5e-2, 5e-2, "convmixer_logits")?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn load_llama2_c_model(
    path: &std::path::Path,
    device: &Device,
) -> Result<(llama2_c::Llama, llama2_c::Cache, llama2_c::Config)> {
    let mut file = File::open(path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
    let vb = weights.var_builder(&config, device)?;
    let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
    let model = llama2_c::Llama::load(vb, config.clone())?;
    Ok((model, cache, config))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn qwen3_gguf_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("CANDLE_QWEN3_GGUF_PATH") {
        return Ok(PathBuf::from(path));
    }
    download_model_artifact("unsloth/Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q4_K_M.gguf")
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn load_quantized_qwen3_model(
    path: &Path,
    device: &Device,
) -> Result<quantized_qwen3::ModelWeights> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| err.with_path(path))?;
    quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_quantized_qwen3_close(
    actual: &Tensor,
    expected: &Tensor,
    tol: f32,
    label: &str,
) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            actual.dims(),
            expected.dims()
        );
    }
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_idx = 0usize;
    let mut max_diff = 0f32;
    let mut max_rel = 0f32;
    let mut mse_diff = 0f64;
    let mut mse_ref = 0f64;
    let mut dot = 0f64;
    let mut actual_norm = 0f64;
    let mut expected_norm = 0f64;
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let rel = diff / expected.abs().max(1.0);
        if diff > max_diff {
            max_idx = idx;
            max_diff = diff;
            max_rel = rel;
        } else {
            max_rel = max_rel.max(rel);
        }
        let diff64 = (*actual as f64) - (*expected as f64);
        mse_diff += diff64 * diff64;
        mse_ref += (*expected as f64) * (*expected as f64);
        dot += (*actual as f64) * (*expected as f64);
        actual_norm += (*actual as f64) * (*actual as f64);
        expected_norm += (*expected as f64) * (*expected as f64);
    }
    let nmse = if mse_ref > 0.0 {
        mse_diff / mse_ref
    } else {
        0.0
    };
    let cosine = if actual_norm > 0.0 && expected_norm > 0.0 {
        dot / (actual_norm.sqrt() * expected_norm.sqrt())
    } else {
        1.0
    };
    let actual_argmax = argmax_index(&actual);
    let expected_argmax = argmax_index(&expected);
    let top5_overlap = topk_overlap(&actual, &expected, 5);
    if max_rel > tol
        && !(nmse <= 1e-2
            && cosine >= 0.995
            && actual_argmax == expected_argmax
            && top5_overlap >= 4)
    {
        candle::bail!(
            "{label}: mismatch: max_idx={max_idx} max_diff={max_diff} max_rel={max_rel} nmse={nmse} cosine={cosine} argmax_actual={actual_argmax} argmax_expected={expected_argmax} top5_overlap={top5_overlap}"
        );
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
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

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn topk_overlap(actual: &[f32], expected: &[f32], k: usize) -> usize {
    fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    let actual_topk = topk_indices(actual, k);
    let expected_topk = topk_indices(expected, k);
    actual_topk
        .iter()
        .filter(|idx| expected_topk.contains(idx))
        .count()
}
