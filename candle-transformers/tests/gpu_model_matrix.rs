mod support;

use candle::{quantized::gguf_file, DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::{
    bert, convmixer, llama2_c, llama2_c_weights, quantized_qwen3, qwen3, whisper,
};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{
    assert_close_tensors, deterministic_f32_data, download_model_artifact, mean_pool,
    native_required, TestBackend,
};

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const CASE_FILTER_ENV: &str = "CANDLE_GPU_MODEL_CASE_FILTER";

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
type ModelCaseFn = fn(&Device) -> Result<()>;

#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_cuda() -> Result<()> {
    native_required(
        "gpu_model_matrix_cuda",
        TestBackend::Cuda,
        run_public_model_matrix,
    )
}

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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn run_public_model_matrix(device: &Device) -> Result<()> {
    if cfg!(debug_assertions) {
        println!(
            "gpu model matrix is running in the debug test profile; for certification runtime use `cargo test --release`"
        );
    }

    let requested_cases = requested_case_names();
    let cases: [(&str, ModelCaseFn); 7] = [
        ("dense_causal_decoder_case", dense_causal_decoder_case),
        ("dense_qwen3_safetensors_case", dense_qwen3_safetensors_case),
        ("quantized_causal_gguf_case", quantized_causal_gguf_case),
        ("quantized_qwen3_multi_quant_case", quantized_qwen3_multi_quant_case),
        ("encoder_only_text_case", encoder_only_text_case),
        ("audio_seq2seq_case", audio_seq2seq_case),
        ("vision_convmixer_case", vision_convmixer_case),
    ];
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
        run_case(name, device, case_fn)?;
    }
    if !ran_any {
        candle::bail!(
            "{CASE_FILTER_ENV} did not match any public GPU model case: dense_causal_decoder_case, dense_qwen3_safetensors_case, quantized_causal_gguf_case, quantized_qwen3_multi_quant_case, encoder_only_text_case, audio_seq2seq_case, vision_convmixer_case"
        );
    }
    Ok(())
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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn fallback_count(device: &Device) -> usize {
    if device.is_wgpu() {
        candle::wgpu_cpu_fallback_count()
    } else if device.is_vulkan() {
        candle::vulkan_cpu_fallback_count()
    } else {
        0
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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
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
    assert!(config.seq_len > ids.len());
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
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

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), bert::DTYPE, &cpu)?
    };
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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn dense_qwen3_safetensors_case(device: &Device) -> Result<()> {
    let (config_path, weights_path) = qwen3_safetensors_paths()?;
    let cpu = Device::Cpu;
    let config: qwen3::Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .map_err(|err| candle::Error::msg(format!("failed to parse Qwen3 config: {err}")))?;
    // F32 for CPU↔GPU parity (source weights are bf16 on disk).
    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let mut cpu_model = qwen3::ModelForCausalLM::new(&config, cpu_vb)?;
    let mut dev_model = qwen3::ModelForCausalLM::new(&config, dev_vb)?;

    let ids = [1u32, 2, 3, 4];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
    let prefill_dev = dev_model.forward(&ids_dev, 0)?;
    assert_close_tensors(
        &prefill_dev,
        &prefill_cpu,
        5e-2,
        5e-2,
        "qwen3_dense_prefill_logits",
    )?;

    let next_token = [5u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
    let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
    let decode_dev = dev_model.forward(&next_dev, ids.len())?;
    assert_close_tensors(
        &decode_dev,
        &decode_cpu,
        5e-2,
        5e-2,
        "qwen3_dense_decode_logits",
    )?;
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn quantized_causal_gguf_case(device: &Device) -> Result<()> {
    let model_path = qwen3_gguf_path()?;
    let cpu = Device::Cpu;
    println!("using GGUF {model_path:?} on {}", backend_name(device));

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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn quantized_qwen3_multi_quant_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    // Exercise distinct GGUF dequant packing paths present under CANDLE_QWEN3_GGUF_DIR.
    let quant_types: &[(&str, &str)] = &[
        ("Q4_K_M", "Qwen3-0.6B-Q4_K_M.gguf"),
        ("Q4_0", "Qwen3-0.6B-Q4_0.gguf"),
        ("Q5_K_M", "Qwen3-0.6B-Q5_K_M.gguf"),
        ("Q6_K", "Qwen3-0.6B-Q6_K.gguf"),
        ("Q8_0", "Qwen3-0.6B-Q8_0.gguf"),
    ];
    let gguf_dir = qwen3_gguf_dir();
    println!("GGUF dir {gguf_dir:?} on {}", backend_name(device));
    let ids = [1u32, 2, 3, 4];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let next_token = [5u32];
    let next_cpu = Tensor::from_slice(&next_token, (1, 1), &cpu)?;
    let mut ran = 0usize;
    for &(quant_name, filename) in quant_types {
        let path = gguf_dir.join(filename);
        if !path.exists() {
            println!("skipping {quant_name}: {path:?} not found");
            continue;
        }
        ran += 1;
        println!("testing {quant_name} on {}", backend_name(device));
        let start = Instant::now();
        let mut cpu_model = load_quantized_qwen3_model(&path, &cpu)?;
        let mut dev_model = load_quantized_qwen3_model(&path, device)?;
        let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
        let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
        let prefill_dev = dev_model.forward(&ids_dev, 0)?;
        assert_quantized_qwen3_close(
            &prefill_dev,
            &prefill_cpu,
            5e-2,
            &format!("qwen3_{quant_name}_prefill"),
        )?;
        let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
        let next_dev = Tensor::from_slice(&next_token, (1, 1), device)?;
        let decode_dev = dev_model.forward(&next_dev, ids.len())?;
        assert_quantized_qwen3_close(
            &decode_dev,
            &decode_cpu,
            5e-2,
            &format!("qwen3_{quant_name}_decode"),
        )?;
        println!(
            "{quant_name} passed in {:.2?}; fallback count: {}",
            start.elapsed(),
            fallback_count(device),
        );
    }
    if ran == 0 {
        candle::bail!(
            "quantized_qwen3_multi_quant_case: no GGUF files found under {gguf_dir:?}; set CANDLE_QWEN3_GGUF_DIR"
        );
    }
    Ok(())
}



#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn audio_seq2seq_case(device: &Device) -> Result<()> {
    let config_path =
        download_model_artifact("openai/whisper-tiny.en", "refs/pr/15", "config.json")?;
    let weights_path =
        download_model_artifact("openai/whisper-tiny.en", "refs/pr/15", "model.safetensors")?;
    let config: whisper::Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
        .map_err(|err| candle::Error::msg(format!("failed to parse Whisper config: {err}")))?;
    let cpu = Device::Cpu;

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&weights_path),
            whisper::DTYPE,
            &cpu,
        )?
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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn vision_convmixer_case(device: &Device) -> Result<()> {
    let weights_path = download_model_artifact(
        "lmz/candle-convmixer",
        "main",
        "convmixer_1024_20_ks9_p14.safetensors",
    )?;
    let cpu = Device::Cpu;

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let cpu_model = convmixer::c1024_20(1000, cpu_vb)?;
    let dev_model = convmixer::c1024_20(1000, dev_vb)?;

    let image = deterministic_f32_data(3 * 224 * 224, 0xA11CE);
    let image_cpu = Tensor::from_vec(image.clone(), (1, 3, 224, 224), &cpu)?;
    let image_dev = Tensor::from_vec(image, (1, 3, 224, 224), device)?;
    let start = Instant::now();
    let logits_dev = dev_model.forward(&image_dev)?;
    let elapsed = start.elapsed();
    let logits_cpu = cpu_model.forward(&image_cpu)?;
    assert_close_tensors(&logits_dev, &logits_cpu, 5e-2, 5e-2, "convmixer_logits")?;
    // ponytail: CUDA baseline ~10s on RTX 3060; 60s is a generous bound for all backends.
    let max_secs = if device.is_cuda() { 30 } else { 60 };
    assert!(
        elapsed.as_secs() < max_secs,
        "ConMixer too slow on {}: {elapsed:.2?} (max {max_secs}s)",
        backend_name(device)
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
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

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn qwen3_gguf_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("CANDLE_QWEN3_GGUF_DIR") {
        return PathBuf::from(path);
    }
    let local = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-0.6B-GGUF");
    if local.is_dir() {
        return local;
    }
    PathBuf::from("/home/mod479711/Downloads/models/Qwen3-0.6B-GGUF")
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn qwen3_gguf_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("CANDLE_QWEN3_GGUF_PATH") {
        return Ok(PathBuf::from(path));
    }
    let local = qwen3_gguf_dir().join("Qwen3-0.6B-Q4_K_M.gguf");
    if local.is_file() {
        return Ok(local);
    }
    download_model_artifact("unsloth/Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q4_K_M.gguf")
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn qwen3_safetensors_paths() -> Result<(PathBuf, PathBuf)> {
    if let (Some(cfg), Some(wts)) = (
        std::env::var_os("CANDLE_QWEN3_CONFIG_PATH"),
        std::env::var_os("CANDLE_QWEN3_SAFETENSORS_PATH"),
    ) {
        return Ok((PathBuf::from(cfg), PathBuf::from(wts)));
    }
    if let Some(dir) = std::env::var_os("CANDLE_QWEN3_DENSE_DIR") {
        let dir = PathBuf::from(dir);
        return Ok((dir.join("config.json"), dir.join("model.safetensors")));
    }
    let local = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-0.6B");
    let config = local.join("config.json");
    let weights = local.join("model.safetensors");
    if config.is_file() && weights.is_file() {
        return Ok((config, weights));
    }
    let config = download_model_artifact("unsloth/Qwen3-0.6B", "main", "config.json")?;
    let weights = download_model_artifact("unsloth/Qwen3-0.6B", "main", "model.safetensors")?;
    Ok((config, weights))
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn load_quantized_qwen3_model(
    path: &Path,
    device: &Device,
) -> Result<quantized_qwen3::ModelWeights> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| err.with_path(path))?;
    quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
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
