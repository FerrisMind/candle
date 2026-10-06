//! GPU certification matrix extension (worker W4): generation + encoder models.
//!
//! Cases (CPU-vs-GPU parity, same local weights on both sides, deterministic
//! inputs, single forward passes -- full multi-step sampling is out of scope):
//!
//! 1. `stable_diffusion_unet_case` -- Stable Diffusion v1.5 UNet noise prediction
//!    for a deterministic latent [1, 4, 64, 64], fixed timestep, and the CLIP
//!    text-encoder output for a fixed token sequence.
//! 2. `stable_diffusion_vae_case` -- Stable Diffusion v1.5 VAE decoder of a
//!    deterministic latent [1, 4, 64, 64].
//! 3. `bge_small_case` -- bge-small-en-v1.5 (BERT encoder) last hidden state and
//!    mean-pooled embedding for fixed token ids [1, 8].
//!
//! Note on SD config files: the local mirror `G:/models/stable-diffusion-v1-5`
//! carries weights + tokenizer only (the `unet/`, `vae/` and `text_encoder/`
//! sub-directories have no `config.json` and the HF cache snapshot for this
//! model contains no snapshots). This is NOT a blocker: candle's
//! `stable_diffusion::StableDiffusionConfig::v1_5` hardcodes the diffusers
//! architecture hyper-parameters (matching the upstream runwayml/stable-diffusion-v1-5
//! configs), so only the safetensors weight files are required.
//!
//! Weights are f16 on disk; they are loaded as F32 on both sides, so the strict
//! tolerance (atol=1e-3, rtol=1e-3) applies per the certification policy.
//! `CANDLE_GEN_EXT_TOL=<f32>` overrides both for supplementary survey runs.
//! `CANDLE_GEN_EXT_OPS_DIAG=1` appends an opt-in `op_diagnosis_case` (op-level
//! probes + UNet latent-size/timestep bisect) after the official cases.

mod support;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{bert, stable_diffusion};
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{
    assert_close_tensors, deterministic_f32_data, mean_pool, native_required, TestBackend,
};

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const CASE_FILTER_ENV: &str = "CANDLE_GPU_GEN_EXT_CASE_FILTER";

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const SD_DIR_ENV: &str = "CANDLE_SD_DIR";

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const BGE_DIR_ENV: &str = "CANDLE_BGE_DIR";

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const SD_DIR_DEFAULT: &str = r"G:\models\stable-diffusion-v1-5";

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const BGE_DIR_DEFAULT: &str = r"G:\models\bge-small-en-v1.5";

// F32 weights on both sides (f16 on disk is converted by VarBuilder on load).
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const ATOL: f32 = 1e-3;

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
const RTOL: f32 = 1e-3;

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
type ModelCaseFn = fn(&Device) -> Result<()>;

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_gen_ext_wgpu() -> Result<()> {
    native_required(
        "gpu_model_matrix_gen_ext_wgpu",
        TestBackend::Wgpu,
        run_gen_ext_matrix,
    )
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual GPU certification matrix"]
fn gpu_model_matrix_gen_ext_vulkan() -> Result<()> {
    native_required(
        "gpu_model_matrix_gen_ext_vulkan",
        TestBackend::Vulkan,
        run_gen_ext_matrix,
    )
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn tolerance() -> (f32, f32) {
    match std::env::var("CANDLE_GEN_EXT_TOL") {
        Ok(value) => {
            let parsed: f32 = value.parse().unwrap_or(ATOL);
            println!(
                "using CANDLE_GEN_EXT_TOL={parsed} (atol=rtol={parsed}) instead of the {ATOL}/{RTOL} policy"
            );
            (parsed, parsed)
        }
        Err(_) => (ATOL, RTOL),
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn run_gen_ext_matrix(device: &Device) -> Result<()> {
    if cfg!(debug_assertions) {
        println!(
            "gen ext matrix is running in the debug test profile; for certification runtime use `cargo test --release`"
        );
    }

    let requested_cases = requested_case_names();
    let cases: [(&str, ModelCaseFn); 3] = [
        ("stable_diffusion_unet_case", stable_diffusion_unet_case),
        ("stable_diffusion_vae_case", stable_diffusion_vae_case),
        ("bge_small_case", bge_small_case),
    ];
    let mut ran_any = false;
    let mut failures: Vec<String> = Vec::new();
    for (name, case_fn) in cases {
        if !case_is_requested(name, requested_cases.as_deref()) {
            println!(
                "skipping {name} on {} due to {CASE_FILTER_ENV}",
                device_name(device)
            );
            continue;
        }
        ran_any = true;
        if let Err(err) = run_case(name, device, case_fn) {
            failures.push(format!("{name}: {err}"));
        }
    }
    if std::env::var("CANDLE_GEN_EXT_OPS_DIAG").as_deref() == Ok("1") {
        // Diagnostics-only case: never gates the official matrix verdict.
        let _ = run_case("op_diagnosis_case", device, op_diagnosis_case);
    }
    if !ran_any {
        candle::bail!(
            "{CASE_FILTER_ENV} did not match any gen ext GPU case: stable_diffusion_unet_case, stable_diffusion_vae_case, bge_small_case"
        );
    }
    if !failures.is_empty() {
        candle::bail!(
            "gen ext matrix failures on {}: {}",
            device_name(device),
            failures.join(" | ")
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
fn device_name(device: &Device) -> &'static str {
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

/// Runs one matrix case; prints timing and the per-case fallback delta. The
/// caller records failures so the matrix keeps running past a failing case.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn run_case(name: &str, device: &Device, f: fn(&Device) -> Result<()>) -> Result<()> {
    println!("running {name} on {}", device_name(device));
    let before = fallback_count(device);
    let start = Instant::now();
    let outcome = f(device);
    let elapsed = start.elapsed();
    let after = fallback_count(device);
    match &outcome {
        Ok(()) => println!(
            "{name}: PASS on {} in {elapsed:.2?}; fallback count after {name}: {after} (delta {})",
            device_name(device),
            after - before
        ),
        Err(err) => println!(
            "{name}: FAIL on {} in {elapsed:.2?}; fallback count after {name}: {after} (delta {}); error: {err}",
            device_name(device),
            after - before
        ),
    }
    outcome
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn required_file(dir: &Path, relative: &str) -> Result<PathBuf> {
    let path = dir.join(relative);
    if !path.is_file() {
        candle::bail!("required file missing: {}", path.display());
    }
    Ok(path)
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn sd_dir() -> PathBuf {
    match std::env::var_os(SD_DIR_ENV) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(SD_DIR_DEFAULT),
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn bge_dir() -> PathBuf {
    match std::env::var_os(BGE_DIR_ENV) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(BGE_DIR_DEFAULT),
    }
}

/// Prints (and returns) the observed max absolute elementwise diff between the
/// two tensors after flattening, so every case records its observed max diff.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn report_max_diff(actual: &Tensor, expected: &Tensor, label: &str) -> Result<f32> {
    let actual = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if actual.len() != expected.len() {
        candle::bail!(
            "{label}: shape mismatch, got {} elems, expected {} elems",
            actual.len(),
            expected.len()
        );
    }
    let mut max_diff = 0f32;
    let mut max_idx = 0usize;
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
        }
    }
    println!(
        "{label}: observed max abs diff {max_diff:.3e} at idx {max_idx} over {} elements",
        actual.len()
    );
    Ok(max_diff)
}

/// Case 1: single SD 1.5 UNet forward for a deterministic latent, fixed
/// timestep and a deterministic text-embedding tensor (the CLIP text encoder
/// output for a fixed token sequence, computed once on CPU and shared with the
/// GPU device so the case isolates the UNet itself).
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn stable_diffusion_unet_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let dir = sd_dir();
    let unet_weights = required_file(&dir, r"unet\diffusion_pytorch_model.fp16.safetensors")?;
    let clip_weights = required_file(&dir, r"text_encoder\model.fp16.safetensors")?;
    println!("unet weights: {}", unet_weights.display());
    println!("clip weights: {}", clip_weights.display());

    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, None, None);
    let seq_len = sd_config.clip.max_position_embeddings;

    // CLIP text encoder on CPU only; output shared across devices.
    let clip_cpu =
        stable_diffusion::build_clip_transformer(&sd_config.clip, &clip_weights, &cpu, DType::F32)?;
    // Fixed token sequence: "<|startoftext|>", "a", "photo", "<|endoftext|>" + padding.
    let mut ids = vec![49406u32, 320, 4558, 49407];
    ids.resize(seq_len, 49407);
    let tokens_cpu = Tensor::from_slice(&ids, (1, seq_len), &cpu)?;
    let text_emb_cpu = clip_cpu.forward(&tokens_cpu)?;
    println!("text embeddings: {:?}", text_emb_cpu.dims());

    let latent_data = deterministic_f32_data(4 * 64 * 64, 0x5D1_5EED);
    let latent_cpu = Tensor::from_vec(latent_data.clone(), (1, 4, 64, 64), &cpu)?;
    let latent_dev = Tensor::from_vec(latent_data, (1, 4, 64, 64), device)?;
    let text_emb_dev = text_emb_cpu.to_device(device)?;

    let unet_cpu = sd_config.build_unet(&unet_weights, &cpu, 4, false, DType::F32)?;
    let unet_dev = sd_config.build_unet(&unet_weights, device, 4, false, DType::F32)?;

    let timestep = 981f64;
    let start = Instant::now();
    let noise_pred_dev = unet_dev.forward(&latent_dev, timestep, &text_emb_dev)?;
    device.synchronize()?;
    println!(
        "unet forward on {} took {:.2?}",
        device_name(device),
        start.elapsed()
    );
    let noise_pred_cpu = unet_cpu.forward(&latent_cpu, timestep, &text_emb_cpu)?;
    println!("unet noise pred: {:?}", noise_pred_dev.dims());

    let (atol, rtol) = tolerance();
    report_max_diff(
        &noise_pred_dev,
        &noise_pred_cpu,
        "stable_diffusion_unet_noise_pred",
    )?;
    assert_close_tensors(
        &noise_pred_dev,
        &noise_pred_cpu,
        atol,
        rtol,
        "stable_diffusion_unet_noise_pred",
    )?;
    Ok(())
}

/// Case 2: single SD 1.5 VAE decode of a deterministic latent [1, 4, 64, 64]
/// (scaled by 1/vae_scale as the sampling loop does before decode).
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn stable_diffusion_vae_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let dir = sd_dir();
    let vae_weights = required_file(&dir, r"vae\diffusion_pytorch_model.fp16.safetensors")?;
    println!("vae weights: {}", vae_weights.display());

    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, None, None);
    let vae_scale = 0.18215f64;

    let latent_data = deterministic_f32_data(4 * 64 * 64, 0xAE1_5EED);
    let latent_cpu = Tensor::from_vec(latent_data.clone(), (1, 4, 64, 64), &cpu)?;
    let latent_dev = Tensor::from_vec(latent_data, (1, 4, 64, 64), device)?;

    let vae_cpu = sd_config.build_vae(&vae_weights, &cpu, DType::F32)?;
    let vae_dev = sd_config.build_vae(&vae_weights, device, DType::F32)?;

    let start = Instant::now();
    let image_dev = vae_dev.decode(&(&latent_dev / vae_scale)?)?;
    device.synchronize()?;
    println!(
        "vae decode on {} took {:.2?}",
        device_name(device),
        start.elapsed()
    );
    let image_cpu = vae_cpu.decode(&(&latent_cpu / vae_scale)?)?;
    println!("decoded image: {:?}", image_dev.dims());

    let (atol, rtol) = tolerance();
    report_max_diff(&image_dev, &image_cpu, "stable_diffusion_vae_decoded_image")?;
    assert_close_tensors(
        &image_dev,
        &image_cpu,
        atol,
        rtol,
        "stable_diffusion_vae_decoded_image",
    )?;
    Ok(())
}

/// Case 3: bge-small-en-v1.5 (BERT encoder) last hidden state and mean-pooled
/// embedding for fixed token ids [1, 8].
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn bge_small_case(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let dir = bge_dir();
    let config_path = required_file(&dir, "config.json")?;
    let weights_path = required_file(&dir, "model.safetensors")?;
    println!("bge config: {}", config_path.display());
    println!("bge weights: {}", weights_path.display());
    let config: bert::Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
        .map_err(|err| candle::Error::msg(format!("failed to parse bge config: {err}")))?;

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, device)?
    };
    let cpu_model = bert::BertModel::load(cpu_vb, &config)?;
    let dev_model = bert::BertModel::load(dev_vb, &config)?;

    let ids = [1u32, 8];
    let mask = [1u32, 1];
    let token_type_ids = [0u32, 0];
    let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
    let mask_cpu = Tensor::from_slice(&mask, (1, mask.len()), &cpu)?;
    let mask_dev = Tensor::from_slice(&mask, (1, mask.len()), device)?;
    let tt_cpu = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), &cpu)?;
    let tt_dev = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), device)?;

    let start = Instant::now();
    let hidden_dev = dev_model.forward(&ids_dev, &tt_dev, Some(&mask_dev))?;
    device.synchronize()?;
    println!(
        "bge forward on {} took {:.2?}",
        device_name(device),
        start.elapsed()
    );
    let hidden_cpu = cpu_model.forward(&ids_cpu, &tt_cpu, Some(&mask_cpu))?;
    println!("bge last hidden state: {:?}", hidden_dev.dims());

    let (atol, rtol) = tolerance();
    report_max_diff(&hidden_dev, &hidden_cpu, "bge_last_hidden_state")?;
    assert_close_tensors(&hidden_dev, &hidden_cpu, atol, rtol, "bge_last_hidden_state")?;

    let pooled_cpu = mean_pool(&hidden_cpu, &mask_cpu)?;
    let pooled_dev = mean_pool(&hidden_dev, &mask_dev)?;
    report_max_diff(&pooled_dev, &pooled_cpu, "bge_pooled_embedding")?;
    assert_close_tensors(&pooled_dev, &pooled_cpu, atol, rtol, "bge_pooled_embedding")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostics (opt-in via CANDLE_GEN_EXT_OPS_DIAG=1): op-level probes at
// UNet-realistic shapes plus a UNet latent-size x timestep bisect.
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn run_probe(name: &str, device: &Device, f: impl Fn(&Device) -> Result<Tensor>) {
    match (|| -> Result<f32> {
        let dev_out = f(device)?;
        let cpu_out = f(&Device::Cpu)?;
        report_max_diff(&dev_out, &cpu_out, name)
    })() {
        Ok(diff) => println!("[probe] {name}: max_abs_diff={diff:.3e}"),
        Err(err) => println!("[probe] {name}: ERROR {err}"),
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "vulkan"))]
fn op_diagnosis_case(device: &Device) -> Result<()> {
    println!("=== gen-ext op diagnosis on {} ===", device_name(device));

    // Replicates unet_2d::Timesteps arguments for timestep 981 (sin/cos of
    // values up to ~981 rad; WGSL sin/cos precision on large args is suspect).
    let timestep_args = |device: &Device| -> Result<Tensor> {
        let timestep = 981f32;
        let half = 160usize;
        let exponent: Vec<f32> = (0..half)
            .map(|idx| -f32::ln(10000.0) * idx as f32 / (half - 1) as f32)
            .collect();
        let exponent = Tensor::from_vec(exponent, half, device)?;
        let emb = exponent.exp()?.affine(timestep as f64, 0.0)?;
        Tensor::cat(&[&emb.sin()?, &emb.cos()?], 0)
    };
    run_probe("timestep_sin_cos_981", device, timestep_args);

    // GroupNorm exactly as used by SD resnets: [1,320,64,64], 32 groups, eps 1e-5.
    let group_norm = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(320 * 64 * 64, 0x9101),
            (1, 320, 64, 64),
            device,
        )?;
        let w = Tensor::from_vec(deterministic_f32_data(320, 0x9102), 320, device)?;
        let b = Tensor::from_vec(deterministic_f32_data(320, 0x9103), 320, device)?;
        candle_nn::GroupNorm::new(w, b, 320, 32, 1e-5)?.forward(&x)
    };
    run_probe("group_norm_320_64x64_g32", device, group_norm);

    // SiLU and erf-GELU (GeGlu) at unet-realistic sizes.
    let silu = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(320 * 64 * 64, 0x9104),
            (1, 320, 64, 64),
            device,
        )?;
        x.silu()
    };
    run_probe("silu_320_64x64", device, silu);

    let gelu = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(4096 * 1280, 0x9105),
            (1, 4096, 1280),
            device,
        )?;
        x.gelu()
    };
    run_probe("gelu_4096x1280", device, gelu);

    // Conv2d 3x3 on 320 channels at 64x64 (SD conv_in / resnet conv).
    let conv2d_320 = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(320 * 64 * 64, 0x9106),
            (1, 320, 64, 64),
            device,
        )?;
        let weight = Tensor::from_vec(
            deterministic_f32_data(320 * 320 * 3 * 3, 0x9107),
            (320, 320, 3, 3),
            device,
        )?;
        let bias = Tensor::from_vec(deterministic_f32_data(320, 0x9108), 320, device)?;
        let config = candle_nn::conv::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        candle_nn::Conv2d::new(weight, Some(bias), config).forward(&x)
    };
    run_probe("conv2d_320x320_3x3_64x64", device, conv2d_320);

    // Attention path (CrossAttention::attention, no flash-attn): rank-3
    // batched matmuls at batch*heads=40, self-attn seq 4096, dim_head 8, then
    // softmax_lastdim, then matmul by v; plus the cross-attn seq-77 shapes.
    let attn_scores_self = |device: &Device| -> Result<Tensor> {
        let q = Tensor::from_vec(
            deterministic_f32_data(40 * 4096 * 8, 0x9109),
            (40, 4096, 8),
            device,
        )?;
        let k = Tensor::from_vec(
            deterministic_f32_data(40 * 4096 * 8, 0x910A),
            (40, 4096, 8),
            device,
        )?;
        q.matmul(&k.t()?)
    };
    run_probe("attn_scores_r3mm_b40_s4096_d8", device, attn_scores_self);

    let attn_scores_cross = |device: &Device| -> Result<Tensor> {
        let q = Tensor::from_vec(
            deterministic_f32_data(40 * 4096 * 8, 0x910B),
            (40, 4096, 8),
            device,
        )?;
        let k = Tensor::from_vec(
            deterministic_f32_data(40 * 77 * 8, 0x910C),
            (40, 77, 8),
            device,
        )?;
        q.matmul(&k.t()?)
    };
    run_probe("attn_scores_r3mm_b40_s4096_k77", device, attn_scores_cross);

    let attn_stack_self_small = |device: &Device| -> Result<Tensor> {
        // Chained scores -> softmax -> v matmul at seq 512 (bounds memory) with
        // the real head count and dim_head.
        let q = Tensor::from_vec(
            deterministic_f32_data(40 * 512 * 8, 0x910D),
            (40, 512, 8),
            device,
        )?;
        let k = Tensor::from_vec(
            deterministic_f32_data(40 * 512 * 8, 0x910E),
            (40, 512, 8),
            device,
        )?;
        let v = Tensor::from_vec(
            deterministic_f32_data(40 * 512 * 8, 0x910F),
            (40, 512, 8),
            device,
        )?;
        let scale = 1.0f32 / (8f32).sqrt();
        let scores = q.matmul(&(k.t()? * scale as f64)?)?;
        candle_nn::ops::softmax(&scores, candle::D::Minus1)?.matmul(&v)
    };
    run_probe("attn_stack_b40_s512_d8", device, attn_stack_self_small);

    let softmax_big_rows = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(4 * 4096 * 4096, 0x9110),
            (4, 4096, 4096),
            device,
        )?;
        candle_nn::ops::softmax(&x, candle::D::Minus1)
    };
    run_probe("softmax_lastdim_4x4096x4096", device, softmax_big_rows);

    // Rank-3 Linear (resnet/geglu FF path on [batch, seq, ch]).
    let linear_rank3 = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(1 * 4096 * 320, 0x9111),
            (1, 4096, 320),
            device,
        )?;
        let w = Tensor::from_vec(deterministic_f32_data(320 * 320, 0x9112), (320, 320), device)?;
        let b = Tensor::from_vec(deterministic_f32_data(320, 0x9113), 320, device)?;
        candle_nn::Linear::new(w, Some(b)).forward(&x)
    };
    run_probe("linear_rank3_1x4096x320", device, linear_rank3);

    // Upsample nearest used by UNet up blocks.
    let upsample = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(320 * 64 * 64, 0x9114),
            (1, 320, 64, 64),
            device,
        )?;
        x.upsample_nearest2d(128, 128)
    };
    run_probe("upsample_nearest2d_320_64_to_128", device, upsample);

    // Timestep-embedding chain: Timesteps module (replicated exactly), the
    // real-weights TimestepEmbedding MLP (m=1 gemv matmuls), and the
    // per-resnet time_emb_proj linear. A temb error is added to EVERY resnet
    // block, so it would produce a near-uniform deviation for every latent
    // size and timestep -- matching the bisect pattern above.
    for t in [1f64, 981.0] {
        let timesteps_module = move |device: &Device| -> Result<Tensor> {
            let timesteps = stable_diffusion::embeddings::Timesteps::new(320, true, 0.0);
            let ts = Tensor::new(&[t], device)?;
            timesteps.forward(&ts)
        };
        let label = format!("timesteps_module_t{t}");
        run_probe(&label, device, timesteps_module);
    }
    let dir = sd_dir();
    let unet_weights_t = required_file(&dir, r"unet\diffusion_pytorch_model.fp16.safetensors")?;
    for t in [1f64, 981.0] {
        let unet_weights_t = unet_weights_t.clone();
        let time_embed_mlp = move |device: &Device| -> Result<Tensor> {
            let vs = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&unet_weights_t),
                    DType::F32,
                    device,
                )?
            };
            let timesteps = stable_diffusion::embeddings::Timesteps::new(320, true, 0.0);
            // Matches unet forward: `Tensor::ones(bsize, xs.dtype()) * timestep` (f32).
            let emb_in = (Tensor::ones(1, DType::F32, device)? * t)?;
            let t_out = timesteps.forward(&emb_in)?;
            let mlp = stable_diffusion::embeddings::TimestepEmbedding::new(
                vs.pp("time_embedding"),
                320,
                1280,
            )?;
            mlp.forward(&t_out)
        };
        let label = format!("time_embed_mlp_real_t{t}");
        run_probe(&label, device, time_embed_mlp);
    }
    let gemv_m1 = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(deterministic_f32_data(320, 0x9120), (1, 320), device)?;
        let w = Tensor::from_vec(deterministic_f32_data(320 * 1280, 0x9121), (1280, 320), device)?;
        let b = Tensor::from_vec(deterministic_f32_data(1280, 0x9122), 1280, device)?;
        candle_nn::Linear::new(w, Some(b)).forward(&x)
    };
    run_probe("gemv_m1_linear_320_to_1280", device, gemv_m1);
    let resnet_temb_proj = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(deterministic_f32_data(1280, 0x9123), (1, 1280), device)?;
        let w = Tensor::from_vec(deterministic_f32_data(320 * 1280, 0x9124), (320, 1280), device)?;
        let b = Tensor::from_vec(deterministic_f32_data(320, 0x9125), 320, device)?;
        candle_nn::Linear::new(w, Some(b)).forward(&x)
    };
    run_probe("resnet_temb_proj_m1_1280_to_320", device, resnet_temb_proj);

    // ---- Stage-wise bisect through the real UNet graph (8x8 latent) ----
    // Rebuilds the UNet pipeline stage by stage with the real weights so the
    // first divergent stage can be identified exactly. temb and text
    // embeddings are computed per device exactly as unet forward does.
    let dir = sd_dir();
    let unet_weights = required_file(&dir, r"unet\diffusion_pytorch_model.fp16.safetensors")?;
    let clip_weights = required_file(&dir, r"text_encoder\model.fp16.safetensors")?;
    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, None, None);
    let seq_len = sd_config.clip.max_position_embeddings;
    let clip_cpu =
        stable_diffusion::build_clip_transformer(&sd_config.clip, &clip_weights, &Device::Cpu, DType::F32)?;
    let mut ids = vec![49406u32, 320, 4558, 49407];
    ids.resize(seq_len, 49407);
    let tokens_cpu = Tensor::from_slice(&ids, (1, seq_len), &Device::Cpu)?;
    let text_emb_cpu = clip_cpu.forward(&tokens_cpu)?;
    let text_emb_dev = text_emb_cpu.to_device(device)?;

    let vb_cpu = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&unet_weights), DType::F32, &Device::Cpu)?
    };
    let vb_dev = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&unet_weights), DType::F32, device)?
    };

    let latent8 = deterministic_f32_data(4 * 8 * 8, 0x5D1_5EED);
    let x0_cpu = Tensor::from_vec(latent8.clone(), (1, 4, 8, 8), &Device::Cpu)?;
    let x0_dev = Tensor::from_vec(latent8, (1, 4, 8, 8), device)?;

    let resnet_cfg = stable_diffusion::resnet::ResnetBlock2DConfig {
        out_channels: Some(320),
        temb_channels: Some(1280),
        eps: 1e-5,
        output_scale_factor: 1.,
        ..Default::default()
    };
    let st_cfg = stable_diffusion::attention::SpatialTransformerConfig {
        depth: 1,
        context_dim: Some(768),
        num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: false,
    };
    let conv_cfg = candle_nn::conv::Conv2dConfig {
        padding: 1,
        ..Default::default()
    };

    let conv_in_cpu = candle_nn::conv2d(4, 320, 3, conv_cfg, vb_cpu.clone().pp("conv_in"))?;
    let conv_in_dev = candle_nn::conv2d(4, 320, 3, conv_cfg, vb_dev.clone().pp("conv_in"))?;
    let res0_cpu = stable_diffusion::resnet::ResnetBlock2D::new(
        vb_cpu.clone().pp("down_blocks.0.resnets.0"),
        320,
        resnet_cfg,
    )?;
    let res0_dev = stable_diffusion::resnet::ResnetBlock2D::new(
        vb_dev.clone().pp("down_blocks.0.resnets.0"),
        320,
        resnet_cfg,
    )?;
    let st0_cpu = stable_diffusion::attention::SpatialTransformer::new(
        vb_cpu.clone().pp("down_blocks.0.attentions.0"),
        320,
        8,
        40,
        false,
        st_cfg,
    )?;
    let st0_dev = stable_diffusion::attention::SpatialTransformer::new(
        vb_dev.clone().pp("down_blocks.0.attentions.0"),
        320,
        8,
        40,
        false,
        st_cfg,
    )?;
    let res1_cpu = stable_diffusion::resnet::ResnetBlock2D::new(
        vb_cpu.clone().pp("down_blocks.0.resnets.1"),
        320,
        resnet_cfg,
    )?;
    let res1_dev = stable_diffusion::resnet::ResnetBlock2D::new(
        vb_dev.clone().pp("down_blocks.0.resnets.1"),
        320,
        resnet_cfg,
    )?;
    let down_cpu = candle_nn::conv2d(
        320,
        320,
        3,
        candle_nn::conv::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        },
        vb_cpu.clone().pp("down_blocks.0.downsamplers.0.conv"),
    )?;
    let down_dev = candle_nn::conv2d(
        320,
        320,
        3,
        candle_nn::conv::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        },
        vb_dev.clone().pp("down_blocks.0.downsamplers.0.conv"),
    )?;
    let mid_cfg = stable_diffusion::unet_2d_blocks::UNetMidBlock2DCrossAttnConfig {
        resnet_eps: 1e-5,
        output_scale_factor: 1.,
        cross_attn_dim: 768,
        attn_num_head_channels: 8,
        resnet_groups: Some(32),
        use_linear_projection: false,
        transformer_layers_per_block: 1,
        ..Default::default()
    };
    let mid_cpu = stable_diffusion::unet_2d_blocks::UNetMidBlock2DCrossAttn::new(
        vb_cpu.clone().pp("mid_block"),
        1280,
        Some(1280),
        false,
        mid_cfg,
    )?;
    let mid_dev = stable_diffusion::unet_2d_blocks::UNetMidBlock2DCrossAttn::new(
        vb_dev.clone().pp("mid_block"),
        1280,
        Some(1280),
        false,
        mid_cfg,
    )?;

    // SpatialTransformer sub-components with the real weights (for the
    // transformer sub-bisect inside the stage loop below).
    let st_prefix = "down_blocks.0.attentions.0";
    let tb_prefix = format!("{st_prefix}.transformer_blocks.0");
    let st_norm_cpu =
        candle_nn::group_norm(32, 320, 1e-6, vb_cpu.clone().pp(format!("{st_prefix}.norm")))?;
    let st_norm_dev =
        candle_nn::group_norm(32, 320, 1e-6, vb_dev.clone().pp(format!("{st_prefix}.norm")))?;
    let st_proj_in_cpu = candle_nn::conv2d(
        320,
        320,
        1,
        Default::default(),
        vb_cpu.clone().pp(format!("{st_prefix}.proj_in")),
    )?;
    let st_proj_in_dev = candle_nn::conv2d(
        320,
        320,
        1,
        Default::default(),
        vb_dev.clone().pp(format!("{st_prefix}.proj_in")),
    )?;
    let st_norm1_cpu = candle_nn::layer_norm(320, 1e-5, vb_cpu.clone().pp(format!("{tb_prefix}.norm1")))?;
    let st_norm1_dev = candle_nn::layer_norm(320, 1e-5, vb_dev.clone().pp(format!("{tb_prefix}.norm1")))?;
    let st_norm2_cpu = candle_nn::layer_norm(320, 1e-5, vb_cpu.clone().pp(format!("{tb_prefix}.norm2")))?;
    let st_norm2_dev = candle_nn::layer_norm(320, 1e-5, vb_dev.clone().pp(format!("{tb_prefix}.norm2")))?;
    let st_norm3_cpu = candle_nn::layer_norm(320, 1e-5, vb_cpu.clone().pp(format!("{tb_prefix}.norm3")))?;
    let st_norm3_dev = candle_nn::layer_norm(320, 1e-5, vb_dev.clone().pp(format!("{tb_prefix}.norm3")))?;
    let st_attn1_cpu = stable_diffusion::attention::CrossAttention::new(
        vb_cpu.clone().pp(format!("{tb_prefix}.attn1")),
        320,
        None,
        8,
        40,
        None,
        false,
    )?;
    let st_attn1_dev = stable_diffusion::attention::CrossAttention::new(
        vb_dev.clone().pp(format!("{tb_prefix}.attn1")),
        320,
        None,
        8,
        40,
        None,
        false,
    )?;
    let st_attn2_cpu = stable_diffusion::attention::CrossAttention::new(
        vb_cpu.clone().pp(format!("{tb_prefix}.attn2")),
        320,
        Some(768),
        8,
        40,
        None,
        false,
    )?;
    let st_attn2_dev = stable_diffusion::attention::CrossAttention::new(
        vb_dev.clone().pp(format!("{tb_prefix}.attn2")),
        320,
        Some(768),
        8,
        40,
        None,
        false,
    )?;
    let st_geglu_cpu = candle_nn::linear(
        320,
        2560,
        vb_cpu.clone().pp(format!("{tb_prefix}.ff.net.0.proj")),
    )?;
    let st_geglu_dev = candle_nn::linear(
        320,
        2560,
        vb_dev.clone().pp(format!("{tb_prefix}.ff.net.0.proj")),
    )?;
    let st_ff_lin_cpu = candle_nn::linear(
        1280,
        320,
        vb_cpu.clone().pp(format!("{tb_prefix}.ff.net.2")),
    )?;
    let st_ff_lin_dev = candle_nn::linear(
        1280,
        320,
        vb_dev.clone().pp(format!("{tb_prefix}.ff.net.2")),
    )?;
    let st_proj_out_cpu = candle_nn::conv2d(
        320,
        320,
        1,
        Default::default(),
        vb_cpu.clone().pp(format!("{st_prefix}.proj_out")),
    )?;
    let st_proj_out_dev = candle_nn::conv2d(
        320,
        320,
        1,
        Default::default(),
        vb_dev.clone().pp(format!("{st_prefix}.proj_out")),
    )?;

    // FF block linears with the real weights (the vulkan diagnosis showed the
    // divergence entering at the FF: geglu proj linear is no-bias, the ff
    // output linear has bias and goes through the fused mul_mat_add path on
    // vulkan when m > 8).
    let ff_geglu = |device: &Device| -> Result<Tensor> {
        let vs = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&unet_weights), DType::F32, device)?
        };
        let lin = candle_nn::linear_no_bias(
            320,
            2560,
            vs.pp(format!("{tb_prefix}.ff.net.0.proj")),
        )?;
        let x = Tensor::from_vec(
            deterministic_f32_data(1 * 64 * 320, 0x9133),
            (1, 64, 320),
            device,
        )?;
        lin.forward(&x)
    };
    run_probe("ff_geglu_lin_r3_m64_k320", device, ff_geglu);

    let ff_out_lin = |device: &Device| -> Result<Tensor> {
        let vs = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&unet_weights), DType::F32, device)?
        };
        let lin = candle_nn::linear(1280, 320, vs.pp(format!("{tb_prefix}.ff.net.2")))?;
        let x = Tensor::from_vec(
            deterministic_f32_data(1 * 64 * 1280, 0x9134),
            (1, 64, 1280),
            device,
        )?;
        lin.forward(&x)
    };
    run_probe("ff_out_lin_r3_m64_k1280_bias", device, ff_out_lin);

    // attn2 (cross-attention) internals with the real weights.
    let at2_to_q_cpu =
        candle_nn::linear_no_bias(320, 320, vb_cpu.clone().pp(format!("{tb_prefix}.attn2.to_q")))?;
    let at2_to_q_dev =
        candle_nn::linear_no_bias(320, 320, vb_dev.clone().pp(format!("{tb_prefix}.attn2.to_q")))?;
    let at2_to_k_cpu =
        candle_nn::linear_no_bias(768, 320, vb_cpu.clone().pp(format!("{tb_prefix}.attn2.to_k")))?;
    let at2_to_k_dev =
        candle_nn::linear_no_bias(768, 320, vb_dev.clone().pp(format!("{tb_prefix}.attn2.to_k")))?;
    let at2_to_v_cpu =
        candle_nn::linear_no_bias(768, 320, vb_cpu.clone().pp(format!("{tb_prefix}.attn2.to_v")))?;
    let at2_to_v_dev =
        candle_nn::linear_no_bias(768, 320, vb_dev.clone().pp(format!("{tb_prefix}.attn2.to_v")))?;
    let at2_to_out_cpu =
        candle_nn::linear(320, 320, vb_cpu.clone().pp(format!("{tb_prefix}.attn2.to_out.0")))?;
    let at2_to_out_dev =
        candle_nn::linear(320, 320, vb_dev.clone().pp(format!("{tb_prefix}.attn2.to_out.0")))?;

    // Softmax / matmul at the exact cross-attn context shapes (row width 77
    // and K=77 are non-power-of-2 and were not covered by earlier probes).
    let softmax77 = |device: &Device| -> Result<Tensor> {
        let x = Tensor::from_vec(
            deterministic_f32_data(8 * 64 * 77, 0x9130),
            (8, 64, 77),
            device,
        )?;
        candle_nn::ops::softmax(&x, candle::D::Minus1)
    };
    run_probe("softmax_lastdim_8x64x77", device, softmax77);

    let mm_k77 = |device: &Device| -> Result<Tensor> {
        let a = Tensor::from_vec(
            deterministic_f32_data(8 * 64 * 77, 0x9131),
            (8, 64, 77),
            device,
        )?;
        let b = Tensor::from_vec(
            deterministic_f32_data(8 * 77 * 40, 0x9132),
            (8, 77, 40),
            device,
        )?;
        a.matmul(&b)
    };
    run_probe("mm_r3_b8_m64_k77_n40", device, mm_k77);

    // K sweep for the rank-3 batched matmul: localizes whether the defect
    // tracks K divisibility (K=77 fails, K=8/320 pass in earlier probes).
    for k in [64usize, 72, 77, 80, 88, 96] {
        let mm_k = move |device: &Device| -> Result<Tensor> {
            let a = Tensor::from_vec(
                deterministic_f32_data(8 * 64 * k, 0x9131),
                (8, 64, k),
                device,
            )?;
            let b = Tensor::from_vec(
                deterministic_f32_data(8 * k * 40, 0x9132),
                (8, k, 40),
                device,
            )?;
            a.matmul(&b)
        };
        let label = format!("mm_r3_b8_m64_k{k}_n40");
        run_probe(&label, device, mm_k);
    }
    // Same shape at batch=1 (Linear-style rank-3 layout).
    let mm_k77_b1 = |device: &Device| -> Result<Tensor> {
        let a = Tensor::from_vec(
            deterministic_f32_data(1 * 64 * 77, 0x9131),
            (1, 64, 77),
            device,
        )?;
        let b = Tensor::from_vec(
            deterministic_f32_data(1 * 77 * 40, 0x9132),
            (1, 77, 40),
            device,
        )?;
        a.matmul(&b)
    };
    run_probe("mm_r3_b1_m64_k77_n40", device, mm_k77_b1);

    for t in [1f64, 981.0] {
        let mk_temb = |vs: VarBuilder, dev: &Device| -> Result<Tensor> {
            let timesteps = stable_diffusion::embeddings::Timesteps::new(320, true, 0.0);
            let emb_in = (Tensor::ones(1, DType::F32, dev)? * t)?;
            let t_out = timesteps.forward(&emb_in)?;
            let mlp = stable_diffusion::embeddings::TimestepEmbedding::new(
                vs.pp("time_embedding"),
                320,
                1280,
            )?;
            mlp.forward(&t_out)
        };
        let temb_cpu = match mk_temb(vb_cpu.clone(), &Device::Cpu) {
            Ok(v) => v,
            Err(err) => {
                println!("[stage] temb_t{t}: CPU ERROR {err}");
                continue;
            }
        };
        let temb_dev = match mk_temb(vb_dev.clone(), device) {
            Ok(v) => v,
            Err(err) => {
                println!("[stage] temb_t{t}: DEV ERROR {err}");
                continue;
            }
        };
        let _ = report_max_diff(&temb_dev, &temb_cpu, &format!("stage_temb_t{t}"));

        let cmp = |label: String, dev_out: &Tensor, cpu_out: &Tensor| {
            let _ = report_max_diff(dev_out, cpu_out, &label);
        };

        let a_cpu = conv_in_cpu.forward(&x0_cpu)?;
        let a_dev = conv_in_dev.forward(&x0_dev)?;
        cmp(format!("stage_conv_in_t{t}"), &a_dev, &a_cpu);

        let b_cpu = res0_cpu.forward(&a_cpu, Some(&temb_cpu))?;
        let b_dev = res0_dev.forward(&a_dev, Some(&temb_dev))?;
        cmp(format!("stage_resnet0_t{t}"), &b_dev, &b_cpu);

        // SpatialTransformer sub-bisect: norm+proj_in, head assembly
        // (transpose/t/reshape), attn1 self-attn, attn2 cross-attn, ff
        // (GeGlu), head disassembly + proj_out, residual.
        let n_cpu = st_norm_cpu.forward(&b_cpu)?;
        let n_dev = st_norm_dev.forward(&b_dev)?;
        let p_cpu = st_proj_in_cpu.forward(&n_cpu)?;
        let p_dev = st_proj_in_dev.forward(&n_dev)?;
        cmp(format!("stsub_norm_proj_in_t{t}"), &p_dev, &p_cpu);

        let ha_cpu = p_cpu.transpose(1, 2)?.t()?.reshape((1, 64, 320))?;
        let ha_dev = p_dev.transpose(1, 2)?.t()?.reshape((1, 64, 320))?;
        cmp(format!("stsub_head_assembly_t{t}"), &ha_dev, &ha_cpu);

        let s1_cpu = (st_attn1_cpu.forward(&st_norm1_cpu.forward(&ha_cpu)?, None)? + &ha_cpu)?;
        let s1_dev = (st_attn1_dev.forward(&st_norm1_dev.forward(&ha_dev)?, None)? + &ha_dev)?;
        cmp(format!("stsub_attn1_t{t}"), &s1_dev, &s1_cpu);

        // CrossAttention (attn2) internals bisect: to_q/to_k/to_v linears,
        // head reshape of k, scores matmul, context matmul, to_out.
        let n2_cpu = st_norm2_cpu.forward(&s1_cpu)?;
        let n2_dev = st_norm2_dev.forward(&s1_dev)?;
        let q_cpu = at2_to_q_cpu.forward(&n2_cpu)?;
        let q_dev = at2_to_q_dev.forward(&n2_dev)?;
        cmp(format!("attn2_to_q_t{t}"), &q_dev, &q_cpu);

        let k_cpu = at2_to_k_cpu.forward(&text_emb_cpu)?;
        let k_dev = at2_to_k_dev.forward(&text_emb_dev)?;
        cmp(format!("attn2_to_k_t{t}"), &k_dev, &k_cpu);

        let v_cpu = at2_to_v_cpu.forward(&text_emb_cpu)?;
        let v_dev = at2_to_v_dev.forward(&text_emb_dev)?;
        cmp(format!("attn2_to_v_t{t}"), &v_dev, &v_cpu);

        let qh_cpu = q_cpu
            .reshape((1, 64, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 64, 40))?;
        let qh_dev = q_dev
            .reshape((1, 64, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 64, 40))?;
        let kh_cpu = k_cpu
            .reshape((1, 77, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 77, 40))?;
        let kh_dev = k_dev
            .reshape((1, 77, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 77, 40))?;
        cmp(format!("attn2_k_head_reshape_t{t}"), &kh_dev, &kh_cpu);

        let scale = 1.0 / f64::sqrt(40.0);
        let scores_cpu = qh_cpu.matmul(&(kh_cpu.t()? * scale)?)?;
        let scores_dev = qh_dev.matmul(&(kh_dev.t()? * scale)?)?;
        cmp(format!("attn2_scores_t{t}"), &scores_dev, &scores_cpu);

        let vh_cpu = v_cpu
            .reshape((1, 77, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 77, 40))?;
        let vh_dev = v_dev
            .reshape((1, 77, 8, 40))?
            .transpose(1, 2)?
            .reshape((8, 77, 40))?;
        let ctx_cpu = candle_nn::ops::softmax(&scores_cpu, candle::D::Minus1)?
            .matmul(&vh_cpu)?;
        let ctx_dev = candle_nn::ops::softmax(&scores_dev, candle::D::Minus1)?
            .matmul(&vh_dev)?;
        cmp(format!("attn2_ctx_t{t}"), &ctx_dev, &ctx_cpu);

        let out_cpu = ctx_cpu
            .reshape((1, 8, 64, 40))?
            .transpose(1, 2)?
            .reshape((1, 64, 320))?;
        let out_dev = ctx_dev
            .reshape((1, 8, 64, 40))?
            .transpose(1, 2)?
            .reshape((1, 64, 320))?;
        let o_cpu = at2_to_out_cpu.forward(&out_cpu)?;
        let o_dev = at2_to_out_dev.forward(&out_dev)?;
        cmp(format!("attn2_to_out_t{t}"), &o_dev, &o_cpu);

        let s2_cpu = (st_attn2_cpu.forward(&st_norm2_cpu.forward(&s1_cpu)?, Some(&text_emb_cpu))?
            + &s1_cpu)?;
        let s2_dev = (st_attn2_dev.forward(&st_norm2_dev.forward(&s1_dev)?, Some(&text_emb_dev))?
            + &s1_dev)?;
        cmp(format!("stsub_attn2_t{t}"), &s2_dev, &s2_cpu);

        let g3_cpu = st_norm3_cpu.forward(&s2_cpu)?;
        let g3_dev = st_norm3_dev.forward(&s2_dev)?;
        cmp(format!("stsub_ff_norm3_t{t}"), &g3_dev, &g3_cpu);

        let ge_cpu = st_geglu_cpu.forward(&g3_cpu)?;
        let ge_dev = st_geglu_dev.forward(&g3_dev)?;
        cmp(format!("stsub_ff_geglu_lin_t{t}"), &ge_dev, &ge_cpu);

        let ge_parts_cpu = ge_cpu.chunk(2, candle::D::Minus1)?;
        let ge_parts_dev = ge_dev.chunk(2, candle::D::Minus1)?;
        let gate_cpu = ge_parts_cpu[1].gelu()?;
        let gate_dev = ge_parts_dev[1].gelu()?;
        cmp(format!("stsub_ff_gate_gelu_t{t}"), &gate_dev, &gate_cpu);

        let hid_cpu = (&ge_parts_cpu[0] * &gate_cpu)?;
        let hid_dev = (&ge_parts_dev[0] * &gate_dev)?;
        cmp(format!("stsub_ff_product_t{t}"), &hid_dev, &hid_cpu);

        let ff_out_cpu = (st_ff_lin_cpu.forward(&hid_cpu)? + &s2_cpu)?;
        let ff_out_dev = (st_ff_lin_dev.forward(&hid_dev)? + &s2_dev)?;
        cmp(format!("stsub_ff_t{t}"), &ff_out_dev, &ff_out_cpu);

        let dis_cpu = ff_out_cpu.reshape((1, 8, 8, 320))?.t()?.transpose(1, 2)?;
        let dis_dev = ff_out_dev.reshape((1, 8, 8, 320))?.t()?.transpose(1, 2)?;
        let po_cpu = st_proj_out_cpu.forward(&dis_cpu)?;
        let po_dev = st_proj_out_dev.forward(&dis_dev)?;
        cmp(format!("stsub_proj_out_t{t}"), &po_dev, &po_cpu);

        let fin_cpu = (po_cpu + &b_cpu)?;
        let fin_dev = (po_dev + &b_dev)?;
        cmp(format!("stsub_residual_t{t}"), &fin_dev, &fin_cpu);

        let c_cpu = st0_cpu.forward(&b_cpu, Some(&text_emb_cpu))?;
        let c_dev = st0_dev.forward(&b_dev, Some(&text_emb_dev))?;
        cmp(format!("stage_transformer0_t{t}"), &c_dev, &c_cpu);

        let d_cpu = res1_cpu.forward(&c_cpu, Some(&temb_cpu))?;
        let d_dev = res1_dev.forward(&c_dev, Some(&temb_dev))?;
        cmp(format!("stage_resnet1_t{t}"), &d_dev, &d_cpu);

        let e_cpu = down_cpu.forward(&d_cpu)?;
        let e_dev = down_dev.forward(&d_dev)?;
        cmp(format!("stage_downsample_t{t}"), &e_dev, &e_cpu);

        // Mid block is probed in isolation (det 1280-channel 4x4 input): the
        // real graph reaches it only after down_blocks 1..3 (320 -> 1280).
        let m_cpu = Tensor::from_vec(
            deterministic_f32_data(1280 * 4 * 4, 0x9140),
            (1, 1280, 4, 4),
            &Device::Cpu,
        )?;
        let m_dev = Tensor::from_vec(
            deterministic_f32_data(1280 * 4 * 4, 0x9140),
            (1, 1280, 4, 4),
            device,
        )?;
        let f_cpu = mid_cpu.forward(&m_cpu, Some(&temb_cpu), Some(&text_emb_cpu))?;
        let f_dev = mid_dev.forward(&m_dev, Some(&temb_dev), Some(&text_emb_dev))?;
        cmp(format!("stage_mid_block_t{t}"), &f_dev, &f_cpu);
    }

    let unet_cpu = sd_config.build_unet(&unet_weights, &Device::Cpu, 4, false, DType::F32)?;
    let unet_dev = sd_config.build_unet(&unet_weights, device, 4, false, DType::F32)?;

    for (size, timestep) in [(8usize, 1f64), (8, 981.0), (16, 981.0), (32, 981.0), (64, 981.0)] {
        let n = 4 * size * size;
        let latent_cpu = Tensor::from_vec(
            deterministic_f32_data(n, 0x5D1_5EED),
            (1, 4, size, size),
            &Device::Cpu,
        )?;
        let latent_dev = Tensor::from_vec(
            deterministic_f32_data(n, 0x5D1_5EED),
            (1, 4, size, size),
            device,
        )?;
        let label = format!("unet_bisect_{size}x{size}_t{timestep}");
        match (|| -> Result<f32> {
            let dev_out = unet_dev.forward(&latent_dev, timestep, &text_emb_dev)?;
            device.synchronize()?;
            let cpu_out = unet_cpu.forward(&latent_cpu, timestep, &text_emb_cpu)?;
            report_max_diff(&dev_out, &cpu_out, &label)
        })() {
            Ok(diff) => println!("[bisect] {label}: max_abs_diff={diff:.3e}"),
            Err(err) => println!("[bisect] {label}: ERROR {err}"),
        }
    }
    Ok(())
}
