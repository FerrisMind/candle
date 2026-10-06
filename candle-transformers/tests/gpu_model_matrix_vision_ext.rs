//! Extended GPU certification matrix for vision model families (worker W1).
//!
//! Mirrors the structure of `gpu_model_matrix.rs` but focuses on three vision
//! families backed by local weights (no network downloads):
//!
//! - `resnet18_case` / `resnet50_case`: ResNet image classification, final
//!   logits `[1, 1000]` compared CPU vs GPU.
//! - `mobile_sam_case`: MobileSAM (TinyViT-5M image encoder) image embedding
//!   `[1, 256, 64, 64]` plus mask-decoder logits / IoU compared CPU vs GPU.
//! - `clip_case`: CLIP ViT-B/32 text features and image features compared
//!   CPU vs GPU for a fixed token-id sequence and a deterministic image.
//!
//! All inputs are synthetic and deterministic, so the same weights are run on
//! CPU and on the GPU backend and the outputs must match within tolerance.
//! The zero-CPU-fallback policy from `native_required` applies.

mod support;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2dConfig, ModuleT, VarBuilder};
use candle_transformers::models::{
    clip::{self, ClipConfig},
    resnet,
    segment_anything::sam,
};
use std::path::PathBuf;
use std::time::Instant;
use support::{assert_close_tensors, deterministic_f32_data, native_required, TestBackend};

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
const CASE_FILTER_ENV: &str = "CANDLE_GPU_VISION_EXT_CASE_FILTER";

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
type ModelCaseFn = fn(&Device) -> Result<()>;

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual GPU vision certification matrix"]
fn gpu_model_matrix_vision_ext_wgpu() -> Result<()> {
    native_required(
        "gpu_model_matrix_vision_ext_wgpu",
        TestBackend::Wgpu,
        run_vision_matrix,
    )
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual GPU vision certification matrix"]
fn gpu_model_matrix_vision_ext_vulkan() -> Result<()> {
    native_required(
        "gpu_model_matrix_vision_ext_vulkan",
        TestBackend::Vulkan,
        run_vision_matrix,
    )
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_vision_matrix(device: &Device) -> Result<()> {
    if cfg!(debug_assertions) {
        println!(
            "gpu vision model matrix is running in the debug test profile; for certification runtime use `cargo test --release`"
        );
    }

    let requested_cases = requested_case_names();
    let cases: [(&str, ModelCaseFn); 5] = [
        ("resnet18_case", resnet18_case),
        ("resnet50_case", resnet50_case),
        ("mobile_sam_case", mobile_sam_case),
        ("clip_case", clip_case),
        ("op_diagnosis_case", op_diagnosis_case),
    ];
    let mut ran_any = false;
    let mut failures: Vec<String> = Vec::new();
    for (name, case_fn) in cases {
        if !case_is_requested(name, requested_cases.as_deref()) {
            println!(
                "skipping {name} on {} due to {CASE_FILTER_ENV}",
                backend_name(device)
            );
            continue;
        }
        ran_any = true;
        run_case(name, device, case_fn, &mut failures);
    }
    if !ran_any {
        candle::bail!(
            "{CASE_FILTER_ENV} did not match any vision GPU model case: resnet18_case, resnet50_case, mobile_sam_case, clip_case, op_diagnosis_case"
        );
    }
    if !failures.is_empty() {
        candle::bail!("vision matrix cases failed on {}: {}", backend_name(device), failures.join(", "));
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
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

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn case_is_requested(name: &str, requested_cases: Option<&[String]>) -> bool {
    match requested_cases {
        None => true,
        Some(requested_cases) => requested_cases.iter().any(|requested| requested == name),
    }
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

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_case(name: &str, device: &Device, f: fn(&Device) -> Result<()>, failures: &mut Vec<String>) {
    println!("running {name} on {}", backend_name(device));
    let start = Instant::now();
    match f(device) {
        Ok(()) => println!(
            "{name} finished in {:.2?}; fallback count after {name}: {} [PASS]",
            start.elapsed(),
            fallback_count(device)
        ),
        Err(err) => {
            println!(
                "{name} failed after {:.2?}; fallback count after {name}: {} [FAIL] error: {err}",
                start.elapsed(),
                fallback_count(device)
            );
            failures.push(name.to_string());
        }
    }
}

/// Parity tolerances for the CPU-vs-GPU comparisons.
///
/// Policy: f32 models start at 1e-3/1e-3. `CANDLE_VISION_EXT_TOL` overrides
/// both atol and rtol for supplementary evidence runs only; the official
/// certification runs use the default.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn parity_tolerances() -> (f32, f32) {
    match std::env::var("CANDLE_VISION_EXT_TOL") {
        Ok(value) => {
            let tol: f32 = value.parse().unwrap_or(1e-3);
            println!(
                "CANDLE_VISION_EXT_TOL={value} supplied: using supplementary tolerance atol=rtol={tol}"
            );
            (tol, tol)
        }
        Err(_) => (1e-3, 1e-3),
    }
}

/// Compare GPU output against the CPU reference, printing the observed max
/// absolute difference (and where it happened) before enforcing the tolerance.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn report_close(
    actual: &Tensor,
    expected: &Tensor,
    atol: f32,
    rtol: f32,
    label: &str,
) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            actual.dims(),
            expected.dims()
        );
    }
    let actual_f32 = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected_f32 = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_diff = 0f32;
    let mut max_idx = 0usize;
    let mut max_rel = 0f32;
    for (idx, (a, e)) in actual_f32.iter().zip(expected_f32.iter()).enumerate() {
        let diff = (a - e).abs();
        // Denominator floored at atol so near-zero references do not explode rel.
        let rel = diff / e.abs().max(atol);
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    println!(
        "{label}: max_abs_diff={max_diff:.3e} at idx={max_idx} max_rel={max_rel:.3e} (atol={atol} rtol={rtol})"
    );
    assert_close_tensors(actual, expected, atol, rtol, label)
}

/// Print the observed CPU-vs-device max differences without asserting;
/// used by the op-level diagnosis case.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn print_max_diff(actual: &Tensor, expected: &Tensor, label: &str) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle::bail!(
            "{label}: shape mismatch, got {:?}, expected {:?}",
            actual.dims(),
            expected.dims()
        );
    }
    let actual_f32 = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected_f32 = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_diff = 0f32;
    let mut max_idx = 0usize;
    let mut max_rel = 0f32;
    let mut max_val = 0f32;
    for (idx, (a, e)) in actual_f32.iter().zip(expected_f32.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = diff / e.abs().max(1e-6);
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
            max_rel = rel;
            max_val = *e;
        }
    }
    println!(
        "{label}: max_abs_diff={max_diff:.3e} at idx={max_idx} (expected value {max_val:.4e}, rel {max_rel:.3e})"
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn env_dir(env: &str, default: &str) -> PathBuf {
    std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn resnet_dir() -> PathBuf {
    env_dir("CANDLE_RESNET_DIR", r"G:\models\candle-resnet")
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn sam_dir() -> PathBuf {
    env_dir("CANDLE_SAM_DIR", r"G:\models\candle-sam")
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn clip_dir() -> PathBuf {
    env_dir(
        "CANDLE_CLIP_DIR",
        r"C:\Users\PC\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\b33cedfd0df4e43b8238760678fcc89e1a0d38b3",
    )
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn require_file(path: &PathBuf) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        candle::bail!(
            "required local weight file is missing: {} (set the matching CANDLE_*_DIR env var)",
            path.display()
        )
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn resnet_case(device: &Device, depth: u32) -> Result<()> {
    let weights_path = resnet_dir().join(format!("resnet{depth}.safetensors"));
    require_file(&weights_path)?;
    let cpu = Device::Cpu;

    // F32 on both sides: the local lmz/candle-resnet checkpoints are f32.
    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let class_count = 1000usize; // imagenet CLASS_COUNT
    let cpu_model = match depth {
        18 => resnet::resnet18(class_count, cpu_vb)?,
        50 => resnet::resnet50(class_count, cpu_vb)?,
        other => candle::bail!("unsupported resnet depth {other}"),
    };
    let dev_model = match depth {
        18 => resnet::resnet18(class_count, dev_vb)?,
        50 => resnet::resnet50(class_count, dev_vb)?,
        other => candle::bail!("unsupported resnet depth {other}"),
    };

    // deterministic_f32_data yields values in [-2, 2], the same range as
    // imagenet-normalized pixels (mean/std normalization), so no rescale needed.
    let image = deterministic_f32_data(3 * 224 * 224, 0xA11CE + depth as u64);
    let image_cpu = Tensor::from_vec(image.clone(), (1, 3, 224, 224), &cpu)?;
    let image_dev = Tensor::from_vec(image, (1, 3, 224, 224), device)?;

    let logits_cpu = cpu_model.forward(&image_cpu)?;
    let logits_dev = dev_model.forward(&image_dev)?;
    let (atol, rtol) = parity_tolerances();
    report_close(
        &logits_dev,
        &logits_cpu,
        atol,
        rtol,
        &format!("resnet{depth}_logits"),
    )?;
    assert_eq!(logits_dev.dims(), [1, 1000]);
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn resnet18_case(device: &Device) -> Result<()> {
    resnet_case(device, 18)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn resnet50_case(device: &Device) -> Result<()> {
    resnet_case(device, 50)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn mobile_sam_case(device: &Device) -> Result<()> {
    let weights_path = sam_dir().join("mobile_sam-tiny-vitt.safetensors");
    require_file(&weights_path)?;
    let cpu = Device::Cpu;

    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let cpu_sam = sam::Sam::new_tiny(cpu_vb)?; // TinyViT-5M image encoder
    let dev_sam = sam::Sam::new_tiny(dev_vb)?;

    // Deterministic image in the 0..255 pixel range the SAM preprocessor
    // expects (pixel_mean/pixel_std normalization happens inside the model).
    // 1024x1024 is the image size the TinyViT encoder was built for; the
    // model's own preprocess pads any smaller input back up to this size.
    let side = sam::IMAGE_SIZE; // 1024
    let image = deterministic_f32_data(3 * side * side, 0x5AEEED)
        .into_iter()
        .map(|v| (v + 2.0) * 63.75) // [-2,2] -> [0,255]
        .collect::<Vec<f32>>();
    let image_cpu = Tensor::from_vec(image.clone(), (3, side, side), &cpu)?;
    let image_dev = Tensor::from_vec(image, (3, side, side), device)?;

    let start = Instant::now();
    let embeddings_cpu = cpu_sam.embeddings(&image_cpu)?;
    println!("cpu tiny_vit image embedding in {:.2?}", start.elapsed());
    let start = Instant::now();
    let embeddings_dev = dev_sam.embeddings(&image_dev)?;
    println!("gpu tiny_vit image embedding in {:.2?}", start.elapsed());
    let (atol, rtol) = parity_tolerances();
    report_close(
        &embeddings_dev,
        &embeddings_cpu,
        atol,
        rtol,
        "sam_image_embeddings",
    )?;

    // Exercise the prompt encoder + mask decoder on top of the embeddings.
    let points = [(0.5f64, 0.5f64, true), (0.25, 0.25, false)];
    let (mask_cpu, iou_cpu) =
        cpu_sam.forward_for_embeddings(&embeddings_cpu, side, side, &points, false)?;
    let (mask_dev, iou_dev) =
        dev_sam.forward_for_embeddings(&embeddings_dev, side, side, &points, false)?;
    report_close(
        &mask_dev,
        &mask_cpu,
        atol,
        rtol,
        "sam_mask_decoder_logits",
    )?;
    report_close(&iou_dev, &iou_cpu, atol, rtol, "sam_iou_predictions")?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn clip_case(device: &Device) -> Result<()> {
    let weights_path = clip_dir().join("model.safetensors");
    require_file(&weights_path)?;
    let cpu = Device::Cpu;

    let config = ClipConfig::vit_base_patch32();
    // F32 on both sides: the openai/clip-vit-base-patch32 checkpoint is f32.
    let cpu_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F32, &cpu)?
    };
    let dev_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let cpu_model = clip::ClipModel::new(cpu_vb, &config)?;
    let dev_model = clip::ClipModel::new(dev_vb, &config)?;

    // Deterministic image tensor in [-1, 1], the CLIP preprocessor output range.
    let image = deterministic_f32_data(3 * config.image_size * config.image_size, 0xC11C0)
        .into_iter()
        .map(|v| v / 2.0) // [-2,2] -> [-1,1]
        .collect::<Vec<f32>>();
    let image_cpu = Tensor::from_vec(
        image.clone(),
        (1, 3, config.image_size, config.image_size),
        &cpu,
    )?;
    let image_dev = Tensor::from_vec(
        image,
        (1, 3, config.image_size, config.image_size),
        device,
    )?;
    // Fixed token-id sequence: BOS + "a photo of a candle" + EOS, padded with
    // EOS to the CLIP context length of 77. Ids only need to be valid vocab
    // entries; CPU and GPU see the exact same sequence.
    let seq: [u32; 7] = [49406, 320, 1125, 522, 320, 2368, 49407];
    let mut ids = vec![0u32; config.text_config.max_position_embeddings]; // 77
    ids[..seq.len()].copy_from_slice(&seq);
    for id in ids[seq.len()..].iter_mut() {
        *id = 49407; // <|endoftext|> pad
    }
    let seq_len = ids.len();
    let ids_cpu = Tensor::from_vec(ids.clone(), (1, seq_len), &cpu)?;
    let ids_dev = Tensor::from_vec(ids, (1, seq_len), device)?;

    let text_cpu = cpu_model.get_text_features(&ids_cpu)?;
    let text_dev = dev_model.get_text_features(&ids_dev)?;
    let (atol, rtol) = parity_tolerances();
    report_close(&text_dev, &text_cpu, atol, rtol, "clip_text_features")?;

    let image_features_cpu = cpu_model.get_image_features(&image_cpu)?;
    let image_features_dev = dev_model.get_image_features(&image_dev)?;
    report_close(
        &image_features_dev,
        &image_features_cpu,
        atol,
        rtol,
        "clip_image_features",
    )?;

    // Joint contrastive logits over the single text/image pair (raw inputs).
    let (logits_per_text_cpu, logits_per_image_cpu) = cpu_model.forward(&image_cpu, &ids_cpu)?;
    let (logits_per_text_dev, logits_per_image_dev) = dev_model.forward(&image_dev, &ids_dev)?;
    report_close(
        &logits_per_text_dev,
        &logits_per_text_cpu,
        atol,
        rtol,
        "clip_logits_per_text",
    )?;
    report_close(
        &logits_per_image_dev,
        &logits_per_image_cpu,
        atol,
        rtol,
        "clip_logits_per_image",
    )?;
    Ok(())
}

/// Op-level CPU-vs-device diagnosis, opt-in via `CANDLE_VISION_EXT_OPS_DIAG=1`.
///
/// Not part of the official pass/fail matrix: it exists to localize backend
/// numeric regressions seen by the model cases (which op diverges, and by how
/// much). No-op unless the env var is set.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn op_diagnosis_case(device: &Device) -> Result<()> {
    if std::env::var_os("CANDLE_VISION_EXT_OPS_DIAG").is_none() {
        println!("op_diagnosis_case: skipped (set CANDLE_VISION_EXT_OPS_DIAG=1)");
        return Ok(());
    }
    let cpu = Device::Cpu;
    println!("op diagnosis on {}", backend_name(device));

    // 1. plain f32 matmul
    let data = deterministic_f32_data(128 * 128, 0xD1A0);
    let a_cpu = Tensor::from_vec(data.clone(), (128, 128), &cpu)?;
    let a_dev = Tensor::from_vec(data, (128, 128), device)?;
    let data = deterministic_f32_data(128 * 128, 0xD1A1);
    let b_cpu = Tensor::from_vec(data.clone(), (128, 128), &cpu)?;
    let b_dev = Tensor::from_vec(data, (128, 128), device)?;
    print_max_diff(&a_dev.matmul(&b_dev)?, &a_cpu.matmul(&b_cpu)?, "diag_matmul")?;

    // 2. chained matmuls (transformer-like accumulation depth)
    let mut x_cpu = a_cpu.clone();
    let mut x_dev = a_dev.clone();
    for _ in 0..12 {
        x_cpu = x_cpu.matmul(&b_cpu)?;
        x_dev = x_dev.matmul(&b_dev)?;
    }
    print_max_diff(&x_dev, &x_cpu, "diag_matmul_chain12")?;

    // 3. standard conv2d (resnet-like)
    let cfg = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let w = deterministic_f32_data(32 * 16 * 3 * 3, 0xD1A2);
    let conv_cpu = candle_nn::conv2d_no_bias(16, 32, 3, cfg, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w.clone(), (32, 16, 3, 3), &cpu)?)]
            .into_iter()
            .collect(),
        DType::F32,
        &cpu,
    ))?;
    let conv_dev = candle_nn::conv2d_no_bias(16, 32, 3, cfg, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w, (32, 16, 3, 3), device)?)]
            .into_iter()
            .collect(),
        DType::F32,
        device,
    ))?;
    let data = deterministic_f32_data(16 * 64 * 64, 0xD1A3);
    let inp_cpu = Tensor::from_vec(data.clone(), (1, 16, 64, 64), &cpu)?;
    let inp_dev = Tensor::from_vec(data, (1, 16, 64, 64), device)?;
    print_max_diff(
        &inp_dev.apply(&conv_dev)?,
        &inp_cpu.apply(&conv_cpu)?,
        "diag_conv2d",
    )?;

    // 4. depthwise grouped conv (TinyViT MBConv-like)
    let cfg_dw = Conv2dConfig {
        padding: 1,
        groups: 16,
        ..Default::default()
    };
    let w = deterministic_f32_data(16 * 3 * 3, 0xD1A4);
    let conv_cpu = candle_nn::conv2d_no_bias(16, 16, 3, cfg_dw, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w.clone(), (16, 1, 3, 3), &cpu)?)]
            .into_iter()
            .collect(),
        DType::F32,
        &cpu,
    ))?;
    let conv_dev = candle_nn::conv2d_no_bias(16, 16, 3, cfg_dw, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w, (16, 1, 3, 3), device)?)]
            .into_iter()
            .collect(),
        DType::F32,
        device,
    ))?;
    print_max_diff(
        &inp_dev.apply(&conv_dev)?,
        &inp_cpu.apply(&conv_cpu)?,
        "diag_depthwise_conv2d",
    )?;

    // 5. batch_norm eval (resnet/TinyViT)
    let chans = 16usize;
    let bn_map = |dev: &Device, seed: u64| -> Result<_> {
        Ok([
            (
                "weight".to_string(),
                Tensor::from_vec(
                    deterministic_f32_data(chans, seed).iter().map(|v| v / 4.0 + 1.0).collect::<Vec<f32>>(),
                    (chans,),
                    dev,
                )?,
            ),
            ("bias".to_string(), Tensor::from_vec(deterministic_f32_data(chans, seed + 1), (chans,), dev)?),
            ("running_mean".to_string(), Tensor::from_vec(deterministic_f32_data(chans, seed + 2), (chans,), dev)?),
            ("running_var".to_string(), Tensor::from_vec(
                deterministic_f32_data(chans, seed + 3).iter().map(|v| v.abs() + 0.5).collect::<Vec<f32>>(),
                (chans,),
                dev,
            )?),
        ]
        .into_iter()
        .collect())
    };
    let bn_cpu = candle_nn::batch_norm(chans, 1e-5, VarBuilder::from_tensors(bn_map(&cpu, 0xD1A5)?, DType::F32, &cpu))?;
    let bn_dev = candle_nn::batch_norm(chans, 1e-5, VarBuilder::from_tensors(bn_map(device, 0xD1A5)?, DType::F32, device))?;
    print_max_diff(
        &bn_dev.forward_t(&inp_dev, false)?,
        &bn_cpu.forward_t(&inp_cpu, false)?,
        "diag_batch_norm",
    )?;

    // 6. layer_norm (CLIP)
    let size = 64usize;
    let ln_map = |dev: &Device, seed: u64| -> Result<_> {
        Ok([
            ("weight".to_string(), Tensor::from_vec(deterministic_f32_data(size, seed).iter().map(|v| v / 4.0 + 1.0).collect::<Vec<f32>>(), (size,), dev)?),
            ("bias".to_string(), Tensor::from_vec(deterministic_f32_data(size, seed + 1), (size,), dev)?),
        ]
        .into_iter()
        .collect())
    };
    let ln_cpu = candle_nn::layer_norm(size, 1e-5, VarBuilder::from_tensors(ln_map(&cpu, 0xD1A6)?, DType::F32, &cpu))?;
    let ln_dev = candle_nn::layer_norm(size, 1e-5, VarBuilder::from_tensors(ln_map(device, 0xD1A6)?, DType::F32, device))?;
    let data = deterministic_f32_data(4 * 64, 0xD1A7);
    let flat_cpu = Tensor::from_vec(data.clone(), (4, 64), &cpu)?;
    let flat_dev = Tensor::from_vec(data, (4, 64), device)?;
    print_max_diff(
        &flat_dev.apply(&ln_dev)?,
        &flat_cpu.apply(&ln_cpu)?,
        "diag_layer_norm",
    )?;

    // 7. quick-gelu (CLIP activation) and softmax
    let data = deterministic_f32_data(4096, 0xD1A8);
    let v_cpu = Tensor::from_vec(data.clone(), (4096,), &cpu)?;
    let v_dev = Tensor::from_vec(data, (4096,), device)?;
    let qg = |t: &Tensor| -> Result<Tensor> {
        let s = t.affine(1.702f64, 0.0)?;
        let sig = candle_nn::ops::sigmoid(&s)?;
        sig.mul(t)
    };
    print_max_diff(&qg(&v_dev)?, &qg(&v_cpu)?, "diag_quick_gelu")?;
    print_max_diff(
        &candle_nn::ops::softmax(&v_dev.reshape((8, 512))?, 1)?,
        &candle_nn::ops::softmax(&v_cpu.reshape((8, 512))?, 1)?,
        "diag_softmax",
    )?;

    // 8. window-attention-style layout shuffle + matmul (TinyViT window partition)
    let data = deterministic_f32_data(64 * 8 * 8 * 64, 0xD1A9);
    let w_cpu = Tensor::from_vec(data.clone(), (1, 64, 8, 8, 64), &cpu)?;
    let w_dev = Tensor::from_vec(data, (1, 64, 8, 8, 64), device)?;
    let data = deterministic_f32_data(64 * 64, 0xD1AA);
    let b64_cpu = Tensor::from_vec(data.clone(), (64, 64), &cpu)?;
    let b64_dev = Tensor::from_vec(data, (64, 64), device)?;
    let shuffle = |t: &Tensor, b: &Tensor| -> Result<Tensor> {
        let t = t.transpose(1, 2)?.contiguous()?; // (1, 8, 64, 8, 64)
        let t = t.reshape((64, 64, 64))?.contiguous()?;
        let b_full = b.unsqueeze(0)?.broadcast_as((64, 64, 64))?;
        t.matmul(&b_full)
    };
    print_max_diff(
        &shuffle(&w_dev, &b64_dev)?,
        &shuffle(&w_cpu, &b64_cpu)?,
        "diag_window_shuffle_matmul",
    )?;

    // 9. index_select / embedding gather (CLIP token+position embeddings,
    //    TinyViT relative-position table) at small and HF-vocab sizes.
    for (rows, tag) in [(1024usize, "small"), (49408usize, "hf_vocab")] {
        let width = 768usize;
        let table = deterministic_f32_data(rows * width, 0xD1AB + rows as u64);
        let table_cpu = Tensor::from_vec(table.clone(), (rows, width), &cpu)?;
        let table_dev = Tensor::from_vec(table, (rows, width), device)?;
        let ids_len = 77usize;
        let ids: Vec<u32> = (0..ids_len)
            .map(|i| ((i as u64 * 7919 + rows as u64) % rows as u64) as u32)
            .collect();
        let ids_cpu = Tensor::from_vec(ids.clone(), (ids_len,), &cpu)?;
        let ids_dev = Tensor::from_vec(ids, (ids_len,), device)?;
        print_max_diff(
            &table_dev.index_select(&ids_dev, 0)?,
            &table_cpu.index_select(&ids_cpu, 0)?,
            &format!("diag_index_select_{tag}"),
        )?;
    }

    // 10. realistic conv sizes (resnet stage-like im2col matmul dimensions)
    let cfg_big = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let w = deterministic_f32_data(128 * 64 * 3 * 3, 0xD1AC);
    let conv_cpu = candle_nn::conv2d_no_bias(64, 128, 3, cfg_big, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w.clone(), (128, 64, 3, 3), &cpu)?)]
            .into_iter()
            .collect(),
        DType::F32,
        &cpu,
    ))?;
    let conv_dev = candle_nn::conv2d_no_bias(64, 128, 3, cfg_big, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w, (128, 64, 3, 3), device)?)]
            .into_iter()
            .collect(),
        DType::F32,
        device,
    ))?;
    let data = deterministic_f32_data(64 * 112 * 112, 0xD1AD);
    let inp_cpu = Tensor::from_vec(data.clone(), (1, 64, 112, 112), &cpu)?;
    let inp_dev = Tensor::from_vec(data, (1, 64, 112, 112), device)?;
    print_max_diff(
        &inp_dev.apply(&conv_dev)?,
        &inp_cpu.apply(&conv_cpu)?,
        "diag_conv2d_resnet_stage",
    )?;

    // 11. erf-based gelu (TinyViT uses Tensor::gelu everywhere)
    let data = deterministic_f32_data(4096, 0xD1AE);
    let v_cpu = Tensor::from_vec(data.clone(), (4096,), &cpu)?;
    let v_dev = Tensor::from_vec(data, (4096,), device)?;
    print_max_diff(&v_dev.gelu()?, &v_cpu.gelu()?, "diag_gelu_erf")?;

    // 12. relu (resnet)
    print_max_diff(&v_dev.relu()?, &v_cpu.relu()?, "diag_relu")?;

    // 13. max_pool2d (resnet stem)
    let data = deterministic_f32_data(64 * 112 * 112, 0xD1AF);
    let p_cpu = Tensor::from_vec(data.clone(), (1, 64, 112, 112), &cpu)?;
    let p_dev = Tensor::from_vec(data, (1, 64, 112, 112), device)?;
    print_max_diff(
        &p_dev.max_pool2d_with_stride(3, 2)?,
        &p_cpu.max_pool2d_with_stride(3, 2)?,
        "diag_max_pool2d",
    )?;

    // 14. global mean reductions (resnet head)
    let data = deterministic_f32_data(512 * 7 * 7, 0xD1B0);
    let r_cpu = Tensor::from_vec(data.clone(), (1, 512, 7, 7), &cpu)?;
    let r_dev = Tensor::from_vec(data, (1, 512, 7, 7), device)?;
    print_max_diff(
        &r_dev.mean(3)?.mean(2)?,
        &r_cpu.mean(3)?.mean(2)?,
        "diag_mean_reduce",
    )?;

    // 15. broadcast add (attention masks / residual shapes)
    let data = deterministic_f32_data(12 * 77 * 77, 0xD1B1);
    let m_cpu = Tensor::from_vec(data.clone(), (1, 12, 77, 77), &cpu)?;
    let m_dev = Tensor::from_vec(data, (1, 12, 77, 77), device)?;
    let mask = deterministic_f32_data(77 * 77, 0xD1B2);
    let mask_cpu = Tensor::from_vec(mask.clone(), (1, 1, 77, 77), &cpu)?;
    let mask_dev = Tensor::from_vec(mask, (1, 1, 77, 77), device)?;
    print_max_diff(
        &m_dev.broadcast_add(&mask_dev)?,
        &m_cpu.broadcast_add(&mask_cpu)?,
        "diag_broadcast_add",
    )?;

    // 16. transformer-realistic matmul shapes (CLIP text: 768-wide, 12 heads)
    let data = deterministic_f32_data(77 * 768, 0xD1B3);
    let x_cpu = Tensor::from_vec(data.clone(), (77, 768), &cpu)?;
    let x_dev = Tensor::from_vec(data, (77, 768), device)?;
    let data = deterministic_f32_data(768 * 768, 0xD1B4);
    let w768_cpu = Tensor::from_vec(data.clone(), (768, 768), &cpu)?;
    let w768_dev = Tensor::from_vec(data, (768, 768), device)?;
    print_max_diff(
        &x_dev.matmul(&w768_dev)?,
        &x_cpu.matmul(&w768_cpu)?,
        "diag_matmul_77x768_768x768",
    )?;
    let data = deterministic_f32_data(768 * 3072, 0xD1B5);
    let w3072_cpu = Tensor::from_vec(data.clone(), (768, 3072), &cpu)?;
    let w3072_dev = Tensor::from_vec(data, (768, 3072), device)?;
    print_max_diff(
        &x_dev.matmul(&w3072_dev)?,
        &x_cpu.matmul(&w3072_cpu)?,
        "diag_matmul_77x768_768x3072",
    )?;
    // batched attention: q@k^T and attn@v
    let data = deterministic_f32_data(12 * 77 * 64, 0xD1B6);
    let q_cpu = Tensor::from_vec(data.clone(), (12, 77, 64), &cpu)?;
    let q_dev = Tensor::from_vec(data, (12, 77, 64), device)?;
    let data = deterministic_f32_data(12 * 64 * 77, 0xD1B7);
    let k_cpu = Tensor::from_vec(data.clone(), (12, 64, 77), &cpu)?;
    let k_dev = Tensor::from_vec(data, (12, 64, 77), device)?;
    let data = deterministic_f32_data(12 * 77 * 64, 0xD1B8);
    let v_cpu = Tensor::from_vec(data.clone(), (12, 77, 64), &cpu)?;
    let v_dev = Tensor::from_vec(data, (12, 77, 64), device)?;
    let attn_scores = |q: &Tensor, kt: &Tensor| -> Result<Tensor> {
        let scores = q.matmul(&kt.contiguous()?)?.contiguous()?;
        Ok(candle_nn::ops::softmax_last_dim(&scores)?)
    };
    let attn_cpu = attn_scores(&q_cpu, &k_cpu)?;
    let attn_dev = attn_scores(&q_dev, &k_dev)?;
    print_max_diff(&attn_dev, &attn_cpu, "diag_attn_qk_softmax")?;
    print_max_diff(
        &attn_dev.matmul(&v_dev)?,
        &attn_cpu.matmul(&v_cpu)?,
        "diag_attn_av",
    )?;

    // 17. weight transfer: mmap'd safetensors -> device copy must be bit-exact
    let resnet_w = resnet_dir().join("resnet18.safetensors");
    if resnet_w.is_file() {
        let vb_c = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&resnet_w), DType::F32, &cpu)?
        };
        let vb_d = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&resnet_w), DType::F32, device)?
        };
        let tc = vb_c.get((64,), "bn1.weight")?;
        let td = vb_d.get((64,), "bn1.weight")?;
        print_max_diff(&td, &tc, "diag_weight_load_resnet_bn1")?;
        let tc = vb_c.get((64, 3, 7, 7), "conv1.weight")?;
        let td = vb_d.get((64, 3, 7, 7), "conv1.weight")?;
        print_max_diff(&td, &tc, "diag_weight_load_resnet_conv1")?;
    } else {
        println!("diag_weight_load_resnet: resnet18.safetensors not found, skipped");
    }
    let clip_w = clip_dir().join("model.safetensors");
    if clip_w.is_file() {
        let vb_c = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&clip_w), DType::F32, &cpu)?
        };
        let vb_d = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&clip_w), DType::F32, device)?
        };
        let tc = vb_c.get((512, 512), "text_model.encoder.layers.0.self_attn.k_proj.weight")?;
        let td = vb_d.get((512, 512), "text_model.encoder.layers.0.self_attn.k_proj.weight")?;
        print_max_diff(&td, &tc, "diag_weight_load_clip_kproj")?;
        let tc = vb_c.get(
            (49408, 512),
            "text_model.embeddings.token_embedding.weight",
        )?;
        let td = vb_d.get(
            (49408, 512),
            "text_model.embeddings.token_embedding.weight",
        )?;
        print_max_diff(&td, &tc, "diag_weight_load_clip_token_embedding")?;
    } else {
        println!("diag_weight_load_clip: model.safetensors not found, skipped");
    }
    let sam_w = sam_dir().join("mobile_sam-tiny-vitt.safetensors");
    if sam_w.is_file() {
        let vb_c = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&sam_w), DType::F32, &cpu)?
        };
        let vb_d = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&sam_w), DType::F32, device)?
        };
        let tc = vb_c.get((32, 3, 3, 3), "image_encoder.patch_embed.seq.0.c.weight")?;
        let td = vb_d.get((32, 3, 3, 3), "image_encoder.patch_embed.seq.0.c.weight")?;
        print_max_diff(&td, &tc, "diag_weight_load_sam_patch_embed")?;
    } else {
        println!("diag_weight_load_sam: mobile_sam weights not found, skipped");
    }

    // 18. stride-2 conv (resnet stem + TinyViT patch embed / patch merging)
    let cfg_s2 = Conv2dConfig {
        padding: 1,
        stride: 2,
        ..Default::default()
    };
    let w = deterministic_f32_data(32 * 3 * 3 * 3, 0xD1B9);
    let conv_cpu = candle_nn::conv2d_no_bias(3, 32, 3, cfg_s2, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w.clone(), (32, 3, 3, 3), &cpu)?)]
            .into_iter()
            .collect(),
        DType::F32,
        &cpu,
    ))?;
    let conv_dev = candle_nn::conv2d_no_bias(3, 32, 3, cfg_s2, VarBuilder::from_tensors(
        [("weight".to_string(), Tensor::from_vec(w, (32, 3, 3, 3), device)?)]
            .into_iter()
            .collect(),
        DType::F32,
        device,
    ))?;
    let data = deterministic_f32_data(3 * 64 * 64, 0xD1BA);
    let inp_cpu = Tensor::from_vec(data.clone(), (1, 3, 64, 64), &cpu)?;
    let inp_dev = Tensor::from_vec(data, (1, 3, 64, 64), device)?;
    print_max_diff(
        &inp_dev.apply(&conv_dev)?,
        &inp_cpu.apply(&conv_cpu)?,
        "diag_conv2d_stride2",
    )?;

    // 19. f32::MIN causal-mask softmax (CLIP text attention)
    let data = deterministic_f32_data(12 * 77 * 77, 0xD1BB);
    let scores_cpu = Tensor::from_vec(data.clone(), (12, 77, 77), &cpu)?;
    let scores_dev = Tensor::from_vec(data, (12, 77, 77), device)?;
    let mask: Vec<f32> = (0..77)
        .flat_map(|i| (0..77).map(move |j| if j > i { f32::MIN } else { 0. }))
        .collect();
    let mask_cpu = Tensor::from_vec(mask.clone(), (1, 1, 77, 77), &cpu)?;
    let mask_dev = Tensor::from_vec(mask, (1, 1, 77, 77), device)?;
    let masked = |s: &Tensor, m: &Tensor| -> Result<Tensor> {
        candle_nn::ops::softmax(&s.broadcast_add(m)?.contiguous()?, 2)
    };
    print_max_diff(
        &masked(&scores_dev, &mask_dev)?,
        &masked(&scores_cpu, &mask_cpu)?,
        "diag_causal_mask_softmax",
    )?;

    // 20. narrow (CLIP position ids), cat (SAM prompt/decoder), pad (SAM preprocess)
    let data = deterministic_f32_data(77 * 512, 0xD1BC);
    let t_cpu = Tensor::from_vec(data.clone(), (77, 512), &cpu)?;
    let t_dev = Tensor::from_vec(data, (77, 512), device)?;
    print_max_diff(&t_dev.narrow(0, 5, 9)?, &t_cpu.narrow(0, 5, 9)?, "diag_narrow")?;
    let a_cpu = t_cpu.narrow(0, 0, 3)?;
    let a_dev = t_dev.narrow(0, 0, 3)?;
    let b_cpu = t_cpu.narrow(0, 3, 4)?;
    let b_dev = t_dev.narrow(0, 3, 4)?;
    print_max_diff(
        &Tensor::cat(&[&a_dev, &b_dev], 0)?,
        &Tensor::cat(&[&a_cpu, &b_cpu], 0)?,
        "diag_cat",
    )?;
    let data = deterministic_f32_data(2 * 8 * 60 * 60, 0xD1BD);
    let pad_cpu = Tensor::from_vec(data.clone(), (2, 8, 60, 60), &cpu)?;
    let pad_dev = Tensor::from_vec(data, (2, 8, 60, 60), device)?;
    print_max_diff(
        &pad_dev.pad_with_zeros(2, 1, 4)?,
        &pad_cpu.pad_with_zeros(2, 1, 4)?,
        "diag_pad_with_zeros",
    )?;
    let data = deterministic_f32_data(256 * 256, 0xD1BE);
    let up_cpu = Tensor::from_vec(data.clone(), (1, 1, 256, 256), &cpu)?;
    let up_dev = Tensor::from_vec(data, (1, 1, 256, 256), device)?;
    print_max_diff(
        &up_dev.upsample_nearest2d(1024, 1024)?,
        &up_cpu.upsample_nearest2d(1024, 1024)?,
        "diag_upsample_nearest2d",
    )?;

    // 21. the vulkan-only fused Linear path (candle_nn::ops::mul_mat_add,
    //     MUL_MAT_ADD bias-epilogue kernel) at model-realistic shapes.
    //     Every Linear-with-bias on vulkan routes through this gate (m > 8).
    for (m, k, n, tag) in [
        (77usize, 512usize, 512usize, "clip_text_attn"),
        (77, 512, 2048, "clip_text_mlp"),
        (50, 768, 768, "clip_vision_attn"),
        (50, 768, 3072, "clip_vision_mlp"),
        (65536, 64, 64, "tinyvit_stage0"),
        (4096, 256, 256, "sam_decoder_attn"),
        (1, 512, 1000, "resnet18_fc_m1"),
        (1, 2048, 1000, "resnet50_fc_m1"),
        (9, 512, 1000, "resnet_fc_m9"),
    ] {
        let data = deterministic_f32_data(m * k, 0xD1BF + m as u64);
        let x_cpu = Tensor::from_vec(data.clone(), (1, m, k), &cpu)?;
        let x_dev = Tensor::from_vec(data, (1, m, k), device)?;
        let data = deterministic_f32_data(k * n, 0xD1C0 + n as u64);
        let w_cpu = Tensor::from_vec(data.clone(), (k, n), &cpu)?;
        let w_dev = Tensor::from_vec(data, (k, n), device)?;
        let data = deterministic_f32_data(n, 0xD1C1 + k as u64);
        let b_cpu = Tensor::from_vec(data.clone(), (n,), &cpu)?;
        let b_dev = Tensor::from_vec(data, (n,), device)?;
        // reference: unfused on CPU
        let ref_cpu = x_cpu
            .broadcast_matmul(&w_cpu)?
            .broadcast_add(&b_cpu.reshape((1, 1, n))?)?;
        // fused vulkan path (same gate as candle_nn::Linear::forward)
        let fused_dev = candle_nn::ops::mul_mat_add(&x_dev, &w_dev, &b_dev)?;
        print_max_diff(
            &fused_dev,
            &ref_cpu,
            &format!("diag_mul_mat_add_{tag}"),
        )?;
        // fused vs unfused, both on device
        let unfused_dev = x_dev
            .broadcast_matmul(&w_dev)?
            .broadcast_add(&b_dev.reshape((1, 1, n))?)?;
        print_max_diff(
            &fused_dev,
            &unfused_dev,
            &format!("diag_mul_mat_add_vs_unfused_{tag}"),
        )?;
    }

    // 22. remaining resnet conv shapes: 1x1 and 7x7 stride 2
    for (out_c, in_c, ks, stride, pad, tag) in [
        (64usize, 64usize, 1usize, 1usize, 0usize, "conv1x1"),
        (128usize, 64usize, 1usize, 2usize, 0usize, "conv1x1s2"),
        (64usize, 3usize, 7usize, 2usize, 3usize, "conv7x7s2"),
    ] {
        let cfg = Conv2dConfig {
            padding: pad,
            stride,
            ..Default::default()
        };
        let w = deterministic_f32_data(out_c * in_c * ks * ks, 0xD1C2 + out_c as u64);
        let conv_cpu = candle_nn::conv2d_no_bias(in_c, out_c, ks, cfg, VarBuilder::from_tensors(
            [("weight".to_string(), Tensor::from_vec(w.clone(), (out_c, in_c, ks, ks), &cpu)?)]
                .into_iter()
                .collect(),
            DType::F32,
            &cpu,
        ))?;
        let conv_dev = candle_nn::conv2d_no_bias(in_c, out_c, ks, cfg, VarBuilder::from_tensors(
            [("weight".to_string(), Tensor::from_vec(w, (out_c, in_c, ks, ks), device)?)]
                .into_iter()
                .collect(),
            DType::F32,
            device,
        ))?;
        let side = if ks == 7 { 112 } else { 56 };
        let data = deterministic_f32_data(in_c * side * side, 0xD1C3 + ks as u64);
        let inp_cpu = Tensor::from_vec(data.clone(), (1, in_c, side, side), &cpu)?;
        let inp_dev = Tensor::from_vec(data, (1, in_c, side, side), device)?;
        print_max_diff(
            &inp_dev.apply(&conv_dev)?,
            &inp_cpu.apply(&conv_cpu)?,
            &format!("diag_{tag}"),
        )?;
    }

    // 23. batch_norm at resnet-realistic spatial sizes
    for (chans_b, side_b, tag) in [
        (64usize, 112usize, "bn_64x112"),
        (256usize, 56usize, "bn_256x56"),
        (512usize, 28usize, "bn_512x28"),
    ] {
        let bn_map = |dev: &Device, seed: u64| -> Result<_> {
            Ok([
                (
                    "weight".to_string(),
                    Tensor::from_vec(
                        deterministic_f32_data(chans_b, seed)
                            .iter()
                            .map(|v| v / 4.0 + 1.0)
                            .collect::<Vec<f32>>(),
                        (chans_b,),
                        dev,
                    )?,
                ),
                (
                    "bias".to_string(),
                    Tensor::from_vec(deterministic_f32_data(chans_b, seed + 1), (chans_b,), dev)?,
                ),
                (
                    "running_mean".to_string(),
                    Tensor::from_vec(deterministic_f32_data(chans_b, seed + 2), (chans_b,), dev)?,
                ),
                (
                    "running_var".to_string(),
                    Tensor::from_vec(
                        deterministic_f32_data(chans_b, seed + 3)
                            .iter()
                            .map(|v| v.abs() + 0.5)
                            .collect::<Vec<f32>>(),
                        (chans_b,),
                        dev,
                    )?,
                ),
            ]
            .into_iter()
            .collect())
        };
        let bn_cpu = candle_nn::batch_norm(
            chans_b,
            1e-5,
            VarBuilder::from_tensors(bn_map(&cpu, 0xD1C4)?, DType::F32, &cpu),
        )?;
        let bn_dev = candle_nn::batch_norm(
            chans_b,
            1e-5,
            VarBuilder::from_tensors(bn_map(device, 0xD1C4)?, DType::F32, device),
        )?;
        let data = deterministic_f32_data(chans_b * side_b * side_b, 0xD1C5 + chans_b as u64);
        let inp_cpu = Tensor::from_vec(data.clone(), (1, chans_b, side_b, side_b), &cpu)?;
        let inp_dev = Tensor::from_vec(data, (1, chans_b, side_b, side_b), device)?;
        print_max_diff(
            &bn_dev.forward_t(&inp_dev, false)?,
            &bn_cpu.forward_t(&inp_cpu, false)?,
            &format!("diag_{tag}"),
        )?;
    }

    // 24. ones-matrix probe on the failing shapes: with x=w=1, bias=0, every
    // output must be exactly K. Any element != K reveals how many k-terms the
    // kernel dropped or duplicated, and where (row/col pattern).
    for (m, k, n, tag) in [
        (77usize, 512usize, 512usize, "ones_77_512_512"),
        (65536usize, 64usize, 64usize, "ones_65536_64_64"),
        (50usize, 768usize, 768usize, "ones_50_768_768_control"),
        (77usize, 512usize, 77usize, "ones_77_512_77"),
        (77usize, 64usize, 512usize, "ones_77_64_512"),
    ] {
        let x_cpu = Tensor::ones((1, m, k), DType::F32, &cpu)?;
        let x_dev = Tensor::ones((1, m, k), DType::F32, device)?;
        let w_cpu = Tensor::ones((k, n), DType::F32, &cpu)?;
        let w_dev = Tensor::ones((k, n), DType::F32, device)?;
        let b_cpu = Tensor::zeros((n,), DType::F32, &cpu)?;
        let b_dev = Tensor::zeros((n,), DType::F32, device)?;
        let fused = candle_nn::ops::mul_mat_add(&x_dev, &w_dev, &b_dev)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = k as f32;
        let mut bad = Vec::new();
        for (idx, v) in fused.iter().enumerate() {
            if (*v - expected).abs() > 0.25 {
                let row = idx / n;
                let col = idx % n;
                bad.push((idx, row, col, *v - expected));
            }
        }
        println!(
            "diag_{tag}: expected {expected} everywhere; bad_elements={} (of {})",
            bad.len(),
            fused.len()
        );
        for &(idx, row, col, delta) in bad.iter().take(12) {
            println!("diag_{tag}:   idx={idx} row={row} col={col} delta={delta}");
        }
    }

    // 25. detail probe on the deterministic-data failing shapes: list every
    // bad element, then explain the first one (dropped / duplicated / swapped
    // k-term) by brute force over the k dimension.
    for (m, k, n, tag) in [
        (77usize, 512usize, 512usize, "detail_77_512_512"),
        (65536, 64, 64, "detail_65536_64_64"),
    ] {
        let data = deterministic_f32_data(m * k, 0xD1BF + m as u64);
        let x_cpu = Tensor::from_vec(data.clone(), (1, m, k), &cpu)?;
        let x_dev = Tensor::from_vec(data, (1, m, k), device)?;
        let data = deterministic_f32_data(k * n, 0xD1C0 + n as u64);
        let w_cpu = Tensor::from_vec(data.clone(), (k, n), &cpu)?;
        let w_dev = Tensor::from_vec(data, (k, n), device)?;
        let data = deterministic_f32_data(n, 0xD1C1 + k as u64);
        let b_cpu = Tensor::from_vec(data.clone(), (n,), &cpu)?;
        let b_dev = Tensor::from_vec(data, (n,), device)?;
        let fused = candle_nn::ops::mul_mat_add(&x_dev, &w_dev, &b_dev)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // unfused, no bias epilogue: is the plain dot exact?
        let unfused = x_dev
            .broadcast_matmul(&w_dev)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let x_v = x_cpu.flatten_all()?.to_vec1::<f32>()?;
        let w_v = w_cpu.flatten_all()?.to_vec1::<f32>()?;
        let b_v = b_cpu.to_vec1::<f32>()?;

        // upload integrity: round-trip every input, count bit mismatches
        let x_back = x_dev.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let w_back = w_dev.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b_back = b_dev.to_device(&cpu)?.to_vec1::<f32>()?;
        let x_bad = x_back.iter().zip(x_v.iter()).filter(|(a, b)| a != b).count();
        let w_bad = w_back.iter().zip(w_v.iter()).filter(|(a, b)| a != b).count();
        let b_bad = b_back.iter().zip(b_v.iter()).filter(|(a, b)| a != b).count();
        println!("diag_{tag}: upload_mismatch x={x_bad} w={w_bad} bias={b_bad}");

        // unfused (no bias epilogue) vs CPU: is the plain dot exact?
        {
            let mut unf_bad = 0usize;
            let mut unf_max = 0f32;
            for (idx, v) in unfused.iter().enumerate() {
                let row = idx / n;
                let col = idx % n;
                let mut exp = 0f32;
                for kk in 0..k {
                    exp += x_v[row * k + kk] * w_v[kk * n + col];
                }
                let d = (exp - v).abs();
                if d > unf_max {
                    unf_max = d;
                }
                if d > 1e-4 {
                    unf_bad += 1;
                }
            }
            println!("diag_{tag}: unfused_vs_cpu bad={unf_bad} max={unf_max:.3e}");
        }

        // identify the wrong bias values: wb[j] = gpu[0][j] - dot(0, j)
        {
            let mut wb = vec![0f32; n];
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += x_v[kk] * w_v[kk * n + j];
                }
                wb[j] = fused[j] - acc;
            }
            // quantize to the 1/64 grid the inputs live on
            let mut wq = vec![0f32; n];
            let mut quant_ok = true;
            for j in 0..n {
                wq[j] = (wb[j] * 64.0).round() / 64.0;
                if (wb[j] - wq[j]).abs() > 1e-3 {
                    quant_ok = false;
                }
            }
            println!(
                "diag_{tag}: wrong_bias[0..8]={:?} (grid_exact={quant_ok})",
                &wb[..8.min(n)]
            );
            println!("diag_{tag}: true_bias [0..8]={:?}", &b_v[..8.min(n)]);
            let diff_const = {
                let d0 = wq[0] - b_v[0];
                wq.iter().zip(b_v.iter()).all(|(&w, &b)| (w - b - d0).abs() < 1e-6)
            };
            println!("diag_{tag}: wb_minus_true_const={diff_const}");
            for (name, buf) in [("bias", &b_v), ("x", &x_v), ("w", &w_v)] {
                if buf.len() < 4 {
                    continue;
                }
                let mut hits = Vec::new();
                for o in 0..buf.len().saturating_sub(4) {
                    if buf[o] == wq[0]
                        && buf[o + 1] == wq[1]
                        && buf[o + 2] == wq[2]
                        && buf[o + 3] == wq[3]
                    {
                        hits.push(o);
                        if hits.len() >= 4 {
                            break;
                        }
                    }
                }
                if !hits.is_empty() {
                    println!(
                        "diag_{tag}: wrong_bias matches {name} buffer at offsets {:?}",
                        hits
                    );
                }
            }
        }

        let mut bad: Vec<(usize, usize, usize, f32, f32)> = Vec::new();
        for (idx, v) in fused.iter().enumerate() {
            let row = idx / n;
            let col = idx % n;
            let mut exp = b_v[col];
            for kk in 0..k {
                exp += x_v[row * k + kk] * w_v[kk * n + col];
            }
            if (exp - *v).abs() > 1e-4 {
                bad.push((idx, row, col, *v, exp));
            }
        }
        println!(
            "diag_{tag}: bad_elements={} (of {})",
            bad.len(),
            fused.len()
        );
        for &(idx, row, col, v, exp) in bad.iter().take(10) {
            println!(
                "diag_{tag}:   idx={idx} row={row} col={col} gpu={v:.6} cpu={exp:.6} delta={:.6}",
                v - exp
            );
        }
        if let Some(&(_i0, row0, col0, _v, _e)) = bad.first() {
            // per-column constancy: same column, other rows — if delta is
            // constant down a column, the corruption is in the bias/epilogue,
            // not the matmul terms.
            for probe_row in 1..4usize {
                if probe_row >= m {
                    break;
                }
                let mut exp = b_v[col0];
                for kk in 0..k {
                    exp += x_v[probe_row * k + kk] * w_v[kk * n + col0];
                }
                let got = fused[probe_row * n + col0];
                println!(
                    "diag_{tag}:   col={col0} row={probe_row} gpu={got:.6} cpu={exp:.6} delta={:.6}",
                    got - exp
                );
            }
            // row-mixing search: does gpu[row0][col0] equal
            // dot(x[r'], w[:,col0]) + bias[col0] for some other row r'?
            let &(idx0, row0, col0, v0, _e) = bad.first().unwrap();
            let _ = idx0;
            let mut found = Vec::new();
            for r2 in 0..m {
                let mut acc = b_v[col0];
                for kk in 0..k {
                    acc += x_v[r2 * k + kk] * w_v[kk * n + col0];
                }
                if (acc - v0).abs() < 1e-3 {
                    found.push(r2);
                }
            }
            println!(
                "diag_{tag}:   row-mix search: gpu[{row0}][{col0}] matches dot(x[r'],w[:,{col0}])+bias for r' in {:?}",
                found
            );
        }
        if let Some(&(_idx, row, col, v, exp)) = bad.first() {
            let delta = v - exp;
            for kk in 0..k {
                let t = x_v[row * k + kk] * w_v[kk * n + col];
                if (t - delta).abs() < 1e-3 {
                    println!("diag_{tag}:   explain: dropped term k={kk} (x*w={t})");
                }
                if (t + delta).abs() < 1e-3 {
                    println!("diag_{tag}:   explain: duplicated term k={kk} (x*w={t})");
                }
            }
            'outer: for k1 in 0..k {
                for k2 in (k1 + 1)..k {
                    let t1 = x_v[row * k + k1] * w_v[k1 * n + col];
                    let t2 = x_v[row * k + k2] * w_v[k2 * n + col];
                    if ((t1 - t2) - delta).abs() < 1e-3 {
                        println!("diag_{tag}:   explain: swapped k={k1}<->{k2}");
                        break 'outer;
                    }
                }
            }
        }
    }
    Ok(())
}
