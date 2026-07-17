//! Real multimodal generation smoke for Qwen3-VL-2B (image in → text out).
//!
//! Unlike `qwen3_vl_vision_matrix` (vision tower logits only), this runs the full
//! `Qwen3VLModel::forward` path with image placeholders + pixel_values and decodes
//! greedy tokens so we can judge whether Vulkan/wgpu produce sensible answers.
//!
//! ```text
//! cargo test --release -p candle-transformers --features vulkan \
//!   --test real_gen_qwen3_vl -- --ignored --nocapture
//! ```
//!
//! Env: `CANDLE_QWEN3_VL_DIR` (default: unsloth Qwen3-VL-2B-Thinking).

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const IMAGE_TOKEN_ID: u32 = 151655;
const VISION_START_ID: u32 = 151652;
const VISION_END_ID: u32 = 151653;
const IM_END_ID: u32 = 151645;
const EOS_IDS: &[u32] = &[151645, 151643];

/// Cap rope / KV tables — upstream config has 262144 which OOMs the fixed KvCache.
const MAX_POS_CAP: usize = 4096;

fn vl_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CANDLE_QWEN3_VL_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking")
}

fn load_cfg(dir: &Path) -> Result<Config> {
    let s = std::fs::read_to_string(dir.join("config.json"))?;
    let mut cfg: Config =
        serde_json::from_str(&s).map_err(|e| candle::Error::msg(format!("VL config: {e}")))?;
    if cfg.text_config.max_position_embeddings > MAX_POS_CAP {
        println!(
            "capping max_position_embeddings {} -> {MAX_POS_CAP} (KV/rope memory)",
            cfg.text_config.max_position_embeddings
        );
        cfg.text_config.max_position_embeddings = MAX_POS_CAP;
    }
    Ok(cfg)
}

fn load_model(dir: &Path, device: &Device, dtype: DType) -> Result<(Qwen3VLModel, Tokenizer, Config)> {
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        candle::bail!("missing {weights:?}; set CANDLE_QWEN3_VL_DIR");
    }
    let cfg = load_cfg(dir)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| candle::Error::msg(format!("tokenizer: {e}")))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
    let model = Qwen3VLModel::new(&cfg, vb)?;
    Ok((model, tokenizer, cfg))
}

/// Build a solid-color image as Qwen3-VL patch tensor.
///
/// Layout matches PatchEmbed: each row is one spatial patch of
/// `in_chans * temporal_patch_size * patch * patch` (frame duplicated in time).
/// Values are channel-first and normalized with mean/std 0.5 (preprocessor).
fn solid_color_pixel_values(
    r: u8,
    g: u8,
    b: u8,
    grid_h: usize,
    grid_w: usize,
    patch: usize,
    temporal: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, usize)> {
    let grid_t = 1usize;
    let n_patches = grid_t * grid_h * grid_w;
    let pe = 3 * temporal * patch * patch;
    let mean = 0.5f32;
    let std = 0.5f32;
    let ch = [
        (r as f32 / 255.0 - mean) / std,
        (g as f32 / 255.0 - mean) / std,
        (b as f32 / 255.0 - mean) / std,
    ];
    let mut data = vec![0f32; n_patches * pe];
    for p in 0..n_patches {
        let base = p * pe;
        // Order: C, T, Ph, Pw  (as expected by reshape in PatchEmbed)
        for c in 0..3 {
            for t in 0..temporal {
                for _pix in 0..(patch * patch) {
                    let idx = base + c * (temporal * patch * patch) + t * (patch * patch) + _pix;
                    data[idx] = ch[c];
                }
            }
        }
    }
    let xs = Tensor::from_vec(data, (n_patches, pe), device)?.to_dtype(dtype)?;
    let grid = Tensor::from_vec(
        vec![grid_t as u32, grid_h as u32, grid_w as u32],
        (1, 3),
        device,
    )?;
    let merge = 2usize;
    let num_image_tokens = grid_t * (grid_h / merge) * (grid_w / merge);
    Ok((xs, grid, num_image_tokens))
}

fn encode_text(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| candle::Error::msg(format!("encode {text:?}: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

const IM_START_ID: u32 = 151644;
const THINK_START_ID: u32 = 151667;
const NL_ID: u32 = 198; // "\n" in Qwen BPE

/// Chat-style prompt with image placeholders expanded to `num_image_tokens` pads.
/// Uses special-token IDs (not string-encoded `<think>`) so the template matches training.
fn build_multimodal_ids(
    tok: &Tokenizer,
    num_image_tokens: usize,
    user_text: &str,
) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let mut ids = Vec::new();
    // <|im_start|>user\n
    ids.push(IM_START_ID);
    ids.extend(encode_text(tok, "user")?);
    ids.push(NL_ID);
    ids.push(VISION_START_ID);
    let pad_start = ids.len();
    ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(num_image_tokens));
    let pad_end = ids.len();
    ids.push(VISION_END_ID);
    ids.extend(encode_text(tok, user_text)?);
    ids.push(IM_END_ID);
    ids.push(NL_ID);
    // generation prompt (Thinking): <|im_start|>assistant\n<think>\n
    ids.push(IM_START_ID);
    ids.extend(encode_text(tok, "assistant")?);
    ids.push(NL_ID);
    ids.push(THINK_START_ID);
    ids.push(NL_ID);
    let continuous_img_pad = vec![(pad_start, pad_end)];
    Ok((ids, continuous_img_pad))
}

/// Prefill using the model's existing KV cache + per-step mask (seqlen==1),
/// without changing model code. Image pads are fed as one contiguous chunk so
/// vision embeds/DeepStack inject correctly; text is token-by-token.
fn prefill_multimodal(
    model: &Qwen3VLModel,
    device: &Device,
    pixel_values: &Tensor,
    image_grid_thw: &Tensor,
    prompt_ids: &[u32],
    img_pad_spans: &[(usize, usize)],
) -> Result<Tensor> {
    if img_pad_spans.len() != 1 {
        candle::bail!("expected a single image pad span, got {img_pad_spans:?}");
    }
    let (pad_start, pad_end) = img_pad_spans[0];
    if pad_end > prompt_ids.len() || pad_start >= pad_end {
        candle::bail!("bad img pad span {pad_start}..{pad_end} for len {}", prompt_ids.len());
    }

    if prompt_ids.is_empty() {
        candle::bail!("empty prompt");
    }
    let mut pos = 0usize;
    let mut last: Option<Tensor> = None;

    // 1) tokens before image pads — one-by-one (seqlen=1 → causal mask in model)
    while pos < pad_start {
        let input = Tensor::new(&[prompt_ids[pos]], device)?.unsqueeze(0)?;
        last = Some(model.forward(
            &input,
            None,
            None,
            None,
            None,
            vec![1],
            vec![vec![]],
            vec![vec![]],
            &[pos],
        )?);
        pos += 1;
    }

    // 2) all image pads in one chunk + pixel_values (DeepStack + embed inject)
    {
        let pads = &prompt_ids[pad_start..pad_end];
        let n = pads.len();
        let input = Tensor::new(pads, device)?.unsqueeze(0)?;
        last = Some(model.forward(
            &input,
            Some(pixel_values.clone()),
            None,
            Some(image_grid_thw.clone()),
            None,
            vec![n],
            vec![vec![(0, n)]], // spans relative to this chunk
            vec![vec![]],
            &[pos],
        )?);
        pos += n;
    }

    // 3) remaining prompt tokens one-by-one
    while pos < prompt_ids.len() {
        let input = Tensor::new(&[prompt_ids[pos]], device)?.unsqueeze(0)?;
        last = Some(model.forward(
            &input,
            None,
            None,
            None,
            None,
            vec![1],
            vec![vec![]],
            vec![vec![]],
            &[pos],
        )?);
        pos += 1;
    }

    last.ok_or_else(|| candle::Error::msg("prefill produced no logits"))
}

fn greedy_multimodal(
    model: &Qwen3VLModel,
    tokenizer: &Tokenizer,
    device: &Device,
    dtype: DType,
    pixel_values: &Tensor,
    image_grid_thw: &Tensor,
    prompt_ids: &[u32],
    img_pad_spans: &[(usize, usize)],
    max_new: usize,
) -> Result<(String, Vec<u32>)> {
    let mut tokens = prompt_ids.to_vec();
    let mut logits_processor = LogitsProcessor::new(42, None, None); // greedy

    let mut logits = prefill_multimodal(
        model,
        device,
        pixel_values,
        image_grid_thw,
        prompt_ids,
        img_pad_spans,
    )?;

    let mut generated = Vec::new();
    for step in 0..max_new {
        let logits_1d = logits.squeeze(0)?.to_dtype(DType::F32)?.flatten_all()?;
        if step == 0 {
            let v = logits_1d.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| {
                v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, v[i])).collect();
            let finite = v.iter().filter(|x| x.is_finite()).count();
            let maxv = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let minv = v.iter().cloned().fold(f32::INFINITY, f32::min);
            println!(
                "prefill logits: shape_elems={} finite={}/{} min={minv} max={maxv} top5={top5:?}",
                v.len(),
                finite,
                v.len()
            );
        }
        let next = logits_processor.sample(&logits_1d)?;
        generated.push(next);
        tokens.push(next);
        if EOS_IDS.contains(&next) || next == IM_END_ID {
            break;
        }
        let offset = tokens.len() - 1;
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        logits = model.forward(
            &input,
            None,
            None,
            None,
            None,
            vec![1],
            vec![vec![]],
            vec![vec![]],
            &[offset],
        )?;
        let _ = (step, dtype);
    }

    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
    Ok((text, generated))
}

fn run_on(device: Device, label: &str, dtype: DType) -> Result<()> {
    let dir = vl_dir();
    println!("=== real VL gen on {label}: {device:?} dtype={dtype:?} dir={dir:?} ===");
    let (model, tokenizer, cfg) = load_model(&dir, &device, dtype)?;

    // Small but valid image: 128×128 → grid 8×8 patches → 16 image tokens after merge=2
    let patch = cfg.vision_config.patch_size;
    let temporal = cfg.vision_config.temporal_patch_size;
    let merge = cfg.vision_config.spatial_merge_size;
    let grid_h = 8usize;
    let grid_w = 8usize;
    assert_eq!(grid_h % merge, 0);
    assert_eq!(grid_w % merge, 0);

    // Solid RED image — model should mention red / color if vision is wired.
    let (pixels, grid, n_img_tok) =
        solid_color_pixel_values(220, 20, 20, grid_h, grid_w, patch, temporal, &device, dtype)?;
    println!(
        "{label}: image grid=1x{grid_h}x{grid_w} patches={} image_tokens={n_img_tok} patch={patch} temporal={temporal}",
        grid_h * grid_w
    );

    let user_text =
        "What is the dominant color of this image? Answer with a single English color word.";
    let (prompt_ids, img_spans) = build_multimodal_ids(&tokenizer, n_img_tok, user_text)?;
    println!(
        "{label}: prompt_len={} img_pad_span={img_spans:?} first_20={:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(20)]
    );

    let max_new = 48usize;
    let start = std::time::Instant::now();
    let (text, gen_toks) = greedy_multimodal(
        &model,
        &tokenizer,
        &device,
        dtype,
        &pixels,
        &grid,
        &prompt_ids,
        &img_spans,
        max_new,
    )?;
    device.synchronize()?;
    let elapsed = start.elapsed();

    println!(
        "{label}: new_tokens={gen_toks:?}\n{label}: decoded={text:?}\n{label}: elapsed={elapsed:.2?}"
    );

    if gen_toks.is_empty() {
        candle::bail!("{label}: generated zero tokens");
    }
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    println!("{label}: alphabetic chars in decode = {letters}");
    if letters < 2 {
        candle::bail!("{label}: decoded text has almost no letters: {text:?}");
    }
    // Solid-red image → answer should mention red (Thinking model may reason first).
    let lower = text.to_lowercase();
    let color_hit = ["red", "crimson", "scarlet"]
        .iter()
        .any(|w| lower.contains(w));
    println!("{label}: color_keyword_hit={color_hit}");
    if !color_hit {
        candle::bail!(
            "{label}: expected a red-ish color word in image-conditioned answer, got {text:?}"
        );
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual real multimodal generation smoke"]
fn real_gen_qwen3_vl_vulkan() -> Result<()> {
    // F16: full 2B+vision fits; F32 often OOMs on discrete VRAM with KV.
    run_on(Device::new_vulkan(0)?, "vulkan", DType::F16)
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual real multimodal generation smoke"]
fn real_gen_qwen3_vl_wgpu() -> Result<()> {
    run_on(Device::new_wgpu(0)?, "wgpu", DType::F16)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual real multimodal generation smoke"]
fn real_gen_qwen3_vl_cuda() -> Result<()> {
    run_on(Device::new_cuda(0)?, "cuda", DType::F16)
}

#[test]
#[ignore = "manual real multimodal generation smoke — slow / heavy RAM"]
fn real_gen_qwen3_vl_cpu() -> Result<()> {
    run_on(Device::Cpu, "cpu", DType::F32)
}
