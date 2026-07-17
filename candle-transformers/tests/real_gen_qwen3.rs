//! Real greedy generation smoke for Qwen3-0.6B dense safetensors.
//!
//! Unlike `gpu_model_matrix` (logits vs CPU), this decodes actual text so we can
//! judge whether Vulkan/wgpu produce sensible tokens.
//!
//! ```text
//! CANDLE_QWEN3_DENSE_DIR=... cargo test --release -p candle-transformers --features vulkan \
//!   --test real_gen_qwen3 -- --ignored --nocapture
//! ```

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn model_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CANDLE_QWEN3_DENSE_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-0.6B")
}

fn load_model(device: &Device, dtype: DType) -> Result<(ModelForCausalLM, Tokenizer, Config)> {
    let dir = model_dir();
    let config_path = dir.join("config.json");
    let weights_path = dir.join("model.safetensors");
    let tok_path = dir.join("tokenizer.json");
    if !weights_path.is_file() {
        candle::bail!("missing model at {weights_path:?}; set CANDLE_QWEN3_DENSE_DIR");
    }
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .map_err(|e| candle::Error::msg(format!("config parse: {e}")))?;
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| candle::Error::msg(format!("tokenizer: {e}")))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };
    let model = ModelForCausalLM::new(&config, vb)?;
    Ok((model, tokenizer, config))
}

fn greedy_generate(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    max_new: usize,
    eos: u32,
) -> Result<(String, Vec<u32>)> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| candle::Error::msg(format!("encode: {e}")))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    if tokens.is_empty() {
        candle::bail!("empty prompt encoding");
    }
    let mut logits_processor = LogitsProcessor::new(42, None, None); // greedy when T=None

    // Prefill
    let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, 0)?;
    let mut generated = Vec::new();
    for step in 0..max_new {
        let logits_1d = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next = logits_processor.sample(&logits_1d)?;
        generated.push(next);
        tokens.push(next);
        if next == eos {
            break;
        }
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        logits = model.forward(&input, tokens.len() - 1)?;
        let _ = step;
    }
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
    Ok((text, generated))
}

fn run_on(device: Device, label: &str) -> Result<()> {
    // F32 for numerical stability / fairer CPU comparison
    let dtype = DType::F32;
    println!("=== real gen on {label}: {device:?} dtype={dtype:?} ===");
    let (mut model, tokenizer, cfg) = load_model(&device, dtype)?;
    let eos = 151645u32; // Qwen3 chat eos from config
    let prompt = "The capital of France is";
    let start = std::time::Instant::now();
    let (text, tokens) = greedy_generate(&mut model, &tokenizer, &device, prompt, 24, eos)?;
    device.synchronize()?;
    println!(
        "{label}: prompt={prompt:?}\n  new_tokens={tokens:?}\n  decoded={text:?}\n  elapsed={:.2?}",
        start.elapsed()
    );
    // Sanity: not empty, not all pad, should contain something letter-like after decode
    if tokens.is_empty() {
        candle::bail!("{label}: generated zero tokens");
    }
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    println!("{label}: alphabetic chars in decode = {letters}");
    let _ = cfg;
    Ok(())
}

#[test]
#[ignore = "manual real-generation smoke"]
fn real_gen_qwen3_cpu() -> Result<()> {
    run_on(Device::Cpu, "cpu")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual real-generation smoke"]
fn real_gen_qwen3_cuda() -> Result<()> {
    run_on(Device::new_cuda(0)?, "cuda")
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual real-generation smoke"]
fn real_gen_qwen3_vulkan() -> Result<()> {
    run_on(Device::new_vulkan(0)?, "vulkan")
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual real-generation smoke"]
fn real_gen_qwen3_wgpu() -> Result<()> {
    run_on(Device::new_wgpu(0)?, "wgpu")
}

/// Compare greedy token sequences CPU vs GPU for the same prompt.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "wgpu"))]
#[test]
#[ignore = "manual real-generation smoke"]
fn real_gen_qwen3_token_match() -> Result<()> {
    let prompt = "The capital of France is";
    let eos = 151645u32;
    let max_new = 16;

    let (mut cpu_model, tokenizer, _) = load_model(&Device::Cpu, DType::F32)?;
    let (cpu_text, cpu_toks) =
        greedy_generate(&mut cpu_model, &tokenizer, &Device::Cpu, prompt, max_new, eos)?;
    println!("CPU tokens={cpu_toks:?} text={cpu_text:?}");
    drop(cpu_model);

    let devices: Vec<(&str, Device)> = {
        let mut v = Vec::new();
        #[cfg(feature = "cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            v.push(("cuda", d));
        }
        #[cfg(feature = "vulkan")]
        if let Ok(d) = Device::new_vulkan(0) {
            v.push(("vulkan", d));
        }
        #[cfg(feature = "wgpu")]
        if let Ok(d) = Device::new_wgpu(0) {
            v.push(("wgpu", d));
        }
        v
    };

    for (label, device) in devices {
        let (mut model, tok, _) = load_model(&device, DType::F32)?;
        let (text, toks) = greedy_generate(&mut model, &tok, &device, prompt, max_new, eos)?;
        device.synchronize()?;
        println!("{label} tokens={toks:?} text={text:?}");
        if toks != cpu_toks {
            // Greedy F32 should match; report clearly if not
            candle::bail!(
                "{label} greedy tokens diverge from CPU.\n  cpu={cpu_toks:?}\n  gpu={toks:?}\n  cpu_text={cpu_text:?}\n  gpu_text={text:?}"
            );
        }
        println!("{label}: greedy token match OK");
    }
    Ok(())
}
