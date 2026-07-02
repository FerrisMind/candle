//! candle-bench: llama-bench-style benchmark for candle quantized models.
//!
//! Measures prefill (prompt processing) and decode (token generation) throughput
//! across GGUF quantization types, matching llama-bench's measurement methodology.
//!
//! Usage:
//!   cargo run -p candle-examples --release --features vulkan --example candle-bench -- \
//!     --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json \
//!     --pp 512,1024,2048,4096 --tg 128,256
//!
//! Output: JSON lines with {test, prompt_tokens, gen_tokens, avg_us, stddev_us, tokens_per_s}.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights;

#[derive(Parser, Debug)]
#[command(name = "candle-bench", about = "llama-bench style benchmark for candle")]
struct Args {
    /// Path to GGUF model file.
    #[arg(long)]
    model: String,

    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: String,

    /// Comma-separated prompt sizes for prefill benchmark (e.g. "512,1024,2048,4096").
    #[arg(long, default_value = "512,1024,2048,4096")]
    pp: String,

    /// Comma-separated generation lengths for decode benchmark (e.g. "128,256").
    #[arg(long, default_value = "128,256")]
    tg: String,

    /// Number of warmup iterations before measurement.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Number of measurement iterations.
    #[arg(long, default_value_t = 5)]
    repeats: usize,

    /// Force CPU even if GPU is available.
    #[arg(long)]
    cpu: bool,
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

fn load_model(args: &Args) -> Result<(ModelWeights, Tokenizer, Device)> {
    let device = candle_examples::device(args.cpu)?;
    let mut file = std::fs::File::open(&args.model)
        .with_context(|| format!("opening model {}", args.model))?;
    let start = Instant::now();
    let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&args.model))?;
    let mut total_size = 0usize;
    for (_, tensor) in model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size += elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    eprintln!(
        "loaded {} tensors ({:.2} MB) in {:.2}s",
        model.tensor_infos.len(),
        total_size as f64 / 1e6,
        start.elapsed().as_secs_f64(),
    );
    let model = ModelWeights::from_gguf(model, &mut file, &device)?;
    let tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((model, tokenizer, device))
}

/// Generate synthetic prompt token IDs of exactly `n` tokens.
fn synthetic_tokens(tokenizer: &Tokenizer, n: usize) -> Result<Vec<u32>> {
    let mut text = String::new();
    while tokenizer.encode(text.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().len() < n {
        text.push_str("the ");
    }
    let mut ids = tokenizer.encode(text.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    ids.truncate(n);
    Ok(ids)
}

struct BenchResult {
    test: String,
    prompt_tokens: usize,
    gen_tokens: usize,
    avg_us: f64,
    stddev_us: f64,
    tokens_per_s: f64,
}

fn bench_pp(
    model: &mut ModelWeights,
    tokens: &[u32],
    device: &Device,
    warmup: usize,
    repeats: usize,
) -> Result<BenchResult> {
    let n = tokens.len();
    let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // Warmup
    for _ in 0..warmup {
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            logits_processor.sample(&logits.squeeze(0)?)?;
        }
        device.synchronize()?;
    }

    // Measure
    let mut durations = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            logits_processor.sample(&logits.squeeze(0)?)?;
        }
        device.synchronize()?;
        durations.push(start.elapsed());
    }

    let avg = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / durations.len() as f64;
    let variance = durations
        .iter()
        .map(|d| {
            let diff = d.as_secs_f64() - avg;
            diff * diff
        })
        .sum::<f64>()
        / durations.len() as f64;
    let stddev = variance.sqrt();
    let tps = n as f64 / avg;

    Ok(BenchResult {
        test: format!("pp{n}"),
        prompt_tokens: n,
        gen_tokens: 0,
        avg_us: avg * 1e6,
        stddev_us: stddev * 1e6,
        tokens_per_s: tps,
    })
}

fn bench_tg(
    model: &mut ModelWeights,
    n_gen: usize,
    device: &Device,
    warmup: usize,
    repeats: usize,
) -> Result<BenchResult> {
    let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let seed_token: u32 = 1;

    // Warmup
    for _ in 0..warmup {
        let mut pos = 0;
        let input = Tensor::new(&[seed_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let mut token = logits_processor.sample(&logits.squeeze(0)?)?;
        pos += 1;
        for _ in 1..n_gen {
            let input = Tensor::new(&[token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            token = logits_processor.sample(&logits.squeeze(0)?)?;
            pos += 1;
        }
        device.synchronize()?;
    }

    // Measure
    let mut durations = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let mut pos = 0;
        let input = Tensor::new(&[seed_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let mut token = logits_processor.sample(&logits.squeeze(0)?)?;
        pos += 1;
        for _ in 1..n_gen {
            let input = Tensor::new(&[token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            token = logits_processor.sample(&logits.squeeze(0)?)?;
            pos += 1;
        }
        device.synchronize()?;
        durations.push(start.elapsed());
    }

    let avg = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / durations.len() as f64;
    let variance = durations
        .iter()
        .map(|d| {
            let diff = d.as_secs_f64() - avg;
            diff * diff
        })
        .sum::<f64>()
        / durations.len() as f64;
    let stddev = variance.sqrt();
    let tps = n_gen as f64 / avg;

    Ok(BenchResult {
        test: format!("tg{n_gen}"),
        prompt_tokens: 0,
        gen_tokens: n_gen,
        avg_us: avg * 1e6,
        stddev_us: stddev * 1e6,
        tokens_per_s: tps,
    })
}

fn print_result(r: &BenchResult) {
    // Human-readable lines matching bench_examples.py regex patterns.
    if r.prompt_tokens > 0 {
        println!(
            "{:4} prompt tokens processed: {:.2} token/s",
            r.prompt_tokens, r.tokens_per_s,
        );
    }
    if r.gen_tokens > 0 {
        println!(
            "{:4} tokens generated: {:.2} token/s",
            r.gen_tokens, r.tokens_per_s,
        );
    }
    // Structured JSON for programmatic consumption.
    let json = serde_json::json!({
        "test": r.test,
        "prompt_tokens": r.prompt_tokens,
        "gen_tokens": r.gen_tokens,
        "avg_us": r.avg_us,
        "stddev_us": r.stddev_us,
        "tokens_per_s": r.tokens_per_s,
    });
    println!("{json}");
    std::io::stdout().flush().ok();
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pp_sizes = parse_usize_list(&args.pp);
    let tg_sizes = parse_usize_list(&args.tg);

    let (mut model, tokenizer, device) = load_model(&args)?;

    eprintln!("prefill sizes: {pp_sizes:?}");
    eprintln!("decode sizes:  {tg_sizes:?}");
    eprintln!("warmup={}, repeats={}", args.warmup, args.repeats);

    // Prefill benchmarks
    for &pp in &pp_sizes {
        let tokens = synthetic_tokens(&tokenizer, pp)?;
        let r = bench_pp(&mut model, &tokens, &device, args.warmup, args.repeats)?;
        eprintln!(
            "pp={pp:>5}  avg={:.2}ms  stddev={:.2}ms  {:.2} tok/s",
            r.avg_us / 1000.0,
            r.stddev_us / 1000.0,
            r.tokens_per_s,
        );
        print_result(&r);
    }

    // Decode benchmarks
    for &tg in &tg_sizes {
        let r = bench_tg(&mut model, tg, &device, args.warmup, args.repeats)?;
        eprintln!(
            "tg={tg:>5}  avg={:.2}ms  stddev={:.2}ms  {:.2} tok/s",
            r.avg_us / 1000.0,
            r.stddev_us / 1000.0,
            r.tokens_per_s,
        );
        print_result(&r);
    }

    Ok(())
}
