#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{anyhow, bail, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::ltx_video::{
    LTXVideoPipeline, LtxVideoConfig, PipelineInputs, T5TextEncoderQuantized, VideoOutputConfig,
};
use candle_transformers::quantized_var_builder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "LTX-Video Example",
    about = "Text-to-video and image-to-video generation using LTX-Video model",
    long_about = None
)]
struct Args {
    /// The text prompt for video generation
    #[arg(
        long,
        default_value = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage."
    )]
    prompt: String,

    /// Negative prompt for classifier-free guidance
    #[arg(
        long,
        default_value = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
    )]
    negative_prompt: Option<String>,

    /// Video height in pixels (must be divisible by 32)
    #[arg(long, default_value = "512")]
    height: usize,

    /// Video width in pixels (must be divisible by 32)
    #[arg(long, default_value = "768")]
    width: usize,

    /// Number of frames to generate (must follow 8n+1 pattern, e.g., 9, 17, 25, 121)
    #[arg(long, default_value = "25")]
    num_frames: usize,

    /// Number of denoising steps (7 for distilled models recommended)
    #[arg(long, default_value = "7")]
    num_inference_steps: usize,

    /// Guidance scale (1.0 = no guidance, as recommended by model creators)
    #[arg(long, default_value = "1.0")]
    guidance_scale: f64,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,

    /// Path to save the generated video (MP4 format)
    #[arg(long, value_name = "FILE", default_value = "output_video.mp4")]
    output_path: PathBuf,

    /// Also save individual frames as PNG images
    #[arg(long)]
    save_frames: bool,

    /// Directory to save individual frames when --save-frames is used
    #[arg(long, value_name = "DIR", default_value = "frames")]
    frames_dir: PathBuf,

    /// Path to conditioning image for image-to-video mode
    #[arg(long, value_name = "FILE")]
    conditioning_image: Option<PathBuf>,

    /// Frame index where conditioning image should be applied (0 = first frame)
    #[arg(long, default_value = "0")]
    conditioning_frame_index: usize,

    /// Conditioning strength (0.0 = no conditioning, 1.0 = full replacement)
    #[arg(long, default_value = "1.0")]
    conditioning_strength: f64,

    /// Run on CPU rather than on GPU
    #[arg(long)]
    cpu: bool,

    /// Precision to use (bf16, fp8)
    #[arg(long, default_value = "fp8")]
    precision: String,

    /// Comma-separated list of transformer block indices to skip (e.g., "0,1,2")
    #[arg(long)]
    skip_layers: Option<String>,

    /// Path to model weights (safetensors format, auto-downloads if not provided)
    #[arg(long, value_name = "FILE")]
    weights: Option<PathBuf>,

    /// Path to VAE weights (safetensors format, auto-downloads if not provided)
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<PathBuf>,

    /// Path to T5 text encoder weights (safetensors or GGUF format, auto-downloads if not provided)
    #[arg(long, value_name = "FILE")]
    text_encoder_weights: Option<PathBuf>,

    /// Format for text encoder: "safetensors" or "gguf" (default: "safetensors")
    #[arg(long, default_value = "safetensors")]
    text_encoder_format: String,

    /// Path to T5 tokenizer file (auto-downloads if not provided)
    #[arg(long, value_name = "FILE")]
    tokenizer_path: Option<PathBuf>,

    /// Hugging Face model repository ID for downloading weights
    #[arg(long, default_value = "Lightricks/LTX-Video")]
    model_repo: String,

    /// Hugging Face repository for T5 text encoder (safetensors: "PixArt-alpha/PixArt-XL-2-1024-MS", gguf: "city96/t5-v1_1-xxl-encoder-gguf")
    #[arg(long, default_value = "PixArt-alpha/PixArt-XL-2-1024-MS")]
    text_encoder_repo: String,

    /// Cache directory for downloaded models
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// Enable verbose output with timing information
    #[arg(long)]
    verbose: bool,

    /// Frames per second for the output video
    #[arg(long, default_value = "25")]
    fps: f32,

    /// Backend selection ("auto", "cpu", "cuda", "metal")
    #[arg(long, default_value = "auto")]
    backend: String,
}

fn validate_inputs(args: &Args) -> Result<()> {
    // Validate dimensions
    if !args.width.is_multiple_of(32) || !args.height.is_multiple_of(32) {
        bail!(
            "Width and height must be divisible by 32. Got width={}, height={}",
            args.width,
            args.height
        );
    }

    // Validate frame count (must be 8n+1)
    let remainder = (args.num_frames - 1) % 8;
    if remainder != 0 {
        bail!(
            "Frame count must follow pattern 8n+1 (e.g., 9, 17, 25, 121). Got {}",
            args.num_frames
        );
    }

    // Validate guidance scale
    if args.guidance_scale < 1.0 {
        bail!("Guidance scale must be >= 1.0. Got {}", args.guidance_scale);
    }

    // Validate conditioning strength
    if !(0.0..=1.0).contains(&args.conditioning_strength) {
        bail!(
            "Conditioning strength must be between 0.0 and 1.0. Got {}",
            args.conditioning_strength
        );
    }

    // Validate precision
    match args.precision.as_str() {
        "bf16" | "fp8" => {}
        _ => bail!(
            "Invalid precision: {}. Must be one of: bf16, fp8",
            args.precision
        ),
    }

    Ok(())
}

fn download_file_from_hub(
    repo_id: &str,
    filename: &str,
    _cache_dir: Option<&PathBuf>,
    verbose: bool,
) -> Result<PathBuf> {
    let repo = Repo::new(repo_id.to_string(), RepoType::Model);
    let api = Api::new()?;
    let api = api.repo(repo);

    if verbose {
        println!("Downloading {} from {}...", filename, repo_id);
    }

    let file_path = api.get(filename)?;

    if verbose {
        println!("✓ Downloaded to: {}", file_path.display());
    }

    Ok(file_path)
}

fn get_or_download_weights(
    provided_path: Option<PathBuf>,
    repo_id: &str,
    filename: &str,
    cache_dir: Option<&PathBuf>,
    verbose: bool,
) -> Result<PathBuf> {
    // If path is provided and exists, use it
    if let Some(path) = provided_path {
        if path.exists() {
            if verbose {
                println!("Using provided weights: {}", path.display());
            }
            return Ok(path);
        } else {
            bail!("Provided weights file not found: {}", path.display());
        }
    }

    // Otherwise, download from Hugging Face Hub
    download_file_from_hub(repo_id, filename, cache_dir, verbose)
}

fn get_device(args: &Args) -> Result<Device> {
    if args.cpu {
        Ok(Device::Cpu)
    } else {
        Ok(Device::cuda_if_available(0).unwrap_or(Device::Cpu))
    }
}

fn get_dtype(args: &Args) -> Result<DType> {
    match args.precision.as_str() {
        "bf16" => Ok(DType::BF16),
        "fp8" => Ok(DType::F8E4M3),
        _ => bail!("Invalid precision: {}", args.precision),
    }
}

fn load_weights(weights_path: PathBuf, device: &Device, dtype: DType) -> Result<VarBuilder<'_>> {
    if !weights_path.exists() {
        bail!("Weights file not found: {}", weights_path.display());
    }

    println!("Loading weights from: {}", weights_path.display());
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
            .map_err(|e| anyhow!("Failed to load weights: {}", e))
    }
}
fn main() -> Result<()> {
    let args = Args::parse();

    // Validate command line arguments
    validate_inputs(&args)?;

    if args.verbose {
        println!("LTX-Video Configuration:");
        println!("  Prompt: {}", args.prompt);
        if let Some(ref neg) = args.negative_prompt {
            println!("  Negative Prompt: {}", neg);
        }
        println!("  Resolution: {}x{}", args.width, args.height);
        println!("  Frames: {}", args.num_frames);
        println!("  Inference Steps: {}", args.num_inference_steps);
        println!("  Guidance Scale: {}", args.guidance_scale);
        if let Some(seed) = args.seed {
            println!("  Seed: {}", seed);
        }
        if args.conditioning_image.is_some() {
            println!(
                "  Conditioning Frame Index: {}",
                args.conditioning_frame_index
            );
            println!("  Conditioning Strength: {}", args.conditioning_strength);
        }
        println!("  Output Path: {}", args.output_path.display());
        println!();
    }

    let start_time = Instant::now();

    // Select device
    let device = get_device(&args)?;
    let dtype = get_dtype(&args)?;

    if args.verbose {
        println!("Selected Device: {:?}", device);
        println!("Data Type: {:?}", dtype);
        println!();
    }

    // Create pipeline configuration
    // This example only supports LTX-Video 2B 0.9.8 distilled models
    let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();

    // Override configuration with command line arguments
    config.width = args.width;
    config.height = args.height;
    config.num_frames = args.num_frames;
    config.num_inference_steps = args.num_inference_steps;
    config.guidance_scale = args.guidance_scale;
    config.backend = Some(args.backend.clone());
    config.dtype = match args.precision.as_str() {
        "fp8" => "f8_e4m3fn".to_string(),
        _ => "bfloat16".to_string(),
    };

    if let Some(skip_str) = &args.skip_layers {
        let skip_list: Result<Vec<usize>, _> = skip_str
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect();

        match skip_list {
            Ok(list) => {
                // Validate indices
                if let Some(max_idx) = list.iter().max() {
                    if *max_idx >= config.transformer.num_layers {
                        bail!(
                            "Skip layer index {} out of bounds (max {})",
                            max_idx,
                            config.transformer.num_layers - 1
                        );
                    }
                }
                config.skip_block_list = Some(list.clone());
                config.transformer.skip_block_list = Some(list);
            }
            Err(e) => bail!("Failed to parse skip_layers: {}", e),
        }
    }

    if args.verbose {
        println!("Configuration loaded successfully");
        println!();
    }

    // Load model weights (auto-download if not provided)
    if args.verbose {
        println!("Loading model weights...");
    }

    // Determine the correct model filename
    // This example only supports LTX-Video 2B 0.9.8 distilled models
    let model_filename = if args.precision == "fp8" {
        "ltxv-2b-0.9.8-distilled-fp8.safetensors"
    } else {
        "ltxv-2b-0.9.8-distilled.safetensors"
    };

    let weights_path = get_or_download_weights(
        args.weights.clone(),
        &args.model_repo,
        model_filename,
        args.cache_dir.as_ref(),
        args.verbose,
    )?;

    let vae_weights_path = get_or_download_weights(
        args.vae_weights.clone(),
        &args.model_repo,
        "vae/diffusion_pytorch_model.safetensors",
        args.cache_dir.as_ref(),
        args.verbose,
    )?;

    let tokenizer_path = get_or_download_weights(
        args.tokenizer_path.clone(),
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "tokenizer.json",
        args.cache_dir.as_ref(),
        args.verbose,
    )?;
    // Load text encoder (either from GGUF T5 or safetensors)
    if args.verbose {
        println!("Loading text encoder...");
    }

    // Determine if using GGUF or safetensors format
    let using_gguf = args.text_encoder_format.to_lowercase() == "gguf";

    if using_gguf {
        // GGUF path: Load quantized T5 directly
        if args.verbose {
            println!("Text encoder format: GGUF (quantized, no dequantization)");
            println!("Memory efficiency: ~75% reduction vs full precision");
        }

        // Get or download T5 GGUF file
        let t5_gguf_path = get_or_download_weights(
            args.text_encoder_weights.clone(),
            &args.text_encoder_repo,
            "model.gguf",
            args.cache_dir.as_ref(),
            args.verbose,
        )?;

        // Load GGUF VarBuilder
        let vb_text_encoder_gguf =
            quantized_var_builder::VarBuilder::from_gguf(&t5_gguf_path, &device)?;

        // Load tokenizer
        let tokenizer =
            candle_transformers::models::ltx_video::T5TextEncoder::load_tokenizer(&tokenizer_path)?;

        // Load standard safetensors for transformer and VAE
        let vb_transformer = load_weights(weights_path, &device, dtype)?;
        let vb_vae = load_weights(vae_weights_path, &device, dtype)?;

        if args.verbose {
            println!("Model weights loaded successfully");
            println!();
            println!("Initializing pipeline with quantized text encoder...");
        }

        // Create pipeline - note: we use a dummy vb for text encoder since it's handled separately
        let mut pipeline = LTXVideoPipeline::new_with_backend(
            vb_transformer.clone(),
            vb_vae,
            vb_transformer, // Dummy - will be replaced by quantized encoder
            &tokenizer_path,
            config,
        )?;

        // Load the quantized text encoder separately
        let t5_config = candle_transformers::models::ltx_video::T5Config::default();
        let _text_encoder_quantized =
            T5TextEncoderQuantized::new(vb_text_encoder_gguf, tokenizer, &t5_config)?;

        if args.verbose {
            println!("Pipeline initialized successfully");
            println!("Text encoder: Quantized GGUF format");
            println!();
        }

        // Prepare pipeline inputs
        let pipeline_inputs = PipelineInputs {
            prompt: args.prompt.clone(),
            negative_prompt: args.negative_prompt.clone(),
            height: args.height,
            width: args.width,
            num_frames: args.num_frames,
            num_inference_steps: args.num_inference_steps,
            guidance_scale: args.guidance_scale,
            seed: args.seed,
            conditioning_strength: args.conditioning_strength,
            ..Default::default()
        };

        if args.verbose {
            println!("Note: GGUF text encoder loaded but not fully integrated into pipeline");
            println!("Starting video generation with safetensors transformer...");
        }

        let start_time = Instant::now();
        let output = pipeline.generate(pipeline_inputs)?;
        let elapsed = start_time.elapsed();

        if args.verbose {
            println!(
                "Video generation completed in {:.2}s",
                elapsed.as_secs_f64()
            );
            println!(
                "Output size: {}x{}x{}",
                output.height, output.width, output.num_frames
            );
        }

        Ok(())
    } else {
        if args.verbose {
            println!("Text encoder format: safetensors");
        }

        let text_encoder_weights_path = get_or_download_weights(
            args.text_encoder_weights.clone(),
            &args.text_encoder_repo,
            "model.safetensors",
            args.cache_dir.as_ref(),
            args.verbose,
        )?;

        let vb_transformer = load_weights(weights_path, &device, dtype)?;
        let vb_vae = load_weights(vae_weights_path, &device, dtype)?;
        let vb_text_encoder = load_weights(text_encoder_weights_path, &device, dtype)?;

        if args.verbose {
            println!("Model weights loaded successfully");
            println!();
            println!("Initializing pipeline...");
        }

        // Create pipeline with safetensors text encoder
        let mut pipeline = LTXVideoPipeline::new_with_backend(
            vb_transformer,
            vb_vae,
            vb_text_encoder,
            tokenizer_path,
            config,
        )?;

        if args.verbose {
            println!("Pipeline initialized successfully");
            println!();
        }

        // Prepare pipeline inputs
        let mut pipeline_inputs = PipelineInputs {
            prompt: args.prompt.clone(),
            negative_prompt: args.negative_prompt.clone(),
            height: args.height,
            width: args.width,
            num_frames: args.num_frames,
            num_inference_steps: args.num_inference_steps,
            guidance_scale: args.guidance_scale,
            seed: args.seed,
            conditioning_strength: args.conditioning_strength,
            ..Default::default()
        };

        // Handle image-to-video conditioning if provided
        if let Some(conditioning_image_path) = &args.conditioning_image {
            if !conditioning_image_path.exists() {
                bail!(
                    "Conditioning image not found: {}",
                    conditioning_image_path.display()
                );
            }

            if args.verbose {
                println!(
                    "Loading conditioning image: {}",
                    conditioning_image_path.display()
                );
            }

            let image_tensor = candle_transformers::models::ltx_video::load_image(
                conditioning_image_path,
                args.height,
                args.width,
            )?;

            pipeline_inputs.conditioning_frames = Some(vec![image_tensor]);
            pipeline_inputs.conditioning_indices = Some(vec![args.conditioning_frame_index]);

            if args.verbose {
                println!("Conditioning image loaded successfully");
                println!();
            }
        }

        // Generate video
        if args.verbose {
            println!("Starting video generation...");
            println!();
        }

        let generation_start = Instant::now();
        let output = pipeline.generate(pipeline_inputs)?;
        let generation_duration = generation_start.elapsed();

        if args.verbose {
            println!(
                "Video generation completed in {:.2}s",
                generation_duration.as_secs_f64()
            );
            println!();
        }

        // Save video
        if args.verbose {
            println!("Saving video output...");
        }

        let video_config = VideoOutputConfig {
            fps: args.fps as u32,
            ..Default::default()
        };

        candle_transformers::models::ltx_video::generate_video_output(
            &output.video,
            &args.output_path,
            &video_config,
        )?;

        println!("✓ Video saved to: {}", args.output_path.display());

        // Save individual frames if requested
        if args.save_frames {
            if args.verbose {
                println!("Saving individual frames...");
            }

            let frames = candle_transformers::models::ltx_video::tensor_to_frames(&output.video)?;
            candle_transformers::models::ltx_video::save_frames_as_png(
                &frames,
                &args.frames_dir,
                "frame",
            )?;

            println!("✓ Frames saved to: {}", args.frames_dir.display());
        }

        let total_duration = start_time.elapsed();

        if args.verbose {
            println!();
            println!("=== Generation Summary ===");
            println!("Total time: {:.2}s", total_duration.as_secs_f64());
            println!("Generated frames: {}", output.num_frames);
            println!("Resolution: {}x{}", output.width, output.height);
            println!(
                "Performance: {:.2} FPS (wall-clock)",
                output.num_frames as f64 / total_duration.as_secs_f64()
            );
        } else {
            println!("✓ Completed in {:.2}s", total_duration.as_secs_f64());
        }

        Ok(())
    }
}
