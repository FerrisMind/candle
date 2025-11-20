//! LTX-Video Inference Pipeline
//!
//! This module implements the main inference pipeline that orchestrates all components
//! for end-to-end video generation. It supports both text-to-video and image-to-video modes.
//! Additionally, it supports loading text encoder from quantized GGUF format for memory efficiency.

use candle::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use super::{
    error::LtxVideoError, generate_video_output, inject_conditioning, prepare_conditioning_latents,
    CausalVideoAutoencoder, LtxVideoConfig, RectifiedFlowScheduler, T5TextEncoder,
    T5TextEncoderQuantized, TextConditioning, Transformer3D, VideoGenerationResult,
    VideoOutputConfig,
};
use crate::quantized_var_builder;

/// Main LTX-Video inference pipeline
pub struct LTXVideoPipeline {
    transformer: Transformer3D,
    vae: CausalVideoAutoencoder,
    text_encoder: T5TextEncoder,
    scheduler: RectifiedFlowScheduler,
    config: LtxVideoConfig,
    device: Device,
}

/// Input parameters for video generation
#[derive(Clone, Debug)]
pub struct PipelineInputs {
    /// Text prompt for video generation
    pub prompt: String,
    /// Optional negative prompt for classifier-free guidance
    pub negative_prompt: Option<String>,
    /// Optional conditioning frames for image-to-video mode
    pub conditioning_frames: Option<Vec<Tensor>>,
    /// Frame indices where conditioning should be applied
    pub conditioning_indices: Option<Vec<usize>>,
    /// Conditioning strength (0.0 = no conditioning, 1.0 = full replacement)
    pub conditioning_strength: f64,
    /// Video height in pixels (must be divisible by 32)
    pub height: usize,
    /// Video width in pixels (must be divisible by 32)
    pub width: usize,
    /// Number of frames to generate (must follow 8n+1 pattern)
    pub num_frames: usize,
    /// Number of denoising steps
    pub num_inference_steps: usize,
    /// Guidance scale for classifier-free guidance (1.0 = no guidance)
    pub guidance_scale: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Optional custom timestep schedule
    pub custom_timesteps: Option<Vec<f64>>,
}

impl Default for PipelineInputs {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            conditioning_frames: None,
            conditioning_indices: None,
            conditioning_strength: 1.0,
            height: 512,
            width: 768,
            num_frames: 121,
            num_inference_steps: 50,
            guidance_scale: 1.0,
            seed: None,
            custom_timesteps: None,
        }
    }
}

/// Output from the pipeline
pub struct PipelineOutput {
    /// Generated video tensor in (B, C, F, H, W) format
    /// Values are in range [0, 255] as F32
    pub video: Tensor,
    /// Number of frames generated
    pub num_frames: usize,
    /// Video height
    pub height: usize,
    /// Video width
    pub width: usize,
}

impl LTXVideoPipeline {
    /// Create a new pipeline instance
    ///
    /// # Arguments
    /// * `vb_transformer` - VarBuilder for transformer weights
    /// * `vb_vae` - VarBuilder for VAE weights
    /// * `vb_text_encoder` - VarBuilder for text encoder weights
    /// * `tokenizer_path` - Path to T5 tokenizer
    /// * `config` - Pipeline configuration
    /// * `device` - Device to run on (CPU, CUDA, Metal)
    pub fn new(
        vb_transformer: VarBuilder,
        vb_vae: VarBuilder,
        vb_text_encoder: VarBuilder,
        tokenizer_path: impl AsRef<Path>,
        config: LtxVideoConfig,
        device: Device,
    ) -> Result<Self> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        // Initialize transformer
        let transformer = Transformer3D::new(vb_transformer.clone(), config.transformer.clone())?;

        // Initialize VAE
        let vae = CausalVideoAutoencoder::new(vb_vae.clone(), config.vae.clone())?;

        // Load tokenizer
        let tokenizer = T5TextEncoder::load_tokenizer(tokenizer_path)?;

        // Initialize text encoder
        let text_encoder =
            T5TextEncoder::new(vb_text_encoder.clone(), tokenizer, &config.text_encoder)?;

        // Initialize scheduler
        let mut scheduler = RectifiedFlowScheduler::new(config.scheduler.clone());
        scheduler.set_timesteps(config.num_inference_steps, None)?;

        Ok(Self {
            transformer,
            vae,
            text_encoder,
            scheduler,
            config,
            device,
        })
    }

    /// Create a new pipeline instance with automatic device selection from config
    ///
    /// # Arguments
    /// * `vb_transformer` - VarBuilder for transformer weights
    /// * `vb_vae` - VarBuilder for VAE weights
    /// * `vb_text_encoder` - VarBuilder for text encoder weights
    /// * `tokenizer_path` - Path to T5 tokenizer
    /// * `config` - Pipeline configuration (backend field determines device)
    pub fn new_with_backend(
        vb_transformer: VarBuilder,
        vb_vae: VarBuilder,
        vb_text_encoder: VarBuilder,
        tokenizer_path: impl AsRef<Path>,
        config: LtxVideoConfig,
    ) -> Result<Self> {
        // Create device manager from config
        let device_manager = config
            .create_device_manager()
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        // Print device information
        device_manager.print_info();

        // Estimate memory requirements
        let memory_estimator = config.estimate_memory()?;
        memory_estimator.print_estimation();

        // Check memory requirements
        device_manager.check_memory_requirements(memory_estimator.total_memory_gb())?;

        // Create pipeline with the selected device
        Self::new(
            vb_transformer,
            vb_vae,
            vb_text_encoder,
            tokenizer_path,
            config,
            device_manager.device().clone(),
        )
    }

    /// Load text encoder from GGUF format (quantized, no dequantization)
    ///
    /// This is a helper method that loads a quantized T5 text encoder from GGUF format.
    /// Unlike loading from safetensors, weights remain quantized throughout inference,
    /// providing significant memory savings (approximately 75% reduction).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gguf_vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
    ///     "t5-xxl.gguf", &device
    /// )?;
    /// let mut text_encoder = T5TextEncoderQuantized::new(gguf_vb, tokenizer, &config)?;
    ///
    /// let embeddings = text_encoder.encode(&["Hello world"], None, None)?;
    /// ```
    pub fn load_text_encoder_quantized_gguf(
        gguf_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: &super::T5Config,
        device: &Device,
    ) -> Result<T5TextEncoderQuantized> {
        let gguf_vb = quantized_var_builder::VarBuilder::from_gguf(gguf_path, device)?;
        let tokenizer = T5TextEncoder::load_tokenizer(tokenizer_path)?;
        T5TextEncoderQuantized::new(gguf_vb, tokenizer, config)
    }

    /// Generate video from text prompt or image conditioning
    ///
    /// # Arguments
    /// * `inputs` - Pipeline input parameters
    ///
    /// # Returns
    /// Generated video tensor in (B, C, F, H, W) format with values in [0, 255] range
    pub fn generate(&mut self, inputs: PipelineInputs) -> Result<PipelineOutput> {
        // Validate inputs
        self.validate_inputs(&inputs)
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        println!("Starting LTX-Video generation...");
        println!("  Resolution: {}x{}", inputs.width, inputs.height);
        println!("  Frames: {}", inputs.num_frames);
        println!("  Steps: {}", inputs.num_inference_steps);
        println!("  Guidance Scale: {}", inputs.guidance_scale);

        // Step 1: Encode text prompt
        println!("Encoding text prompt...");
        let prompt_embeds = self.encode_prompt(&inputs.prompt, &inputs.negative_prompt)?;

        // Step 2: Prepare initial latents
        println!("Preparing latents...");
        let mut latents = self.prepare_latents(
            1, // batch_size
            inputs.num_frames,
            inputs.height,
            inputs.width,
            inputs.seed,
        )?;

        // Step 3: Apply conditioning if provided
        if let (Some(frames), Some(indices)) =
            (&inputs.conditioning_frames, &inputs.conditioning_indices)
        {
            println!(
                "Applying conditioning at {} frame positions...",
                indices.len()
            );
            latents =
                self.apply_conditioning(&latents, frames, indices, inputs.conditioning_strength)?;
        }

        // Step 4: Update scheduler timesteps
        let mut scheduler = self.scheduler.clone();
        scheduler.set_timesteps(inputs.num_inference_steps, inputs.custom_timesteps.clone())?;

        // Step 5: Denoising loop
        println!("Starting denoising loop...");
        latents = self.denoise_loop(latents, &prompt_embeds, inputs.guidance_scale, &scheduler)?;

        // Step 6: Decode latents to video
        println!("Decoding latents to video...");
        let video = self.vae.decode(&latents, None)?;

        // Step 7: Post-process video
        let video = self.postprocess_video(&video)?;

        println!("Generation complete!");

        Ok(PipelineOutput {
            video,
            num_frames: inputs.num_frames,
            height: inputs.height,
            width: inputs.width,
        })
    }

    /// Encode text prompt to embeddings
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative_prompt: &Option<String>,
    ) -> Result<TextConditioning> {
        let prompts = vec![prompt];
        let negative_prompts = negative_prompt.as_ref().map(|p| vec![p.as_str()]);

        self.text_encoder.encode(
            &prompts,
            negative_prompts.as_deref(),
            Some(self.config.text_encoder.max_length),
        )
    }

    /// Initialize random latents with optional seed
    fn prepare_latents(
        &self,
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        _seed: Option<u64>,
    ) -> Result<Tensor> {
        let latent_channels = self.config.vae.latent_channels;
        let temporal_compression = self.config.vae.temporal_compression;
        let spatial_compression = self.config.vae.spatial_compression;

        let latent_frames = num_frames / temporal_compression;
        let latent_height = height / spatial_compression;
        let latent_width = width / spatial_compression;

        let shape = (
            batch_size,
            latent_channels,
            latent_frames,
            latent_height,
            latent_width,
        );

        // Generate random noise
        // Note: Candle currently doesn't support seeded random generation
        // For reproducibility, consider setting a global seed before calling this
        let latents = Tensor::randn(0.0f32, 1.0, shape, &self.device)?;

        Ok(latents)
    }

    /// Apply conditioning frames to latents
    fn apply_conditioning(
        &self,
        latents: &Tensor,
        conditioning_frames: &[Tensor],
        conditioning_indices: &[usize],
        strength: f64,
    ) -> Result<Tensor> {
        // Prepare conditioning latents from images
        let conditioning_items = prepare_conditioning_latents(
            &self.vae,
            conditioning_frames,
            conditioning_indices,
            &self.device,
            self.config
                .get_dtype()
                .map_err(|e| candle::Error::Msg(e.to_string()))?,
        )?;

        // Get temporal compression for inject_conditioning
        let temporal_compression = self.config.vae.temporal_compression;

        // Update conditioning items with the desired strength
        let mut updated_items = Vec::new();
        for mut item in conditioning_items {
            item.scale = strength;
            updated_items.push(item);
        }

        // Inject conditioning into latents
        inject_conditioning(latents, &updated_items, temporal_compression)
            .map_err(|e| candle::Error::Msg(e.to_string()))
    }

    /// Main denoising loop with classifier-free guidance
    fn denoise_loop(
        &self,
        mut latents: Tensor,
        prompt_embeds: &TextConditioning,
        guidance_scale: f64,
        scheduler: &RectifiedFlowScheduler,
    ) -> Result<Tensor> {
        let timesteps = scheduler.timesteps();
        let num_steps = timesteps.len();

        let use_cfg = guidance_scale > 1.0;

        for (step, &timestep) in timesteps.iter().enumerate() {
            println!(
                "  Step {}/{}, timestep: {:.4}",
                step + 1,
                num_steps,
                timestep
            );

            // Prepare timestep tensor
            let timestep_tensor = Tensor::new(&[timestep as f32], &self.device)?;

            // Expand latents for CFG if needed
            let latent_model_input = if use_cfg {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Prepare encoder hidden states
            let encoder_hidden_states = if use_cfg {
                // Concatenate conditional and unconditional embeddings
                Tensor::cat(
                    &[
                        &prompt_embeds.prompt_embeds,
                        &prompt_embeds.negative_prompt_embeds,
                    ],
                    0,
                )?
            } else {
                prompt_embeds.prompt_embeds.clone()
            };

            // Forward pass through transformer
            let noise_pred = self.transformer.forward(
                &latent_model_input,
                &encoder_hidden_states,
                &timestep_tensor,
            )?;

            // Apply classifier-free guidance
            let noise_pred = if use_cfg {
                // Split predictions
                let (cond_pred, uncond_pred) = Self::split_tensor(&noise_pred, 0)?;

                // Apply guidance: pred = uncond + guidance_scale * (cond - uncond)
                let diff = cond_pred.sub(&uncond_pred)?;
                let scaled_diff =
                    diff.mul(&Tensor::new(&[guidance_scale as f32], &self.device)?)?;
                uncond_pred.add(&scaled_diff)?
            } else {
                noise_pred
            };

            // Update latents using scheduler
            latents = scheduler.step(&noise_pred, timestep, &latents)?;
        }

        Ok(latents)
    }

    /// Post-process decoded video to [0, 255] range
    fn postprocess_video(&self, video: &Tensor) -> Result<Tensor> {
        // Video is in range [-1, 1], convert to [0, 255]
        // Step 1: Scale to [0, 1]
        let video = video
            .add(&Tensor::new(&[1.0f32], &self.device)?)? // [-1, 1] -> [0, 2]
            .div(&Tensor::new(&[2.0f32], &self.device)?)?; // [0, 2] -> [0, 1]

        // Step 2: Scale to [0, 255]
        let video = video.mul(&Tensor::new(&[255.0f32], &self.device)?)?;

        // Step 3: Clamp to valid range
        let video = video.clamp(0.0f32, 255.0f32)?;

        Ok(video)
    }

    /// Validate pipeline inputs
    fn validate_inputs(&self, inputs: &PipelineInputs) -> std::result::Result<(), LtxVideoError> {
        // Validate height and width
        if !inputs.height.is_multiple_of(32) {
            return Err(LtxVideoError::InvalidDimension("Height".to_string(), 32));
        }
        if !inputs.width.is_multiple_of(32) {
            return Err(LtxVideoError::InvalidDimension("Width".to_string(), 32));
        }

        // Validate num_frames follows 8n+1 pattern
        if (inputs.num_frames as i32 - 1) % 8 != 0 {
            return Err(LtxVideoError::InvalidFrameCount(inputs.num_frames));
        }

        // Validate conditioning
        if let (Some(frames), Some(indices)) =
            (&inputs.conditioning_frames, &inputs.conditioning_indices)
        {
            if frames.len() != indices.len() {
                return Err(LtxVideoError::ConfigError(format!(
                    "Conditioning frames count ({}) must match indices count ({})",
                    frames.len(),
                    indices.len()
                )));
            }

            // Validate all indices are within bounds
            for &idx in indices.iter() {
                if idx >= inputs.num_frames {
                    return Err(LtxVideoError::ConditioningFrameOutOfBounds(
                        idx,
                        inputs.num_frames - 1,
                    ));
                }
            }
        }

        // Validate guidance scale
        if inputs.guidance_scale < 0.0 {
            return Err(LtxVideoError::ConfigError(format!(
                "Guidance scale must be >= 0.0, got {}",
                inputs.guidance_scale
            )));
        }

        // Validate conditioning strength
        if inputs.conditioning_strength < 0.0 || inputs.conditioning_strength > 1.0 {
            return Err(LtxVideoError::ConfigError(format!(
                "Conditioning strength must be in [0.0, 1.0], got {}",
                inputs.conditioning_strength
            )));
        }

        Ok(())
    }

    /// Split tensor along dimension
    fn split_tensor(tensor: &Tensor, dim: usize) -> Result<(Tensor, Tensor)> {
        let size = tensor.dim(dim)?;
        let half = size / 2;

        let first = tensor.narrow(dim, 0, half)?;
        let second = tensor.narrow(dim, half, half)?;

        Ok((first, second))
    }

    /// Get device used by the pipeline
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get configuration
    pub fn config(&self) -> &LtxVideoConfig {
        &self.config
    }

    /// Save generated video to file
    ///
    /// # Arguments
    /// * `video_tensor` - Tensor in (B, C, F, H, W) format with values [0, 255]
    /// * `output_path` - Base path for output (without extension)
    /// * `config` - Output configuration
    ///
    /// # Returns
    /// Information about the generated files
    pub fn save_video(
        &self,
        video_tensor: &Tensor,
        output_path: impl AsRef<Path>,
        config: &VideoOutputConfig,
    ) -> Result<VideoGenerationResult> {
        generate_video_output(video_tensor, output_path, config)
    }

    /// Generate and save video in a single call
    ///
    /// # Arguments
    /// * `inputs` - Pipeline input parameters
    /// * `output_path` - Base path for output (without extension)
    /// * `video_config` - Video output configuration
    ///
    /// # Returns
    /// Information about the generated files
    pub fn generate_and_save(
        &mut self,
        inputs: PipelineInputs,
        output_path: impl AsRef<Path>,
        video_config: &VideoOutputConfig,
    ) -> Result<VideoGenerationResult> {
        // Generate video
        let pipeline_output = self.generate(inputs)?;

        // Save video
        self.save_video(&pipeline_output.video, output_path, video_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_inputs_validation() {
        let _inputs = PipelineInputs {
            prompt: "test".to_string(),
            height: 512,
            width: 768,
            num_frames: 121,
            ..Default::default()
        };

        let _config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        let _device = Device::Cpu;

        // Just test that validation logic works
        // We can't create a full pipeline without weights
    }

    #[test]
    fn test_invalid_height() {
        let _inputs = PipelineInputs {
            prompt: "test".to_string(),
            height: 513, // Not divisible by 32
            width: 768,
            num_frames: 121,
            ..Default::default()
        };

        // Would fail validation if we had a pipeline instance
    }

    #[test]
    fn test_invalid_frames() {
        let _inputs = PipelineInputs {
            prompt: "test".to_string(),
            height: 512,
            width: 768,
            num_frames: 120, // Not 8n+1
            ..Default::default()
        };
    }
}
