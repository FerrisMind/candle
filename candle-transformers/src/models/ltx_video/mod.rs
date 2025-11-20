//! LTX-Video: Latent Transformer for Video Generation
//!
//! LTX-Video is a DiT-based (Diffusion Transformer) video generation model capable of
//! text-to-video and image-to-video generation. This implementation supports the
//! ltxv-2b-0.9.8-distilled model variant.
//!
//! # Example
//!
//! ```ignore
//! use candle_transformers::models::ltx_video::{LtxVideoConfig, Transformer3D};
//! use candle_nn::VarBuilder;
//! use candle::{Device, DType};
//!
//! let device = Device::cuda_if_available(0)?;
//! let config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
//! let vb = VarBuilder::from_safetensors(&["model.safetensors"], DType::F32, &device)?;
//! let transformer = Transformer3D::new(vb, config.transformer)?;
//! ```

pub mod attention;
pub mod conditioning;
pub mod device;
pub mod embeddings;
pub mod patchifier;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer3d;
pub mod vae;
pub mod vae_blocks;

use candle::{DType, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub use attention::Attention;
pub use conditioning::{
    inject_conditioning, load_image, prepare_conditioning_latents, preprocess_image,
    ConditioningItem,
};
pub use device::{Backend, DeviceManager, MemoryEstimator};
pub use embeddings::{AdaLayerNormSingle, RoPEEmbedding};
pub use patchifier::Patchifier;
pub use pipeline::{LTXVideoPipeline, PipelineInputs, PipelineOutput};
pub use scheduler::{RectifiedFlowConfig, RectifiedFlowScheduler};
pub use text_encoder::{T5Config, T5TextEncoder, TextConditioning};
pub use transformer3d::{BasicTransformerBlock, Transformer3D, Transformer3DConfig};
pub use vae::{CausalVaeConfig, CausalVideoAutoencoder};

/// Main configuration struct for LTX-Video model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LtxVideoConfig {
    /// Video generation parameters
    pub width: usize,
    pub height: usize,
    pub num_frames: usize,

    /// Model architecture
    pub transformer: Transformer3DConfig,
    pub vae: CausalVaeConfig,
    pub text_encoder: T5Config,

    /// Sampling parameters
    pub scheduler: RectifiedFlowConfig,
    pub guidance_scale: f64,
    pub num_inference_steps: usize,

    /// Precision settings
    pub dtype: String, // "float32", "float16", "bfloat16"

    /// Backend selection (auto-detect if None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>, // "auto", "cpu", "cuda", "metal"

    /// Optional skip layer configuration
    pub skip_block_list: Option<Vec<usize>>,
}

impl LtxVideoConfig {
    /// Returns the configuration for the ltxv-2b-0.9.8-distilled model variant
    pub fn ltxv_2b_0_9_8_distilled() -> Self {
        Self {
            width: 768,
            height: 512,
            num_frames: 121,
            transformer: Transformer3DConfig::ltxv_2b_0_9_8_distilled(),
            vae: CausalVaeConfig::default(),
            text_encoder: T5Config::default(),
            scheduler: RectifiedFlowConfig::default(),
            guidance_scale: 1.0,
            num_inference_steps: 50,
            dtype: "float32".to_string(),
            backend: None,
            skip_block_list: None,
        }
    }

    /// Load configuration from a YAML file
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| candle::Error::Msg(format!("Failed to read config file: {}", e)))?;
        Self::from_yaml_str(&content)
    }

    /// Load configuration from a YAML string
    pub fn from_yaml_str(yaml_str: &str) -> Result<Self> {
        serde_yaml::from_str(yaml_str)
            .map_err(|e| candle::Error::Msg(format!("Failed to parse YAML config: {}", e)))
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate height and width are divisible by 32
        if !self.height.is_multiple_of(32) {
            return Err(candle::Error::Msg(format!(
                "Height {} must be divisible by 32",
                self.height
            )));
        }
        if !self.width.is_multiple_of(32) {
            return Err(candle::Error::Msg(format!(
                "Width {} must be divisible by 32",
                self.width
            )));
        }

        // Validate num_frames follows 8n+1 pattern
        if (self.num_frames as i32 - 1) % 8 != 0 {
            return Err(candle::Error::Msg(format!(
                "Number of frames {} must follow pattern 8n+1 (e.g., 9, 17, 25, 121, 257)",
                self.num_frames
            )));
        }

        Ok(())
    }

    /// Get the latent shape for this configuration
    pub fn latent_shape(&self) -> (usize, usize, usize, usize) {
        let latent_channels = self.vae.latent_channels;
        let temporal_compression = self.vae.temporal_compression;
        let spatial_compression = self.vae.spatial_compression;

        (
            self.num_frames / temporal_compression,
            latent_channels,
            self.height / spatial_compression,
            self.width / spatial_compression,
        )
    }

    /// Get the DType from the dtype string
    pub fn get_dtype(&self) -> Result<DType> {
        match self.dtype.as_str() {
            "float32" | "f32" => Ok(DType::F32),
            "float16" | "f16" => Ok(DType::F16),
            "bfloat16" | "bf16" => Ok(DType::BF16),
            _ => Err(candle::Error::Msg(format!(
                "Unsupported dtype: {}",
                self.dtype
            ))),
        }
    }

    /// Get the size in bytes for the configured dtype
    pub fn dtype_bytes(&self) -> Result<usize> {
        match self.dtype.as_str() {
            "float32" | "f32" => Ok(4),
            "float16" | "f16" => Ok(2),
            "bfloat16" | "bf16" => Ok(2),
            _ => Err(candle::Error::Msg(format!(
                "Unsupported dtype: {}",
                self.dtype
            ))),
        }
    }

    /// Create a device manager from configuration
    pub fn create_device_manager(&self) -> Result<DeviceManager> {
        use device::Backend;

        match self.backend.as_deref() {
            None | Some("auto") => DeviceManager::auto(),
            Some("cpu") => DeviceManager::new(Backend::Cpu),
            Some("cuda") => DeviceManager::new(Backend::Cuda(0)),
            Some(backend) if backend.starts_with("cuda:") => {
                let idx = backend
                    .strip_prefix("cuda:")
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| {
                        candle::Error::Msg(format!("Invalid CUDA device index: {}", backend))
                    })?;
                DeviceManager::new(Backend::Cuda(idx))
            }
            Some("metal") => DeviceManager::new(Backend::Metal(0)),
            Some(backend) if backend.starts_with("metal:") => {
                let idx = backend
                    .strip_prefix("metal:")
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| {
                        candle::Error::Msg(format!("Invalid Metal device index: {}", backend))
                    })?;
                DeviceManager::new(Backend::Metal(idx))
            }
            Some(backend) => Err(candle::Error::Msg(format!(
                "Unknown backend: {}. Valid options: auto, cpu, cuda, cuda:N, metal, metal:N",
                backend
            ))),
        }
    }

    /// Estimate memory requirements for this configuration
    pub fn estimate_memory(&self) -> Result<MemoryEstimator> {
        use device::MemoryEstimator;

        let mut estimator = MemoryEstimator::new();

        // Estimate model memory (2B parameters for 2B model variant)
        let num_parameters = 2_000_000_000; // 2B model
        let dtype_bytes = self.dtype_bytes()?;
        estimator.estimate_model_memory(num_parameters, dtype_bytes);

        // Estimate activation memory
        estimator.estimate_activation_memory(
            1, // batch size
            self.num_frames,
            self.height,
            self.width,
            dtype_bytes,
        );

        Ok(estimator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_valid_dimensions() {
        let config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_height() {
        let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        config.height = 513; // Not divisible by 32
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_width() {
        let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        config.width = 769; // Not divisible by 32
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_frames() {
        let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        config.num_frames = 10; // Not following 8n+1 pattern
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_valid_frame_patterns() {
        let valid_frames = vec![9, 17, 25, 33, 121, 257];
        for frames in valid_frames {
            let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
            config.num_frames = frames;
            assert!(config.validate().is_ok(), "Failed for frames={}", frames);
        }
    }

    #[test]
    fn test_latent_shape() {
        let config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();
        let (_frames, channels, height, width) = config.latent_shape();
        assert_eq!(channels, 128);
        assert_eq!(height, config.height / 8);
        assert_eq!(width, config.width / 8);
    }

    #[test]
    fn test_get_dtype() {
        let mut config = LtxVideoConfig::ltxv_2b_0_9_8_distilled();

        config.dtype = "float32".to_string();
        assert_eq!(config.get_dtype().unwrap(), DType::F32);

        config.dtype = "f32".to_string();
        assert_eq!(config.get_dtype().unwrap(), DType::F32);

        config.dtype = "float16".to_string();
        assert_eq!(config.get_dtype().unwrap(), DType::F16);

        config.dtype = "bfloat16".to_string();
        assert_eq!(config.get_dtype().unwrap(), DType::BF16);

        config.dtype = "invalid".to_string();
        assert!(config.get_dtype().is_err());
    }
}
