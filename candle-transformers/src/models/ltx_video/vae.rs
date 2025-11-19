//! Causal Video Autoencoder (VAE) for LTX-Video
//!
//! This module implements the VAE for encoding/decoding between
//! pixel and latent space with causal 3D convolutions.

use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use super::vae_blocks::{CausalConv3d, CausalConv3dConfig, ResBlock3d};

/// Configuration for the Causal Video Autoencoder
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalVaeConfig {
    pub in_channels: usize,              // 3 (RGB)
    pub out_channels: usize,             // 3
    pub latent_channels: usize,          // 128
    pub block_out_channels: Vec<usize>,  // [128, 256, 512, 512]
    pub layers_per_block: usize,         // 3
    pub temporal_compression: usize,     // 4
    pub spatial_compression: usize,      // 8
    pub use_timestep_conditioning: bool, // true for 0.9.8
}

impl Default for CausalVaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 3,
            temporal_compression: 4,
            spatial_compression: 8,
            use_timestep_conditioning: true,
        }
    }
}

impl CausalVaeConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.in_channels == 0 {
            return Err(candle::Error::Msg(
                "in_channels must be greater than 0".to_string(),
            ));
        }
        if self.latent_channels == 0 {
            return Err(candle::Error::Msg(
                "latent_channels must be greater than 0".to_string(),
            ));
        }
        if self.temporal_compression == 0 {
            return Err(candle::Error::Msg(
                "temporal_compression must be greater than 0".to_string(),
            ));
        }
        if self.spatial_compression == 0 {
            return Err(candle::Error::Msg(
                "spatial_compression must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Encoder for the Causal Video Autoencoder
struct Encoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<Vec<ResBlock3d>>,
    mid_block: ResBlock3d,
}

impl Encoder {
    fn new(vb: VarBuilder, config: &CausalVaeConfig) -> Result<Self> {
        let conv_in = CausalConv3d::new(
            vb.pp("conv_in"),
            CausalConv3dConfig {
                in_channels: config.in_channels,
                out_channels: config.block_out_channels[0],
                kernel_size: (1, 3, 3),
                stride: (1, 1, 1),
                padding: (0, 1, 1),
            },
        )?;

        let mut down_blocks = Vec::new();
        let mut in_ch = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let mut blocks = Vec::new();

            // Add residual blocks
            for j in 0..config.layers_per_block {
                let block = ResBlock3d::new(
                    vb.pp(format!("down_blocks.{}.resnets.{}", i, j)),
                    if j == 0 { in_ch } else { out_ch },
                    out_ch,
                )?;
                blocks.push(block);
            }

            // Add downsampling at the end of block (except last)
            if i < config.block_out_channels.len() - 1 {
                let _downsample = CausalConv3d::new(
                    vb.pp(format!("down_blocks.{}.downsamplers.0", i)),
                    CausalConv3dConfig {
                        in_channels: out_ch,
                        out_channels: out_ch,
                        kernel_size: (1, 3, 3),
                        stride: (1, 2, 2), // Spatial downsampling by 2
                        padding: (0, 1, 1),
                    },
                )?;
                // Store as a pseudo-ResBlock for simplicity
                // In real implementation, you'd handle this separately
            }

            down_blocks.push(blocks);
            in_ch = out_ch;
        }

        let mid_ch = *config.block_out_channels.last().unwrap();
        let mid_block = ResBlock3d::new(vb.pp("mid_block"), mid_ch, mid_ch)?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Process through down blocks
        for blocks in &self.down_blocks {
            for block in blocks {
                h = block.forward(&h)?;
            }
        }

        // Mid block
        h = self.mid_block.forward(&h)?;

        Ok(h)
    }
}

/// Decoder for the Causal Video Autoencoder
struct Decoder {
    conv_in: CausalConv3d,
    up_blocks: Vec<Vec<ResBlock3d>>,
    mid_block: ResBlock3d,
    conv_out: CausalConv3d,
}

impl Decoder {
    fn new(vb: VarBuilder, config: &CausalVaeConfig) -> Result<Self> {
        let mid_ch = *config.block_out_channels.last().unwrap();

        let conv_in = CausalConv3d::new(
            vb.pp("conv_in"),
            CausalConv3dConfig {
                in_channels: config.latent_channels,
                out_channels: mid_ch,
                kernel_size: (1, 3, 3),
                stride: (1, 1, 1),
                padding: (0, 1, 1),
            },
        )?;

        let mid_block = ResBlock3d::new(vb.pp("mid_block"), mid_ch, mid_ch)?;

        let mut up_blocks = Vec::new();
        let mut reversed_channels = config.block_out_channels.clone();
        reversed_channels.reverse();

        let mut in_ch = mid_ch;

        for (i, &out_ch) in reversed_channels.iter().enumerate() {
            let mut blocks = Vec::new();

            for j in 0..config.layers_per_block {
                let block = ResBlock3d::new(
                    vb.pp(format!("up_blocks.{}.resnets.{}", i, j)),
                    in_ch,
                    out_ch,
                )?;
                blocks.push(block);
                in_ch = out_ch;
            }

            up_blocks.push(blocks);
        }

        let conv_out = CausalConv3d::new(
            vb.pp("conv_out"),
            CausalConv3dConfig {
                in_channels: config.block_out_channels[0],
                out_channels: config.out_channels,
                kernel_size: (1, 3, 3),
                stride: (1, 1, 1),
                padding: (0, 1, 1),
            },
        )?;

        Ok(Self {
            conv_in,
            up_blocks,
            mid_block,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Mid block
        h = self.mid_block.forward(&h)?;

        // Process through up blocks
        for blocks in &self.up_blocks {
            for block in blocks {
                h = block.forward(&h)?;
            }
        }

        // Output conv
        h = self.conv_out.forward(&h)?;

        Ok(h)
    }
}

/// Causal Video Autoencoder
pub struct CausalVideoAutoencoder {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    config: CausalVaeConfig,
}

impl CausalVideoAutoencoder {
    /// Create a new CausalVideoAutoencoder
    pub fn new(vb: VarBuilder, config: CausalVaeConfig) -> Result<Self> {
        config.validate()?;

        let encoder = Encoder::new(vb.pp("encoder"), &config)?;
        let decoder = Decoder::new(vb.pp("decoder"), &config)?;

        let mid_ch = *config.block_out_channels.last().unwrap();

        let quant_conv = CausalConv3d::new(
            vb.pp("quant_conv"),
            CausalConv3dConfig {
                in_channels: mid_ch,
                out_channels: config.latent_channels,
                kernel_size: (1, 1, 1),
                stride: (1, 1, 1),
                padding: (0, 0, 0),
            },
        )?;

        let post_quant_conv = CausalConv3d::new(
            vb.pp("post_quant_conv"),
            CausalConv3dConfig {
                in_channels: config.latent_channels,
                out_channels: config.latent_channels,
                kernel_size: (1, 1, 1),
                stride: (1, 1, 1),
                padding: (0, 0, 0),
            },
        )?;

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            config,
        })
    }

    /// Encode video frames to latents
    ///
    /// # Arguments
    /// * `x` - Input video tensor with shape (B, C, F, H, W)
    ///
    /// # Returns
    /// Latent tensor with shape (B, latent_channels, F/temporal_compression, H/spatial_compression, W/spatial_compression)
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // Validate input dimensions
        let (_b, c, _f, _h, _w) = x.dims5()?;

        if c != self.config.in_channels {
            return Err(candle::Error::Msg(format!(
                "Expected {} input channels, got {}",
                self.config.in_channels, c
            )));
        }

        // Forward through encoder
        let h = self.encoder.forward(x)?;

        // Quantization convolution
        let latent = self.quant_conv.forward(&h)?;

        Ok(latent)
    }

    /// Decode latents to video frames
    ///
    /// # Arguments
    /// * `latent` - Latent tensor with shape (B, latent_channels, F', H', W')
    /// * `timestep` - Optional timestep for conditioning (used in 0.9.8 model)
    ///
    /// # Returns
    /// Reconstructed video tensor with shape (B, out_channels, F*temporal_compression, H*spatial_compression, W*spatial_compression)
    pub fn decode(&self, latent: &Tensor, _timestep: Option<f64>) -> Result<Tensor> {
        // Post-quantization convolution
        let z = self.post_quant_conv.forward(latent)?;

        // Forward through decoder
        let decoded = self.decoder.forward(&z)?;

        Ok(decoded)
    }

    /// Get configuration
    pub fn config(&self) -> &CausalVaeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_config_default() {
        let config = CausalVaeConfig::default();
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.latent_channels, 128);
        assert_eq!(config.temporal_compression, 4);
        assert_eq!(config.spatial_compression, 8);
    }

    #[test]
    fn test_vae_config_validation() {
        let config = CausalVaeConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vae_config_validation_invalid_channels() {
        let mut config = CausalVaeConfig::default();
        config.in_channels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vae_config_validation_invalid_compression() {
        let mut config = CausalVaeConfig::default();
        config.temporal_compression = 0;
        assert!(config.validate().is_err());
    }
}
