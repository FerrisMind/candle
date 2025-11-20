//! 3D Transformer model for LTX-Video
//!
//! This module implements the core Transformer3D model that processes
//! spatiotemporal latent representations for video generation.

use super::attention::{Attention, AttentionConfig};
use super::embeddings::{AdaLayerNormSingle, RoPEEmbedding, TimestepEmbedding};
use candle::{Result, Tensor, D};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for the 3D Transformer model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transformer3DConfig {
    pub num_layers: usize,                   // 28 for 2B model
    pub num_attention_heads: usize,          // 24
    pub attention_head_dim: usize,           // 96
    pub in_channels: usize,                  // 128 (VAE latent channels)
    pub cross_attention_dim: usize,          // 4096 (T5 embedding dim)
    pub patch_size: (usize, usize, usize),   // (1, 2, 2) for (t, h, w)
    pub use_rope: bool,                      // true
    pub rope_theta: f64,                     // 10000.0
    pub qk_norm: bool,                       // true
    pub hidden_act: String,                  // "gelu"
    pub hidden_dropout_prob: f64,            // 0.0
    pub attention_probs_dropout_prob: f64,   // 0.0
    pub initializer_range: f64,              // 0.02
    pub layer_norm_eps: f64,                 // 1e-6
    pub skip_block_list: Option<Vec<usize>>, // List of block indices to skip
}

impl Transformer3DConfig {
    /// Returns the configuration for the ltxv-2b-0.9.8-distilled model variant
    pub fn ltxv_2b_0_9_8_distilled() -> Self {
        Self {
            num_layers: 28,
            num_attention_heads: 24,
            attention_head_dim: 96,
            in_channels: 128,
            cross_attention_dim: 4096,
            patch_size: (1, 2, 2),
            use_rope: true,
            rope_theta: 10000.0,
            qk_norm: true,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            skip_block_list: None,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(candle::Error::Msg(
                "num_layers must be greater than 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(candle::Error::Msg(
                "num_attention_heads must be greater than 0".to_string(),
            ));
        }
        if self.in_channels == 0 {
            return Err(candle::Error::Msg(
                "in_channels must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Get the total hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

pub struct FeedForward {
    net: candle_nn::Sequential,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, dim: usize, hidden_dim: usize, act: &str) -> Result<Self> {
        let net = candle_nn::seq()
            .add(linear(dim, hidden_dim, vb.pp("net.0"))?)
            .add(match act {
                "gelu" => Activation::Gelu,
                _ => Activation::Gelu,
            })
            .add(linear(hidden_dim, dim, vb.pp("net.2"))?);
        Ok(Self { net })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.net.forward(x)
    }
}

pub struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff: FeedForward,
    scale_shift_table: Option<Linear>,
}

impl BasicTransformerBlock {
    pub fn new(vb: VarBuilder, config: &Transformer3DConfig) -> Result<Self> {
        let dim = config.hidden_dim();
        let attn_config = AttentionConfig::new(
            config.num_attention_heads,
            config.attention_head_dim,
            config.use_rope,
            config.qk_norm,
        );

        let norm1 = layer_norm(dim, config.layer_norm_eps, vb.pp("norm1"))?;
        let attn1 = Attention::new(vb.pp("attn1"), &attn_config, None)?;

        let norm2 = layer_norm(dim, config.layer_norm_eps, vb.pp("norm2"))?;
        let attn2 = Attention::new(
            vb.pp("attn2"),
            &attn_config,
            Some(config.cross_attention_dim),
        )?;

        let norm3 = layer_norm(dim, config.layer_norm_eps, vb.pp("norm3"))?;
        let ff = FeedForward::new(vb.pp("ff"), dim, dim * 4, &config.hidden_act)?;

        // AdaLN-Single projection
        // Projects timestep embedding to 6 * dim (scale + shift for 3 norms)
        let scale_shift_table = Some(linear(dim, 6 * dim, vb.pp("scale_shift_table"))?);

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep_emb: Option<&Tensor>,
        rope_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let mut shift_ms = Vec::new();
        let mut scale_ms = Vec::new();

        if let Some(sst) = &self.scale_shift_table {
            if let Some(t_emb) = timestep_emb {
                let scale_shift = sst.forward(t_emb)?;
                // Split into 6 chunks: shift1, scale1, shift2, scale2, shift3, scale3
                let chunks = scale_shift.chunk(6, D::Minus1)?;
                for i in 0..3 {
                    shift_ms.push(chunks[i * 2].clone());
                    scale_ms.push(chunks[i * 2 + 1].clone());
                }
            }
        }

        // 1. Self-Attention
        let norm1 = self.apply_norm(
            &self.norm1,
            hidden_states,
            shift_ms.first(),
            scale_ms.first(),
        )?;
        let attn1 = self.attn1.forward_self_attention(&norm1, rope_cos_sin)?;
        let hidden_states = (hidden_states + attn1)?;

        // 2. Cross-Attention
        let norm2 = self.apply_norm(
            &self.norm2,
            &hidden_states,
            shift_ms.get(1),
            scale_ms.get(1),
        )?;
        let attn2 = self
            .attn2
            .forward_cross_attention(&norm2, encoder_hidden_states)?;
        let hidden_states = (&hidden_states + attn2)?;

        // 3. Feed-Forward
        let norm3 = self.apply_norm(
            &self.norm3,
            &hidden_states,
            shift_ms.get(2),
            scale_ms.get(2),
        )?;
        let ff = self.ff.forward(&norm3)?;
        let hidden_states = (&hidden_states + ff)?;

        Ok(hidden_states)
    }

    fn apply_norm(
        &self,
        norm: &LayerNorm,
        x: &Tensor,
        shift: Option<&Tensor>,
        scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x_norm = norm.forward(x)?;

        if let (Some(shift), Some(scale)) = (shift, scale) {
            // scale, shift: (batch, dim)
            // x_norm: (batch, seq_len, dim)
            // Broadcast scale/shift
            let scale = scale.unsqueeze(1)?.broadcast_as(x_norm.shape())?;
            let shift = shift.unsqueeze(1)?.broadcast_as(x_norm.shape())?;

            // x * (1 + scale) + shift
            let one = Tensor::ones_like(&scale)?;
            (x_norm * (one + scale)?)? + shift
        } else {
            Ok(x_norm)
        }
    }
}

/// Transformer3D model for LTX-Video
#[allow(dead_code)]
pub struct Transformer3D {
    patchify_proj: Linear,
    transformer_blocks: Vec<BasicTransformerBlock>,
    norm_out: AdaLayerNormSingle,
    proj_out: Linear,
    pos_embed: Option<RoPEEmbedding>,
    timestep_embedder: TimestepEmbedding,
    patch_size: (usize, usize, usize),
    in_channels: usize,
    _config: Transformer3DConfig,
}

impl Transformer3D {
    /// Create a new Transformer3D model
    pub fn new(vb: VarBuilder, config: Transformer3DConfig) -> Result<Self> {
        let dim = config.hidden_dim();
        let (pt, ph, pw) = config.patch_size;
        let patch_dim = config.in_channels * pt * ph * pw;

        let patchify_proj = linear(patch_dim, dim, vb.pp("patchify_proj"))?;

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..config.num_layers {
            let block = BasicTransformerBlock::new(vb_blocks.pp(i), &config)?;
            transformer_blocks.push(block);
        }

        // Final normalization is AdaLayerNormSingle
        let norm_out = AdaLayerNormSingle::new(vb.pp("norm_out"), dim)?;

        let proj_out = linear(dim, patch_dim, vb.pp("proj_out"))?;

        let pos_embed = if config.use_rope {
            // Max pos is arbitrary here, will be used for theta
            Some(RoPEEmbedding::new(config.rope_theta, (1, 1, 1)))
        } else {
            None
        };

        // Timestep embedding
        let timestep_embedder = TimestepEmbedding::new(vb.pp("timestep_embedder"), dim, 256)?;

        Ok(Self {
            patchify_proj,
            transformer_blocks,
            norm_out,
            proj_out,
            pos_embed,
            timestep_embedder,
            patch_size: config.patch_size,
            in_channels: config.in_channels,
            _config: config,
        })
    }

    /// Forward pass through the transformer
    pub fn forward(
        &self,
        hidden_states: &Tensor,         // (B, C, F, H, W)
        encoder_hidden_states: &Tensor, // (B, seq_len, 4096)
        timestep: &Tensor,              // (B,)
    ) -> Result<Tensor> {
        let (_batch, _channels, frames, height, width) = hidden_states.dims5()?;

        // 1. Patchify
        // (B, C, F, H, W) -> (B, N, patch_dim)
        let patches = self.patchify(hidden_states)?;

        // 2. Project to hidden dim
        let mut hidden_states = self.patchify_proj.forward(&patches)?;

        // 3. Timestep embedding
        let timestep_emb = self.timestep_embedder.forward(timestep)?;

        // 4. Positional embeddings (RoPE)
        let rope_cos_sin = if let Some(rope) = &self.pos_embed {
            let (pt, ph, pw) = self.patch_size;
            let f_patches = frames / pt;
            let h_patches = height / ph;
            let w_patches = width / pw;

            Some(rope.forward(
                (f_patches, h_patches, w_patches),
                hidden_states.device(),
                hidden_states.dtype(),
            )?)
        } else {
            None
        };
        let rope_refs = rope_cos_sin.as_ref().map(|(c, s)| (c, s));

        // 5. Transformer blocks
        let skip_blocks = self._config.skip_block_list.as_deref().unwrap_or(&[]);
        for (i, block) in self.transformer_blocks.iter().enumerate() {
            if skip_blocks.contains(&i) {
                continue;
            }
            hidden_states = block.forward(
                &hidden_states,
                encoder_hidden_states,
                Some(&timestep_emb),
                rope_refs,
            )?;
        }

        // 6. Final Norm
        hidden_states = self.norm_out.forward(&hidden_states, &timestep_emb)?;

        // 7. Output projection
        hidden_states = self.proj_out.forward(&hidden_states)?;

        // 8. Unpatchify
        // (B, N, patch_dim) -> (B, C, F, H, W)
        self.unpatchify(&hidden_states, (frames, height, width))
    }

    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = x.dims5()?;
        let (pt, ph, pw) = self.patch_size;

        // Check dimensions
        if frames % pt != 0 || height % ph != 0 || width % pw != 0 {
            return Err(candle::Error::Msg(format!(
                "Dimensions ({}, {}, {}) not divisible by patch size ({}, {}, {})",
                frames, height, width, pt, ph, pw
            )));
        }

        let f_patches = frames / pt;
        let h_patches = height / ph;
        let w_patches = width / pw;

        // (B, C, F, H, W) -> (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
        let x = x.reshape(&[batch, channels, f_patches, pt, h_patches, ph, w_patches, pw])?;

        // Permute to (B, F/pt, H/ph, W/pw, C, pt, ph, pw)
        let x = x.permute(vec![0usize, 2, 4, 6, 1, 3, 5, 7])?;

        // Flatten to (B, N, patch_dim)
        // N = f_patches * h_patches * w_patches
        // patch_dim = C * pt * ph * pw
        x.flatten_from(1)?.flatten_from(2)
    }

    fn unpatchify(&self, x: &Tensor, shape: (usize, usize, usize)) -> Result<Tensor> {
        let (batch, num_patches, _patch_dim) = x.dims3()?;
        let (frames, height, width) = shape;
        let (pt, ph, pw) = self.patch_size;
        let channels = self.in_channels;

        let f_patches = frames / pt;
        let h_patches = height / ph;
        let w_patches = width / pw;

        if num_patches != f_patches * h_patches * w_patches {
            return Err(candle::Error::Msg(format!(
                "Number of patches {} does not match dimensions ({}, {}, {}) with patch size ({}, {}, {})",
                num_patches, frames, height, width, pt, ph, pw
            )));
        }

        // (B, N, patch_dim) -> (B, F/pt, H/ph, W/pw, C, pt, ph, pw)
        let x = x.reshape(&[batch, f_patches, h_patches, w_patches, channels, pt, ph, pw])?;

        // Permute to (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
        let x = x.permute(vec![0usize, 4, 1, 5, 2, 6, 3, 7])?;

        // Reshape to (B, C, F, H, W)
        x.reshape(&[batch, channels, frames, height, width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_config_creation() {
        let config = Transformer3DConfig::ltxv_2b_0_9_8_distilled();
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.attention_head_dim, 96);
    }

    #[test]
    fn test_config_validation() {
        let config = Transformer3DConfig::ltxv_2b_0_9_8_distilled();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hidden_dim() {
        let config = Transformer3DConfig::ltxv_2b_0_9_8_distilled();
        assert_eq!(config.hidden_dim(), 24 * 96);
    }

    #[test]
    fn test_feed_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle::DType::F32, &device);
        let ff = FeedForward::new(vb, 128, 512, "gelu")?;
        let x = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;
        let out = ff.forward(&x)?;
        assert_eq!(out.dims(), &[1, 10, 128]);
        Ok(())
    }
}
