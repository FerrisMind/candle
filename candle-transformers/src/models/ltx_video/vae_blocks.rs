//! Building blocks for the Causal Video Autoencoder
//!
//! This module implements 3D convolution blocks, residual blocks,
//! and attention blocks used in the VAE.

use candle::{Result, Tensor};
use candle_nn::{group_norm, GroupNorm, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Causal 3D Convolution with temporal causal padding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalConv3dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
}

/// Causal 3D Convolution implementation
///
/// This struct implements 3D convolutions by treating them as a series of 2D convolutions
/// applied across the temporal dimension with causal padding.
#[derive(Clone, Debug)]
pub struct CausalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    _kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize, usize, usize, usize), // (pad_t_left, pad_t_right, pad_h_left, pad_h_right, pad_w_left, pad_w_right)
}

impl CausalConv3d {
    /// Create a new CausalConv3d layer
    pub fn new(vb: VarBuilder, config: CausalConv3dConfig) -> Result<Self> {
        let (k_t, k_h, k_w) = config.kernel_size;
        let in_c = config.in_channels;
        let out_c = config.out_channels;

        // Causal padding for temporal dimension: pad left by k_t - 1
        let pad_t = k_t - 1;
        // Symmetric padding for spatial dimensions
        let pad_h = k_h / 2;
        let pad_w = k_w / 2;

        // Padding: (pad_t_left, pad_t_right, pad_h_left, pad_h_right, pad_w_left, pad_w_right)
        let padding = (pad_t, 0, pad_h, pad_h, pad_w, pad_w);

        // Weight shape: (out_channels, in_channels, k_t, k_h, k_w)
        let weight = vb.get((out_c, in_c, k_t, k_h, k_w), "weight")?;
        let bias = vb.get(out_c, "bias").ok();

        Ok(Self {
            weight,
            bias,
            _kernel_size: config.kernel_size,
            stride: config.stride,
            padding,
        })
    }

    /// Forward pass with causal padding in temporal dimension
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, F, H, W)
        let (b, c_in, _f, _h, _w) = x.dims5()?;
        let (out_c, in_c_w, k_t, k_h, k_w) = self.weight.dims5()?;

        if c_in != in_c_w {
            return Err(candle::Error::Msg(format!(
                "Input channels mismatch: got {}, expected {}",
                c_in, in_c_w
            )));
        }

        // Apply temporal causal padding and spatial symmetric padding
        let (pad_t_l, pad_t_r, pad_h_l, pad_h_r, pad_w_l, pad_w_r) = self.padding;

        // Pad temporal dimension (dim 2)
        let x = x.pad_with_zeros(2, pad_t_l, pad_t_r)?;
        // Pad height (dim 3)
        let x = x.pad_with_zeros(3, pad_h_l, pad_h_r)?;
        // Pad width (dim 4)
        let x = x.pad_with_zeros(4, pad_w_l, pad_w_r)?;

        let (_, _, f_padded, h_padded, w_padded) = x.dims5()?;

        // Compute output spatial dimensions
        let (s_t, s_h, s_w) = self.stride;
        let h_out = (h_padded - k_h) / s_h + 1;
        let w_out = (w_padded - k_w) / s_w + 1;
        let f_out = (f_padded - k_t) / s_t + 1;

        // Perform 3D convolution
        if k_t == 1 {
            // Optimized path for temporal kernel = 1
            let x_reshaped = x.reshape((b * f_padded, c_in, h_padded, w_padded))?;

            // Weight shape for 2D conv: (out_c, in_c, k_h, k_w)
            let weight_2d = self.weight.squeeze(2)?; // Remove temporal dim
            let out = x_reshaped.conv2d(&weight_2d, 0, s_h, s_w, 1)?;

            // Add bias if present
            let out = if let Some(bias) = &self.bias {
                let bias = bias.reshape((1, out_c, 1, 1))?;
                out.broadcast_add(&bias)?
            } else {
                out
            };

            // Reshape back: (B*F_p, out_c, H_out, W_out) -> (B, out_c, F_p, H_out, W_out)
            let out = out.reshape((b, f_padded, out_c, h_out, w_out))?;
            let out = out.permute(vec![0, 2, 1, 3, 4])?; // (B, out_c, F_p, H_out, W_out)

            // Apply temporal stride if needed
            self.apply_stride(&out, (s_t, 1, 1))
        } else {
            // General 3D convolution for k_t > 1
            // Process each output frame by accumulating contributions from temporal kernel
            let mut output_frames = Vec::with_capacity(f_out);

            for t_out in 0..f_out {
                let t_start = t_out * s_t;
                let _t_end = t_start + k_t;

                // For each temporal position in the kernel
                let mut frame_accum: Option<Tensor> = None;

                for t_k in 0..k_t {
                    let t_in = t_start + t_k;

                    // Extract spatial slice at this temporal position: (B, C, H_p, W_p)
                    let x_slice = x.narrow(2, t_in, 1)?.squeeze(2)?;

                    // Extract weight for this temporal kernel position: (out_c, in_c, k_h, k_w)
                    let weight_slice = self.weight.narrow(2, t_k, 1)?.squeeze(2)?;

                    // Perform 2D convolution
                    let x_batch = x_slice.reshape((b, c_in, h_padded, w_padded))?;
                    let conv_out = x_batch.conv2d(&weight_slice, 0, s_h, s_w, 1)?;

                    // Accumulate
                    frame_accum = Some(match frame_accum {
                        None => conv_out,
                        Some(acc) => (acc + conv_out)?,
                    });
                }

                // Add bias to accumulated frame
                let frame_out = if let Some(mut frame) = frame_accum {
                    if let Some(bias) = &self.bias {
                        let bias = bias.reshape((1, out_c, 1, 1))?;
                        frame = frame.broadcast_add(&bias)?;
                    }
                    frame
                } else {
                    return Err(candle::Error::Msg(
                        "No frames accumulated in 3D conv".to_string(),
                    ));
                };

                output_frames.push(frame_out);
            }

            // Stack output frames along temporal dimension
            // Each frame is (B, out_c, H_out, W_out), stack to (B, out_c, F_out, H_out, W_out)
            let stacked = Tensor::stack(&output_frames, 2)?;
            Ok(stacked)
        }
    }

    fn apply_stride(&self, x: &Tensor, stride: (usize, usize, usize)) -> Result<Tensor> {
        let (s_t, s_h, s_w) = stride;
        if s_t == 1 && s_h == 1 && s_w == 1 {
            return Ok(x.clone());
        }

        let (_b, _c, f, _h, _w) = x.dims5()?;
        let mut out = x.clone();

        // Subsample temporal dimension
        if s_t > 1 {
            let indices = Tensor::arange_step(0u32, f as u32, s_t as u32, x.device())?;
            out = out.index_select(&indices, 2)?;
        }

        // Subsample height
        if s_h > 1 {
            let (_, _, _f, h, _) = out.dims5()?;
            let indices = Tensor::arange_step(0u32, h as u32, s_h as u32, x.device())?;
            out = out.index_select(&indices, 3)?;
        }

        // Subsample width
        if s_w > 1 {
            let (_, _, _, _, w) = out.dims5()?;
            let indices = Tensor::arange_step(0u32, w as u32, s_w as u32, x.device())?;
            out = out.index_select(&indices, 4)?;
        }

        Ok(out)
    }
}

/// 3D Residual Block with group normalization
#[derive(Clone, Debug)]
pub struct ResBlock3d {
    norm1: GroupNorm,
    conv1: CausalConv3d,
    norm2: GroupNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResBlock3d {
    /// Create a new ResBlock3d
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let norm1 = group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;

        let conv1 = CausalConv3d::new(
            vb.pp("conv1"),
            CausalConv3dConfig {
                in_channels,
                out_channels,
                kernel_size: (1, 3, 3), // Using k_t=1 for now
                stride: (1, 1, 1),
                padding: (0, 1, 1),
            },
        )?;

        let norm2 = group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;

        let conv2 = CausalConv3d::new(
            vb.pp("conv2"),
            CausalConv3dConfig {
                in_channels: out_channels,
                out_channels,
                kernel_size: (1, 3, 3),
                stride: (1, 1, 1),
                padding: (0, 1, 1),
            },
        )?;

        let shortcut = if in_channels != out_channels {
            Some(CausalConv3d::new(
                vb.pp("conv_shortcut"),
                CausalConv3dConfig {
                    in_channels,
                    out_channels,
                    kernel_size: (1, 1, 1),
                    stride: (1, 1, 1),
                    padding: (0, 0, 0),
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    /// Forward pass with skip connection
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;

        let h = self.norm1.forward(x)?;
        let h = h.silu()?; // Swish activation
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        let shortcut = if let Some(s) = &self.shortcut {
            s.forward(residual)?
        } else {
            residual.clone()
        };

        h + shortcut
    }
}

/// 3D Attention Block for VAE
///
/// Implements spatial self-attention across the spatial dimensions (H, W)
/// for each frame independently
#[derive(Clone, Debug)]
pub struct AttentionBlock3d {
    norm: GroupNorm,
    qkv: CausalConv3d,
    proj_out: CausalConv3d,
    num_heads: usize,
    channels: usize,
}

impl AttentionBlock3d {
    /// Create a new AttentionBlock3d
    pub fn new(vb: VarBuilder, channels: usize, num_heads: usize) -> Result<Self> {
        let norm = group_norm(32, channels, 1e-6, vb.pp("norm"))?;

        // QKV projection: channels -> 3 * channels
        let qkv = CausalConv3d::new(
            vb.pp("qkv"),
            CausalConv3dConfig {
                in_channels: channels,
                out_channels: 3 * channels,
                kernel_size: (1, 1, 1),
                stride: (1, 1, 1),
                padding: (0, 0, 0),
            },
        )?;

        // Output projection: channels -> channels
        let proj_out = CausalConv3d::new(
            vb.pp("proj_out"),
            CausalConv3dConfig {
                in_channels: channels,
                out_channels: channels,
                kernel_size: (1, 1, 1),
                stride: (1, 1, 1),
                padding: (0, 0, 0),
            },
        )?;

        Ok(Self {
            norm,
            qkv,
            proj_out,
            num_heads,
            channels,
        })
    }

    /// Forward pass with spatial self-attention
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, _c, f, h, w) = x.dims5()?;

        // Normalize
        let h_tensor = self.norm.forward(x)?;

        // Generate Q, K, V
        let qkv = self.qkv.forward(&h_tensor)?;
        let (_, c_qkv, _, _, _) = qkv.dims5()?;

        // Split into Q, K, V: (B, 3*C, F, H, W) -> 3 x (B, C, F, H, W)
        let chunk_size = c_qkv / 3;
        let q = qkv.narrow(1, 0, chunk_size)?;
        let k = qkv.narrow(1, chunk_size, chunk_size)?;
        let v = qkv.narrow(1, 2 * chunk_size, chunk_size)?;

        // Process each frame independently
        // Reshape for multi-head attention: (B, C, F, H, W) -> (B*F, num_heads, H*W, head_dim)
        let head_dim = self.channels / self.num_heads;

        let q = q.reshape((b * f, self.num_heads, head_dim, h * w))?;
        let q = q.transpose(2, 3)?; // (B*F, num_heads, H*W, head_dim)

        let k = k.reshape((b * f, self.num_heads, head_dim, h * w))?; // (B*F, num_heads, head_dim, H*W)

        let v = v.reshape((b * f, self.num_heads, head_dim, h * w))?;
        let v = v.transpose(2, 3)?; // (B*F, num_heads, H*W, head_dim)

        // Compute attention scores: Q @ K^T
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k)?;
        let attn_scores = (attn_scores * scale)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        // Apply attention to values: attn @ V
        let out = attn_weights.matmul(&v)?; // (B*F, num_heads, H*W, head_dim)

        // Reshape back: (B*F, num_heads, H*W, head_dim) -> (B, C, F, H, W)
        let out = out.transpose(2, 3)?; // (B*F, num_heads, head_dim, H*W)
        let out = out.reshape((b * f, self.channels, h, w))?;
        let out = out.reshape((b, f, self.channels, h, w))?;
        let out = out.permute(vec![0, 2, 1, 3, 4])?; // (B, C, F, H, W)

        // Output projection
        let out = self.proj_out.forward(&out)?;

        // Add residual
        out + residual
    }
}

/// Pixel Shuffle for 3D upsampling
///
/// Rearranges channels into spatial and temporal dimensions for upsampling
#[derive(Clone, Debug)]
pub struct PixelShuffle3d {
    upscale_factor: (usize, usize, usize), // (t_factor, h_factor, w_factor)
}

impl PixelShuffle3d {
    /// Create a new PixelShuffle3d
    ///
    /// # Arguments
    /// * `upscale_factor` - Upsampling factors for (temporal, height, width) dimensions
    pub fn new(upscale_factor: (usize, usize, usize)) -> Result<Self> {
        Ok(PixelShuffle3d { upscale_factor })
    }

    /// Forward pass
    ///
    /// Rearranges (B, C*r_t*r_h*r_w, F, H, W) -> (B, C, F*r_t, H*r_h, W*r_w)
    /// where r_t, r_h, r_w are upscale factors
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, f, h, w) = x.dims5()?;
        let (r_t, r_h, r_w) = self.upscale_factor;

        // Calculate output channels
        let c_out = c_in / (r_t * r_h * r_w);

        if c_in % (r_t * r_h * r_w) != 0 {
            return Err(candle::Error::Msg(format!(
                "Input channels {} not divisible by upscale factor product {}",
                c_in,
                r_t * r_h * r_w
            )));
        }

        // Simplified implementation for common cases
        if r_t == 1 && r_h == 2 && r_w == 2 {
            // Spatial upsampling by 2x2 only
            // (B, C*4, F, H, W) -> (B, C, F, H*2, W*2)

            // Step 1: Reshape channels: (B, C*4, F, H, W) -> (B, C, 4, F, H, W)
            let mut x = x.reshape((b, c_out, 4, f, h, w))?;

            // Step 2: Reshape upscale factors: (B, C, 4, F, H, W) -> (B, C, 2, 2, F, H, W)
            // We need to process width and height separately

            // First handle width: (B, C*4, F, H, W) -> (B, C*2, F, H, W*2)
            x = x.reshape((b, c_out * 2, 2, f, h, w))?;
            x = x.permute(vec![0, 1, 3, 4, 5, 2])?; // (B, C*2, F, H, W, 2)
            x = x.reshape((b, c_out * 2, f, h, w * 2))?;

            // Then handle height: (B, C*2, F, H, W*2) -> (B, C, F, H*2, W*2)
            x = x.reshape((b, c_out, 2, f, h, w * 2))?;
            x = x.permute(vec![0, 1, 3, 4, 2, 5])?; // (B, C, F, H, 2, W*2)
            x = x.reshape((b, c_out, f, h * 2, w * 2))?;

            Ok(x)
        } else {
            // General case - not implemented
            Err(candle::Error::Msg(format!(
                "PixelShuffle3d only supports (1, 2, 2) upscale factors. Got ({}, {}, {})",
                r_t, r_h, r_w
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_causal_conv3d_config() {
        let config = CausalConv3dConfig {
            in_channels: 3,
            out_channels: 128,
            kernel_size: (1, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 1, 1),
        };
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.out_channels, 128);
    }

    #[test]
    fn test_causal_conv3d_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = CausalConv3dConfig {
            in_channels: 3,
            out_channels: 64,
            kernel_size: (1, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 1, 1),
        };
        let _conv = CausalConv3d::new(vb, config)?;
        Ok(())
    }
}
