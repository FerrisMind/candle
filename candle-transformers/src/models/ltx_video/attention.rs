//! Attention mechanisms for LTX-Video
//!
//! This module implements multi-head attention with RoPE support,
//! QK normalization, and both self-attention and cross-attention modes.

use candle::{Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Multi-head attention with RoPE and QK normalization support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub use_rope: bool,
    pub qk_norm: bool,
    pub eps: f64,
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(num_heads: usize, head_dim: usize, use_rope: bool, qk_norm: bool) -> Self {
        Self {
            num_heads,
            head_dim,
            use_rope,
            qk_norm,
            eps: 1e-6,
        }
    }

    /// Get the total hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Attention layer for LTX-Video
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    use_rope: bool,
    qk_norm: bool,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        vb: VarBuilder,
        config: &AttentionConfig,
        cross_attention_dim: Option<usize>,
    ) -> Result<Self> {
        let inner_dim = config.hidden_dim();
        let query_dim = inner_dim;
        let context_dim = cross_attention_dim.unwrap_or(inner_dim);

        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(context_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(context_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out"))?;

        let mut q_norm = None;
        let mut k_norm = None;
        if config.qk_norm {
            q_norm = Some(layer_norm(config.head_dim, config.eps, vb.pp("q_norm"))?);
            k_norm = Some(layer_norm(config.head_dim, config.eps, vb.pp("k_norm"))?);
        }

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads: config.num_heads,
            dim_head: config.head_dim,
            use_rope: config.use_rope,
            qk_norm: config.qk_norm,
            q_norm,
            k_norm,
        })
    }

    fn reshape_heads_to_batch(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        x.reshape((batch, seq_len, self.heads, self.dim_head))?
            .transpose(1, 2)
    }

    fn reshape_batch_to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, heads, seq_len, dim_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((batch, seq_len, heads * dim_head))
    }

    /// Forward pass for self-attention
    pub fn forward_self_attention(
        &self,
        hidden_states: &Tensor,
        rope_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        let mut q = self.reshape_heads_to_batch(&q)?;
        let mut k = self.reshape_heads_to_batch(&k)?;
        let v = self.reshape_heads_to_batch(&v)?;

        if self.qk_norm {
            if let Some(ln) = &self.q_norm {
                q = ln.forward(&q)?;
            }
            if let Some(ln) = &self.k_norm {
                k = ln.forward(&k)?;
            }
        }

        if self.use_rope {
            if let Some((cos, sin)) = rope_cos_sin {
                let (q_rope, k_rope) = self.apply_rope(&q, &k, cos, sin)?;
                q = q_rope;
                k = k_rope;
            }
        }

        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;
        let output = self.reshape_batch_to_heads(&attn_output)?;

        self.to_out.forward(&output)
    }

    /// Forward pass for cross-attention
    pub fn forward_cross_attention(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(encoder_hidden_states)?;
        let v = self.to_v.forward(encoder_hidden_states)?;

        let mut q = self.reshape_heads_to_batch(&q)?;
        let mut k = self.reshape_heads_to_batch(&k)?;
        let v = self.reshape_heads_to_batch(&v)?;

        if self.qk_norm {
            if let Some(ln) = &self.q_norm {
                q = ln.forward(&q)?;
            }
            if let Some(ln) = &self.k_norm {
                k = ln.forward(&k)?;
            }
        }

        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;
        let output = self.reshape_batch_to_heads(&attn_output)?;

        self.to_out.forward(&output)
    }

    /// Apply RoPE to query and key tensors
    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, heads, seq_len, _) = q.dims4()?;

        // cos and sin are (num_positions, head_dim)
        // We assume num_positions == seq_len and broadcast to (batch, heads, seq_len, head_dim)

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let cos = cos.broadcast_as((batch, heads, seq_len, self.dim_head))?;
        let sin = sin.broadcast_as((batch, heads, seq_len, self.dim_head))?;

        let q_rotated = (q.broadcast_mul(&cos)? + &rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_rotated = (k.broadcast_mul(&cos)? + &rotate_half(k)?.broadcast_mul(&sin)?)?;

        Ok((q_rotated, k_rotated))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_attention_config_creation() {
        let config = AttentionConfig::new(24, 96, true, true);
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.head_dim, 96);
        assert!(config.use_rope);
        assert!(config.qk_norm);
    }

    #[test]
    fn test_attention_config_hidden_dim() {
        let config = AttentionConfig::new(24, 96, true, true);
        assert_eq!(config.hidden_dim(), 2304);
    }

    #[test]
    fn test_rotate_half() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let rotated = rotate_half(&x)?;
        // x1 = [1, 2], x2 = [3, 4]
        // res = [-3, -4, 1, 2]
        let expected = Tensor::new(&[[-3.0f32, -4.0, 1.0, 2.0]], &device)?;
        assert_eq!(rotated.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }
}
