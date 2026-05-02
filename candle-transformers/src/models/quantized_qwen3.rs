//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::{
    quantized_nn::{QEmbedding, RmsNorm},
    utils::repeat_kv,
};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use candle_nn::attention::{flash_attn, AttnMask};
use candle_nn::kv_cache::{ConcatKvCache, InterleavedKvCache, RawInterleavedKvCache};
use candle_nn::{Activation, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Pre-extracted flat f32 cos/sin for fused decode (zero allocation)
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
    rope_theta: f32,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let half_dim = inv_freq.len();
        let mut sin_f32 = Vec::with_capacity(max_seq_len * half_dim);
        let mut cos_f32 = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            let p = pos as f32;
            for &f in &inv_freq {
                let v = p * f;
                sin_f32.push(v.sin());
                cos_f32.push(v.cos());
            }
        }
        let sin =
            Tensor::from_vec(sin_f32.clone(), (max_seq_len, half_dim), dev)?.to_dtype(dtype)?;
        let cos =
            Tensor::from_vec(cos_f32.clone(), (max_seq_len, half_dim), dev)?.to_dtype(dtype)?;
        Ok(Self {
            sin,
            cos,
            cos_f32,
            sin_f32,
            half_d: dim / 2,
            rope_theta: rope_theta as f32,
            head_dim,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        if q.device().is_wgpu() || q.device().is_vulkan() {
            // ggml rope shaders expect positions on i2 and GPT-NeoX pairing.
            const GGML_ROPE_TYPE_NEOX: u32 = 2;
            let (_, _, seq_len, _) = q.dims4()?;
            let positions = Tensor::from_vec(
                (offset..offset + seq_len).map(|v| v as i32).collect(),
                seq_len,
                q.device(),
            )?;
            let q_ggml = q.transpose(1, 2)?.contiguous()?;
            let k_ggml = k.transpose(1, 2)?.contiguous()?;
            let q_embed = candle_nn::rotary_emb::rope_ggml(
                &q_ggml,
                &positions,
                self.head_dim,
                self.rope_theta,
                GGML_ROPE_TYPE_NEOX,
            )?
            .transpose(1, 2)?
            .contiguous()?;
            let k_embed = candle_nn::rotary_emb::rope_ggml(
                &k_ggml,
                &positions,
                self.head_dim,
                self.rope_theta,
                GGML_ROPE_TYPE_NEOX,
            )?
            .transpose(1, 2)?
            .contiguous()?;
            Ok((q_embed, k_embed))
        } else {
            let (_, _, seq_len, _) = q.dims4()?;
            let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
            let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
            let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }
    }

    /// Zero-allocation cos/sin slices for a single position.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<ConcatKvCache>,
    interleaved_cache: Option<InterleavedKvCache>,
    raw_cache: Option<RawInterleavedKvCache>,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden_size = num_heads * head_dim;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // CPU: use interleaved + raw caches for flash attention
        // GPU: use standard concat KV cache (fallback path)
        let on_cpu = device.is_cpu();
        let kv_cache = if on_cpu {
            None
        } else {
            Some(ConcatKvCache::new(2))
        };
        let interleaved_cache = if on_cpu {
            Some(InterleavedKvCache::new(head_dim))
        } else {
            None
        };
        let raw_cache = if on_cpu {
            Some(RawInterleavedKvCache::new(num_kv_heads, head_dim, 4096))
        } else {
            None
        };

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            interleaved_cache,
            raw_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        // QKV projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head Q/K norms (must stay as tensor ops)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // TODO: b > 1 needs varlen CPU flash with interleaved cache support.
        if x.device().is_cpu() && b == 1 {
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            if l == 1 && b == 1 && q.dtype() == DType::F32 {
                // Fused decode: raw slices -> raw cache -> kernel.
                let q_cont = q.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (q_g, q_l) = q_cont.storage_and_layout();
                let q_data: &[f32] = match &*q_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                let k_cont = k.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                let v_cont = v.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                // Write K, V into raw cache (no tensor allocation)
                let k_len = self.num_kv_heads * self.head_dim;
                let rc = self.raw_cache.as_mut().unwrap();
                rc.write_kv(&k_data[..k_len], &v_data[..k_len]);

                // Run interleaved decode kernel
                let kv_len = rc.len();
                let q_len = self.num_heads * self.head_dim;
                let ctx = causal_decode_f32_interleaved(
                    &q_data[..q_len],
                    rc.data(),
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    kv_len,
                    scale,
                )?;

                let ctx = ctx.unsqueeze(0)?.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            } else {
                // Prefill: interleaved cache + flash_attn; also populate raw cache for decode.
                let ic = self.interleaved_cache.as_mut().unwrap();
                let kv = ic.append(&k, &v)?;

                // Populate raw cache for subsequent decode steps
                {
                    let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let (kg, kl) = k_cont.storage_and_layout();
                    let k_d: &[f32] = match &*kg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                        _ => candle::bail!("Expected CPU"),
                    };
                    let (vg, vl) = v_cont.storage_and_layout();
                    let v_d: &[f32] = match &*vg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                        _ => candle::bail!("Expected CPU"),
                    };
                    self.raw_cache.as_mut().unwrap().write_kv_batch(k_d, v_d, l);
                }

                let kv_k = kv.narrow(2, 0, self.head_dim)?.unsqueeze(0)?;
                let kv_v = kv.narrow(2, self.head_dim, self.head_dim)?.unsqueeze(0)?;

                let q = q.transpose(1, 2)?.contiguous()?;
                let k = kv_k.contiguous()?;
                let v = kv_v.contiguous()?;

                let ctx = flash_attn::<f32>(
                    &q,
                    &k,
                    &v,
                    scale,
                    AttnMask::causal_with_offset(offset),
                    None,
                    None,
                )?;
                let ctx = ctx.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            }
        } else {
            // Standard matmul attention (no flash)
            let (k, v) = self.kv_cache.as_mut().unwrap().append(&k, &v)?;

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = attn_mask {
                let scores_dtype = scores.dtype();
                let mask = if m.dtype() != scores_dtype {
                    m.to_dtype(scores_dtype)?
                } else {
                    m.clone()
                };
                scores = scores.broadcast_add(&mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v)?;
            let reshaped_ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
            self.o_proj.forward(&reshaped_ctx)
        }
    }

    fn clear_kv_cache(&mut self) {
        if let Some(c) = &mut self.kv_cache {
            c.reset();
        }
        if let Some(c) = &mut self.interleaved_cache {
            c.reset();
        }
        if let Some(c) = &mut self.raw_cache {
            c.reset();
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        device: &Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            device,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: QEmbedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let _hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = Arc::new(gg.tensor("token_embd.weight")?);
        let embed_tokens = QEmbedding::from_arc(embed_tensor.clone());

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                device,
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head = match gg.tensor("output.weight") {
            Ok(tensor) => QMatMul::from_weights(Arc::new(tensor))?,
            Err(_) => QMatMul::from_weights(embed_tensor)?,
        };
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Skip mask materialization when using CPU flash attention
        let causal_mask = if l == 1 || self.device.is_cpu() {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file;
    use std::path::{Path, PathBuf};

    fn qwen3_gguf_path() -> Option<PathBuf> {
        std::env::var_os("CANDLE_QWEN3_GGUF_PATH").map(PathBuf::from)
    }

    fn load_model(path: &Path, device: &Device) -> Result<ModelWeights> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        ModelWeights::from_gguf(content, &mut file, device)
    }

    fn tensor_diff_stats(actual: &Tensor, expected: &Tensor) -> Result<(usize, f32, f32, f64)> {
        assert_eq!(
            actual.dims(),
            expected.dims(),
            "shape mismatch: {:?} vs {:?}",
            actual.dims(),
            expected.dims()
        );
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_idx = 0usize;
        let mut max_diff = 0f32;
        let mut max_rel = 0f32;
        let mut mse_diff = 0f64;
        let mut mse_ref = 0f64;
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            let rel = diff / expected.abs().max(1.0);
            if diff > max_diff {
                max_idx = idx;
                max_diff = diff;
            }
            max_rel = max_rel.max(rel);
            let diff64 = (*actual as f64) - (*expected as f64);
            mse_diff += diff64 * diff64;
            mse_ref += (*expected as f64) * (*expected as f64);
        }
        let nmse = if mse_ref > 0.0 {
            mse_diff / mse_ref
        } else {
            0.0
        };
        Ok((max_idx, max_diff, max_rel, nmse))
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        tol: f32,
    ) -> Result<()> {
        let (max_idx, max_diff, max_rel, nmse) = tensor_diff_stats(actual, expected)?;
        println!(
            "{label}: max_idx={max_idx} max_diff={max_diff:.6} max_rel={max_rel:.6} nmse={nmse:.6e}"
        );
        assert!(
            max_rel <= tol || nmse <= 5e-4,
            "{label} mismatch: max_idx={max_idx} max_diff={max_diff} max_rel={max_rel} nmse={nmse}"
        );
        Ok(())
    }

    fn log_layout(label: &str, tensor: &Tensor) {
        let (_storage, layout) = tensor.storage_and_layout();
        println!(
            "{label}: dims={:?} stride={:?} start_offset={} contiguous={}",
            layout.dims(),
            layout.stride(),
            layout.start_offset(),
            layout.is_contiguous()
        );
    }

    fn log_qproj_rows(label: &str, tensor: &Tensor) -> Result<()> {
        let tensor = tensor.to_dtype(DType::F32)?;
        match tensor.dims() {
            [b, l, h] if *b == 1 && *l <= 4 && *h >= 4 => {
                let rows = tensor.to_vec3::<f32>()?;
                for row in 0..*l {
                    println!(
                        "{label}[row={row}]: {:.6} {:.6} {:.6} {:.6}",
                        rows[0][row][0], rows[0][row][1], rows[0][row][2], rows[0][row][3],
                    );
                }
            }
            other => println!("{label}: preview skipped for dims={other:?}"),
        }
        Ok(())
    }

    fn log_gguf_tensor_dtype(path: &Path, name: &str) -> Result<()> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let dtype = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle::Error::msg(format!("missing gguf tensor info: {name}")))?;
        println!("{name}: {:?}", dtype.ggml_dtype);
        Ok(())
    }

    #[test]
    #[ignore = "requires CANDLE_QWEN3_GGUF_PATH and a usable wgpu adapter"]
    #[cfg(feature = "wgpu")]
    fn qwen3_local_wgpu_stage_parity() -> Result<()> {
        let Some(path) = qwen3_gguf_path() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let wgpu = Device::new_wgpu(0)?;
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut wgpu_model = load_model(&path, &wgpu)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_wgpu = Tensor::from_slice(&ids, (1, ids.len()), &wgpu)?;

        let emb_cpu = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let emb_wgpu = wgpu_model.embed_tokens.forward(&ids_wgpu)?;
        assert_tensor_close("embedding", &emb_wgpu, &emb_cpu, 1e-4)?;

        let ln1_cpu = cpu_model.layers[0].ln1.forward(&emb_cpu)?;
        let ln1_wgpu = wgpu_model.layers[0].ln1.forward(&emb_wgpu)?;
        assert_tensor_close("ln1", &ln1_wgpu, &ln1_cpu, 1e-4)?;

        let q_proj_cpu = cpu_model.layers[0].self_attn.q_proj.forward(&ln1_cpu)?;
        let q_proj_wgpu = wgpu_model.layers[0].self_attn.q_proj.forward(&ln1_wgpu)?;
        assert_tensor_close("q_proj", &q_proj_wgpu, &q_proj_cpu, 3e-2)?;

        let mask_cpu = cpu_model.causal_mask(1, ids.len(), 0, None)?;
        let mask_wgpu = wgpu_model.causal_mask(1, ids.len(), 0, None)?;
        let layer0_cpu = cpu_model.layers[0].forward(&emb_cpu, Some(&mask_cpu), 0)?;
        let layer0_wgpu = wgpu_model.layers[0].forward(&emb_wgpu, Some(&mask_wgpu), 0)?;
        assert_tensor_close("layer0", &layer0_wgpu, &layer0_cpu, 5e-2)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires CANDLE_QWEN3_GGUF_PATH and a usable Vulkan device"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_stage_parity() -> Result<()> {
        let Some(path) = qwen3_gguf_path() else {
            return Ok(());
        };
        log_gguf_tensor_dtype(&path, "blk.0.attn_q.weight")?;
        log_gguf_tensor_dtype(&path, "blk.0.ffn_up.weight")?;
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut vk_model = load_model(&path, &vk)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_vk = Tensor::from_slice(&ids, (1, ids.len()), &vk)?;

        let emb_cpu = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let emb_vk = vk_model.embed_tokens.forward(&ids_vk)?;
        assert_tensor_close("embedding", &emb_vk, &emb_cpu, 1e-4)?;

        let ln1_cpu = cpu_model.layers[0].ln1.forward(&emb_cpu)?;
        let ln1_vk = vk_model.layers[0].ln1.forward(&emb_vk)?;
        assert_tensor_close("ln1", &ln1_vk, &ln1_cpu, 1e-4)?;

        let q_proj_cpu = cpu_model.layers[0].self_attn.q_proj.forward(&ln1_cpu)?;
        let q_proj_vk = vk_model.layers[0].self_attn.q_proj.forward(&ln1_vk)?;
        log_qproj_rows("q_proj_cpu", &q_proj_cpu)?;
        log_qproj_rows("q_proj_vk", &q_proj_vk)?;
        assert_tensor_close("q_proj", &q_proj_vk, &q_proj_cpu, 3e-2)?;

        let k_proj_cpu = cpu_model.layers[0].self_attn.k_proj.forward(&ln1_cpu)?;
        let k_proj_vk = vk_model.layers[0].self_attn.k_proj.forward(&ln1_vk)?;
        assert_tensor_close("k_proj", &k_proj_vk, &k_proj_cpu, 3e-2)?;

        let v_proj_cpu = cpu_model.layers[0].self_attn.v_proj.forward(&ln1_cpu)?;
        let v_proj_vk = vk_model.layers[0].self_attn.v_proj.forward(&ln1_vk)?;
        assert_tensor_close("v_proj", &v_proj_vk, &v_proj_cpu, 3e-2)?;

        let mask_cpu = cpu_model.causal_mask(1, ids.len(), 0, None)?;
        let mask_vk = vk_model.causal_mask(1, ids.len(), 0, None)?;
        let attn_cpu = &mut cpu_model.layers[0].self_attn;
        let attn_vk = &mut vk_model.layers[0].self_attn;
        let (b, l, _) = ln1_cpu.dims3()?;

        let q_cpu = q_proj_cpu
            .reshape((b, l, attn_cpu.num_heads, attn_cpu.head_dim))?
            .transpose(1, 2)?;
        let q_vk = q_proj_vk
            .reshape((b, l, attn_vk.num_heads, attn_vk.head_dim))?
            .transpose(1, 2)?;
        assert_tensor_close("q_reshape", &q_vk, &q_cpu, 3e-2)?;

        let k_cpu = k_proj_cpu
            .reshape((b, l, attn_cpu.num_kv_heads, attn_cpu.head_dim))?
            .transpose(1, 2)?;
        let k_vk = k_proj_vk
            .reshape((b, l, attn_vk.num_kv_heads, attn_vk.head_dim))?
            .transpose(1, 2)?;
        assert_tensor_close("k_reshape", &k_vk, &k_cpu, 3e-2)?;

        let v_cpu = v_proj_cpu
            .reshape((b, l, attn_cpu.num_kv_heads, attn_cpu.head_dim))?
            .transpose(1, 2)?;
        let v_vk = v_proj_vk
            .reshape((b, l, attn_vk.num_kv_heads, attn_vk.head_dim))?
            .transpose(1, 2)?;
        assert_tensor_close("v_reshape", &v_vk, &v_cpu, 3e-2)?;

        let q_flat_cpu = attn_cpu.q_norm.forward(&q_cpu.flatten(0, 2)?)?;
        let q_flat_vk = attn_vk.q_norm.forward(&q_vk.flatten(0, 2)?)?;
        assert_tensor_close("q_norm", &q_flat_vk, &q_flat_cpu, 3e-2)?;
        let q_cpu = q_flat_cpu.reshape((b, attn_cpu.num_heads, l, attn_cpu.head_dim))?;
        let q_vk = q_flat_vk.reshape((b, attn_vk.num_heads, l, attn_vk.head_dim))?;

        let k_flat_cpu = attn_cpu.k_norm.forward(&k_cpu.flatten(0, 2)?)?;
        let k_flat_vk = attn_vk.k_norm.forward(&k_vk.flatten(0, 2)?)?;
        assert_tensor_close("k_norm", &k_flat_vk, &k_flat_cpu, 3e-2)?;
        let k_cpu = k_flat_cpu.reshape((b, attn_cpu.num_kv_heads, l, attn_cpu.head_dim))?;
        let k_vk = k_flat_vk.reshape((b, attn_vk.num_kv_heads, l, attn_vk.head_dim))?;

        let (q_cpu, k_cpu) = attn_cpu.rotary_emb.apply(&q_cpu, &k_cpu, 0)?;
        let (q_vk, k_vk) = attn_vk.rotary_emb.apply(&q_vk, &k_vk, 0)?;
        assert_tensor_close("q_rope", &q_vk, &q_cpu, 3e-2)?;
        assert_tensor_close("k_rope", &k_vk, &k_cpu, 3e-2)?;

        let k_cpu = repeat_kv(k_cpu, attn_cpu.num_kv_groups)?.contiguous()?;
        let k_vk = repeat_kv(k_vk, attn_vk.num_kv_groups)?.contiguous()?;
        let v_cpu = repeat_kv(v_cpu, attn_cpu.num_kv_groups)?.contiguous()?;
        let v_vk = repeat_kv(v_vk, attn_vk.num_kv_groups)?.contiguous()?;
        assert_tensor_close("k_repeat", &k_vk, &k_cpu, 3e-2)?;
        assert_tensor_close("v_repeat", &v_vk, &v_cpu, 3e-2)?;

        let scale = 1.0 / (attn_cpu.head_dim as f64).sqrt();
        let scores_cpu = (q_cpu.matmul(&k_cpu.transpose(2, 3)?)? * scale)?;
        let mask_cpu = if mask_cpu.dtype() != scores_cpu.dtype() {
            mask_cpu.to_dtype(scores_cpu.dtype())?
        } else {
            mask_cpu.clone()
        };
        let scores_cpu = scores_cpu.broadcast_add(&mask_cpu)?;
        let scores_vk = (q_vk.matmul(&k_vk.transpose(2, 3)?)? * scale)?;
        let mask_vk = if mask_vk.dtype() != scores_vk.dtype() {
            mask_vk.to_dtype(scores_vk.dtype())?
        } else {
            mask_vk.clone()
        };
        let scores_vk = scores_vk.broadcast_add(&mask_vk)?;
        assert_tensor_close("scores", &scores_vk, &scores_cpu, 5e-2)?;

        let probs_cpu = candle_nn::ops::softmax_last_dim(&scores_cpu)?;
        let probs_vk = candle_nn::ops::softmax_last_dim(&scores_vk)?;
        assert_tensor_close("probs", &probs_vk, &probs_cpu, 5e-2)?;

        let ctx_cpu = probs_cpu.matmul(&v_cpu)?;
        let ctx_vk = probs_vk.matmul(&v_vk)?;
        assert_tensor_close("ctx", &ctx_vk, &ctx_cpu, 5e-2)?;

        let reshaped_ctx_cpu = ctx_cpu
            .transpose(1, 2)?
            .reshape((b, l, attn_cpu.hidden_size))?;
        let reshaped_ctx_vk = ctx_vk
            .transpose(1, 2)?
            .reshape((b, l, attn_vk.hidden_size))?;
        let attn_out_cpu = attn_cpu.o_proj.forward(&reshaped_ctx_cpu)?;
        let attn_out_vk = attn_vk.o_proj.forward(&reshaped_ctx_vk)?;
        assert_tensor_close("attn_out", &attn_out_vk, &attn_out_cpu, 5e-2)?;

        let _ = attn_cpu;
        let _ = attn_vk;
        log_layout("emb_vk", &emb_vk);
        log_layout("attn_out_vk", &attn_out_vk);
        let zero_vk = Tensor::zeros_like(&emb_vk)?;
        let emb_plus_zero_vk = (&emb_vk + &zero_vk)?;
        let attn_plus_zero_vk = (&attn_out_vk + &zero_vk)?;
        assert_tensor_close("emb_plus_zero", &emb_plus_zero_vk, &emb_vk, 1e-4)?;
        assert_tensor_close("attn_plus_zero", &attn_plus_zero_vk, &attn_out_vk, 1e-4)?;
        let post_attn_cpu = (&emb_cpu + &attn_out_cpu)?;
        let post_attn_vk = (&emb_vk + &attn_out_vk)?;
        assert_tensor_close("post_attn", &post_attn_vk, &post_attn_cpu, 5e-2)?;

        let ln2_cpu = cpu_model.layers[0].ln2.forward(&post_attn_cpu)?;
        let ln2_vk = vk_model.layers[0].ln2.forward(&post_attn_vk)?;
        assert_tensor_close("ln2", &ln2_vk, &ln2_cpu, 5e-2)?;

        let gate_cpu = cpu_model.layers[0].mlp.gate_proj.forward(&ln2_cpu)?;
        let gate_vk = vk_model.layers[0].mlp.gate_proj.forward(&ln2_vk)?;
        assert_tensor_close("mlp_gate", &gate_vk, &gate_cpu, 5e-2)?;

        let up_cpu = cpu_model.layers[0].mlp.up_proj.forward(&ln2_cpu)?;
        let up_vk = vk_model.layers[0].mlp.up_proj.forward(&ln2_vk)?;
        assert_tensor_close("mlp_up", &up_vk, &up_cpu, 5e-2)?;

        let gate_act_cpu = gate_cpu.apply(&cpu_model.layers[0].mlp.act_fn)?;
        let gate_act_vk = gate_vk.apply(&vk_model.layers[0].mlp.act_fn)?;
        assert_tensor_close("mlp_gate_act", &gate_act_vk, &gate_act_cpu, 5e-2)?;

        let gated_cpu = (&gate_act_cpu * &up_cpu)?;
        let gated_vk = (&gate_act_vk * &up_vk)?;
        assert_tensor_close("mlp_gated", &gated_vk, &gated_cpu, 5e-2)?;

        let mlp_out_cpu = cpu_model.layers[0].mlp.down_proj.forward(&gated_cpu)?;
        let mlp_out_vk = vk_model.layers[0].mlp.down_proj.forward(&gated_vk)?;
        assert_tensor_close("mlp_down", &mlp_out_vk, &mlp_out_cpu, 5e-2)?;

        let layer0_cpu = cpu_model.layers[0].forward(&emb_cpu, Some(&mask_cpu), 0)?;
        let layer0_vk = vk_model.layers[0].forward(&emb_vk, Some(&mask_vk), 0)?;
        assert_tensor_close("layer0", &layer0_vk, &layer0_cpu, 5e-2)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires CANDLE_QWEN3_GGUF_PATH and a usable Vulkan device"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_forward_parity() -> Result<()> {
        let Some(path) = qwen3_gguf_path() else {
            return Ok(());
        };
        log_gguf_tensor_dtype(&path, "blk.1.attn_q.weight")?;
        log_gguf_tensor_dtype(&path, "blk.1.ffn_up.weight")?;
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut vk_model = load_model(&path, &vk)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_vk = Tensor::from_slice(&ids, (1, ids.len()), &vk)?;
        let mut h_cpu = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let mut h_vk = vk_model.embed_tokens.forward(&ids_vk)?;
        let mask_cpu = cpu_model.causal_mask(1, ids.len(), 0, None)?;
        let mask_vk = vk_model.causal_mask(1, ids.len(), 0, None)?;
        for (layer_idx, (cpu_layer, vk_layer)) in cpu_model
            .layers
            .iter_mut()
            .zip(vk_model.layers.iter_mut())
            .enumerate()
        {
            h_cpu = cpu_layer.forward(&h_cpu, Some(&mask_cpu), 0)?;
            h_vk = vk_layer.forward(&h_vk, Some(&mask_vk), 0)?;
            assert_tensor_close(&format!("prefill_layer_{layer_idx}"), &h_vk, &h_cpu, 5e-2)?;
        }
        let h_cpu = cpu_model.norm.forward(&h_cpu)?;
        let h_vk = vk_model.norm.forward(&h_vk)?;
        assert_tensor_close("prefill_norm", &h_vk, &h_cpu, 5e-2)?;
        let last_cpu = h_cpu.narrow(1, ids.len() - 1, 1)?;
        let last_vk = h_vk.narrow(1, ids.len() - 1, 1)?;
        assert_tensor_close("prefill_last_hidden", &last_vk, &last_cpu, 5e-2)?;
        let prefill_cpu = cpu_model.lm_head.forward(&last_cpu)?.squeeze(1)?;
        let prefill_vk = vk_model.lm_head.forward(&last_vk)?.squeeze(1)?;
        assert_tensor_close("prefill_logits", &prefill_vk, &prefill_cpu, 5e-2)?;

        let next_cpu = Tensor::from_slice(&[5u32], (1, 1), &cpu)?;
        let next_vk = Tensor::from_slice(&[5u32], (1, 1), &vk)?;
        let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
        let decode_vk = vk_model.forward(&next_vk, ids.len())?;
        assert_tensor_close("decode_logits", &decode_vk, &decode_cpu, 5e-2)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires CANDLE_QWEN3_GGUF_PATH and a usable Vulkan device"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_layer1_topology_parity() -> Result<()> {
        let Some(path) = qwen3_gguf_path() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut vk_model = load_model(&path, &vk)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_vk = Tensor::from_slice(&ids, (1, ids.len()), &vk)?;
        let mask_cpu = cpu_model.causal_mask(1, ids.len(), 0, None)?;
        let mask_vk = vk_model.causal_mask(1, ids.len(), 0, None)?;

        let layer0_cpu_in = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let layer0_vk_in = vk_model.embed_tokens.forward(&ids_vk)?;
        let layer0_cpu = cpu_model.layers[0].forward(&layer0_cpu_in, Some(&mask_cpu), 0)?;
        let layer0_vk = vk_model.layers[0].forward(&layer0_vk_in, Some(&mask_vk), 0)?;
        assert_tensor_close("layer0_input_to_layer1", &layer0_vk, &layer0_cpu, 5e-2)?;

        let ln1_cpu = vk_model.layers.len(); // keep borrow scopes simple below
        let _ = ln1_cpu;
        let l1_ln1_cpu = cpu_model.layers[1].ln1.forward(&layer0_cpu)?;
        let l1_ln1_vk = vk_model.layers[1].ln1.forward(&layer0_vk)?;
        assert_tensor_close("layer1_ln1", &l1_ln1_vk, &l1_ln1_cpu, 5e-2)?;

        let l1_attn_cpu = cpu_model.layers[1]
            .self_attn
            .forward(&l1_ln1_cpu, Some(&mask_cpu), 0)?;
        let l1_attn_vk = vk_model.layers[1]
            .self_attn
            .forward(&l1_ln1_vk, Some(&mask_vk), 0)?;
        assert_tensor_close("layer1_attn", &l1_attn_vk, &l1_attn_cpu, 5e-2)?;

        let l1_post_attn_cpu = (&layer0_cpu + &l1_attn_cpu)?;
        let l1_post_attn_vk = (&layer0_vk + &l1_attn_vk)?;
        assert_tensor_close(
            "layer1_post_attn",
            &l1_post_attn_vk,
            &l1_post_attn_cpu,
            5e-2,
        )?;

        let l1_ln2_cpu = cpu_model.layers[1].ln2.forward(&l1_post_attn_cpu)?;
        let l1_ln2_vk = vk_model.layers[1].ln2.forward(&l1_post_attn_vk)?;
        assert_tensor_close("layer1_ln2", &l1_ln2_vk, &l1_ln2_cpu, 5e-2)?;

        let l1_gate_cpu = cpu_model.layers[1].mlp.gate_proj.forward(&l1_ln2_cpu)?;
        let l1_gate_vk = vk_model.layers[1].mlp.gate_proj.forward(&l1_ln2_vk)?;
        assert_tensor_close("layer1_gate", &l1_gate_vk, &l1_gate_cpu, 5e-2)?;

        let l1_up_cpu = cpu_model.layers[1].mlp.up_proj.forward(&l1_ln2_cpu)?;
        let l1_up_vk = vk_model.layers[1].mlp.up_proj.forward(&l1_ln2_vk)?;
        let l1_ln2_cpu_values = l1_ln2_cpu
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let l1_ln2_cpu_on_vk =
            Tensor::from_vec(l1_ln2_cpu_values, l1_ln2_cpu.shape().clone(), &vk)?;
        let l1_up_vk_exact_input = vk_model.layers[1].mlp.up_proj.forward(&l1_ln2_cpu_on_vk)?;
        assert_tensor_close(
            "layer1_up_exact_input",
            &l1_up_vk_exact_input,
            &l1_up_cpu,
            5e-2,
        )?;
        assert_tensor_close("layer1_up", &l1_up_vk, &l1_up_cpu, 5e-2)?;

        let l1_gate_act_cpu = l1_gate_cpu.apply(&cpu_model.layers[1].mlp.act_fn)?;
        let l1_gate_act_vk = l1_gate_vk.apply(&vk_model.layers[1].mlp.act_fn)?;
        assert_tensor_close("layer1_gate_act", &l1_gate_act_vk, &l1_gate_act_cpu, 5e-2)?;

        let l1_gated_cpu = (&l1_gate_act_cpu * &l1_up_cpu)?;
        let l1_gated_vk = (&l1_gate_act_vk * &l1_up_vk)?;
        assert_tensor_close("layer1_gated", &l1_gated_vk, &l1_gated_cpu, 5e-2)?;

        let l1_mlp_cpu = cpu_model.layers[1].mlp.down_proj.forward(&l1_gated_cpu)?;
        let l1_mlp_vk = vk_model.layers[1].mlp.down_proj.forward(&l1_gated_vk)?;
        assert_tensor_close("layer1_mlp", &l1_mlp_vk, &l1_mlp_cpu, 5e-2)?;

        let l1_out_cpu = (&l1_post_attn_cpu + &l1_mlp_cpu)?;
        let l1_out_vk = (&l1_post_attn_vk + &l1_mlp_vk)?;
        assert_tensor_close("layer1_out", &l1_out_vk, &l1_out_cpu, 5e-2)?;
        Ok(())
    }
}
