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
use candle_nn::kv_cache::{ConcatKvCache, GrowableKvCache, InterleavedKvCache, RawInterleavedKvCache};
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
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Zero-allocation cos/sin slices for a single position.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

/// Union KV-cache type selecting between the amortized O(1)-per-token
/// `GrowableKvCache` (used on Vulkan/WGPU, where the O(N²) re-copy of
/// `ConcatKvCache::append` dominates prefill) and the original `ConcatKvCache`
/// (used untouched on CPU/CUDA/Metal). Both implement the same `append`/`reset`
/// contract; only the dispatch differs, so the CUDA/Metal path is unchanged.
#[derive(Debug, Clone)]
enum AttentionKvCache {
    Concat(ConcatKvCache),
    Growable(GrowableKvCache),
}

impl AttentionKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Concat(c) => c.append(k, v),
            Self::Growable(c) => c.append(k, v),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Concat(c) => c.reset(),
            Self::Growable(c) => c.reset(),
        }
    }
}

/// Device-gated EXPANDED (num_heads-wide) KV cache used on Vulkan to avoid the
/// O(N^2) whole-cache re-expansion of GQA every decode step.
///
/// The model stores an 8-head (num_kv_heads) cache and `repeat_kv`s it to the
/// 16-head (num_heads) width inside every forward so the score/ctx matmuls can
/// broadcast each KV head to its group of query heads. That re-expansion re-reads
/// and re-writes the ENTIRE cache every token (~2.8 GB/token at 4096 ctx), which is
/// the dominant pp4096 residual. This cache keeps the EXPANDED 16-head K/V buffers
/// persistently and only `repeat_kv`s the NEW tail (the tokens being appended this
/// step) before an in-place `GrowableKvCache` append, so the incremental per-token
/// cost is O(seq_len_this_step) instead of O(total_len).
///
/// The expansion is a pure data duplication with the same head ordering as the CPU
/// path's `repeat_kv`, so the attention math (scores, softmax, ctx) is bit-identical
/// to the existing `repeat_kv` + matmul path — only the materialization is avoided.
#[derive(Debug, Clone)]
struct ExpandedKvCache {
    inner: GrowableKvCache,
    num_kv_groups: usize,
}

impl ExpandedKvCache {
    fn new(num_kv_groups: usize) -> Self {
        Self {
            inner: GrowableKvCache::new(2),
            num_kv_groups,
        }
    }

    /// Expand the just-arrived `(k, v)` (num_kv_heads wide) to num_heads and append
    /// in-place, returning the full (num_heads-wide) live-prefix views (`narrow`ed to
    /// the used length, strided by the backing-buffer capacity). The returned views
    /// alias the cache and are only valid until the next grow, matching the
    /// `GrowableKvCache` contract (consumed within the same forward).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;
        self.inner.append(&k, &v)
    }

    fn reset(&mut self) {
        self.inner.reset();
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
    kv_cache: Option<AttentionKvCache>,
    expanded_cache: Option<ExpandedKvCache>,
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
        // GPU: standard KV-cache path. Vulkan uses the in-place GROWABLE EXPANDED
        // (num_heads-wide) cache so the O(N^2) whole-cache GQA re-expansion on every
        // decode step is replaced by an O(1)-per-token tail append; wgpu keeps the
        // in-place 8-head growable cache + repeat_kv (B15 owns wgpu), and CUDA/Metal
        // keep the original ConcatKvCache + repeat_kv path byte-for-byte.
        let on_cpu = device.is_cpu();
        let (kv_cache, expanded_cache) = if on_cpu {
            (None, None)
        } else if device.is_vulkan() {
            (None, Some(ExpandedKvCache::new(num_kv_groups)))
        } else if device.is_wgpu() {
            (
                Some(AttentionKvCache::Growable(GrowableKvCache::new(2))),
                None,
            )
        } else {
            (Some(AttentionKvCache::Concat(ConcatKvCache::new(2))), None)
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
            expanded_cache,
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
            // Standard matmul attention (no flash). On Vulkan the expanded cache
            // returns the num_heads-wide (16-head) K/V live-prefix views already, so
            // the whole-cache `repeat_kv` materialization is skipped (the pp4096
            // residual); CUDA/Metal/wgpu keep the existing `repeat_kv` path.
            let (k, v) = if let Some(ec) = &mut self.expanded_cache {
                ec.append(&k, &v)?
            } else {
                let (k, v) = self.kv_cache.as_mut().unwrap().append(&k, &v)?;
                let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
                (k, v)
            };

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
        if let Some(c) = &mut self.expanded_cache {
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
        let x2 = (x + h)?;
        let h2 = self.ln2.forward(&x2)?;
        let h2 = h2.apply(&self.mlp)?;
        x2 + h2
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

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Skip mask materialization when using CPU flash attention
        let causal_mask = if l == 1 || self.device.is_cpu() {
            None
        } else {
            Some(crate::utils::build_additive_causal_mask(
                l,
                offset,
                None,
                &self.device,
                self.dtype,
            )?)
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
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use std::path::{Path, PathBuf};

    fn qwen3_gguf_path() -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("CANDLE_QWEN3_GGUF_PATH") {
            return Ok(PathBuf::from(path));
        }
        let api = Api::new()
            .map_err(|err| candle::Error::msg(format!("failed to create hf-hub client: {err}")))?;
        let repo = Repo::with_revision(
            "unsloth/Qwen3-0.6B-GGUF".to_string(),
            RepoType::Model,
            "main".to_string(),
        );
        api.repo(repo).get("Qwen3-0.6B-Q4_K_M.gguf").map_err(|err| {
            candle::Error::msg(format!("failed to download Qwen3 GGUF from hf-hub: {err}"))
        })
    }

    fn load_model(path: &Path, device: &Device) -> Result<ModelWeights> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        ModelWeights::from_gguf(content, &mut file, device)
    }

    fn reset_gpu_fallback_count(device: &Device) {
        if device.is_wgpu() {
            candle::reset_wgpu_cpu_fallback_count();
        } else if device.is_vulkan() {
            candle::reset_vulkan_cpu_fallback_count();
        }
    }

    fn assert_no_gpu_fallbacks(label: &str, device: &Device) -> Result<()> {
        if !device.is_wgpu() && !device.is_vulkan() {
            return Ok(());
        }
        device.synchronize()?;
        let count = if device.is_wgpu() {
            candle::wgpu_cpu_fallback_count()
        } else {
            candle::vulkan_cpu_fallback_count()
        };
        assert_eq!(count, 0, "{label}: unexpected CPU fallback count {count}");
        Ok(())
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

    fn assert_tensor_close_with_nmse(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        tol: f32,
        nmse_tol: f64,
    ) -> Result<()> {
        let (max_idx, max_diff, max_rel, nmse) = tensor_diff_stats(actual, expected)?;
        println!(
            "{label}: max_idx={max_idx} max_diff={max_diff:.6} max_rel={max_rel:.6} nmse={nmse:.6e}"
        );
        assert!(
            max_rel <= tol || nmse <= nmse_tol,
            "{label} mismatch: max_idx={max_idx} max_diff={max_diff} max_rel={max_rel} nmse={nmse}"
        );
        Ok(())
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        tol: f32,
    ) -> Result<()> {
        assert_tensor_close_with_nmse(label, actual, expected, tol, 5e-4)
    }

    fn assert_logits_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        tol: f32,
    ) -> Result<()> {
        let (max_idx, max_diff, max_rel, nmse) = tensor_diff_stats(actual, expected)?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let cosine = {
            let mut dot = 0f64;
            let mut actual_norm = 0f64;
            let mut expected_norm = 0f64;
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                dot += (*actual as f64) * (*expected as f64);
                actual_norm += (*actual as f64) * (*actual as f64);
                expected_norm += (*expected as f64) * (*expected as f64);
            }
            if actual_norm > 0.0 && expected_norm > 0.0 {
                dot / (actual_norm.sqrt() * expected_norm.sqrt())
            } else {
                1.0
            }
        };
        let actual_argmax = argmax_index(&actual);
        let expected_argmax = argmax_index(&expected);
        let top5_overlap = topk_overlap(&actual, &expected, 5);
        println!(
            "{label}: max_idx={max_idx} max_diff={max_diff:.6} max_rel={max_rel:.6} nmse={nmse:.6e} cosine={cosine:.6} argmax_actual={actual_argmax} argmax_expected={expected_argmax} top5_overlap={top5_overlap}"
        );
        assert!(
            max_rel <= tol
                || (nmse <= 1e-2
                    && cosine >= 0.995
                    && actual_argmax == expected_argmax
                    && top5_overlap >= 4),
            "{label} mismatch: max_idx={max_idx} max_diff={max_diff} max_rel={max_rel} nmse={nmse} cosine={cosine} argmax_actual={actual_argmax} argmax_expected={expected_argmax} top5_overlap={top5_overlap}"
        );
        Ok(())
    }

    fn argmax_index(values: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (idx, &value) in values.iter().enumerate() {
            if value > best {
                best = value;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn topk_overlap(actual: &[f32], expected: &[f32], k: usize) -> usize {
        fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
            let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
        }

        let actual_topk = topk_indices(actual, k);
        let expected_topk = topk_indices(expected, k);
        actual_topk
            .iter()
            .filter(|idx| expected_topk.contains(idx))
            .count()
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn log_qproj_rows(label: &str, tensor: &Tensor) -> Result<()> {
        let tensor = tensor.to_dtype(DType::F32)?;
        match tensor.dims() {
            [b, l, h] if *b == 1 && *l <= 4 && *h >= 4 => {
                let rows = tensor.to_vec3::<f32>()?;
                for (row, row_vals) in rows[0].iter().take(*l).enumerate() {
                    println!(
                        "{label}[row={row}]: {:.6} {:.6} {:.6} {:.6}",
                        row_vals[0], row_vals[1], row_vals[2], row_vals[3],
                    );
                }
            }
            other => println!("{label}: preview skipped for dims={other:?}"),
        }
        Ok(())
    }

    #[allow(dead_code)]
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
    #[ignore = "requires a usable wgpu adapter and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "wgpu")]
    fn qwen3_local_wgpu_stage_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        let cpu = Device::Cpu;
        let wgpu = Device::new_wgpu(0)?;
        reset_gpu_fallback_count(&wgpu);
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

        let mask_cpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &cpu,
            cpu_model.dtype,
        )?;
        let mask_wgpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &wgpu,
            wgpu_model.dtype,
        )?;
        let layer0_cpu = cpu_model.layers[0].forward(&emb_cpu, Some(&mask_cpu), 0)?;
        let layer0_wgpu = wgpu_model.layers[0].forward(&emb_wgpu, Some(&mask_wgpu), 0)?;
        assert_tensor_close("layer0", &layer0_wgpu, &layer0_cpu, 5e-2)?;
        assert_no_gpu_fallbacks("qwen3_local_wgpu_stage_parity", &wgpu)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a usable wgpu adapter and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "wgpu")]
    fn qwen3_local_wgpu_forward_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        let cpu = Device::Cpu;
        let wgpu = Device::new_wgpu(0)?;
        reset_gpu_fallback_count(&wgpu);
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut wgpu_model = load_model(&path, &wgpu)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_wgpu = Tensor::from_slice(&ids, (1, ids.len()), &wgpu)?;
        let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
        let prefill_wgpu = wgpu_model.forward(&ids_wgpu, 0)?;
        assert_logits_close("prefill_logits", &prefill_wgpu, &prefill_cpu, 5e-2)?;

        let next_cpu = Tensor::from_slice(&[5u32], (1, 1), &cpu)?;
        let next_wgpu = Tensor::from_slice(&[5u32], (1, 1), &wgpu)?;
        let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
        let decode_wgpu = wgpu_model.forward(&next_wgpu, ids.len())?;
        assert_logits_close("decode_logits", &decode_wgpu, &decode_cpu, 5e-2)?;
        assert_no_gpu_fallbacks("qwen3_local_wgpu_forward_parity", &wgpu)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a usable wgpu adapter and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "wgpu")]
    fn qwen3_local_wgpu_layer1_topology_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        let cpu = Device::Cpu;
        let wgpu = Device::new_wgpu(0)?;
        reset_gpu_fallback_count(&wgpu);
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut wgpu_model = load_model(&path, &wgpu)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_wgpu = Tensor::from_slice(&ids, (1, ids.len()), &wgpu)?;
        let mask_cpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &cpu,
            cpu_model.dtype,
        )?;
        let mask_wgpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &wgpu,
            wgpu_model.dtype,
        )?;

        let layer0_cpu_in = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let layer0_wgpu_in = wgpu_model.embed_tokens.forward(&ids_wgpu)?;
        let layer0_cpu = cpu_model.layers[0].forward(&layer0_cpu_in, Some(&mask_cpu), 0)?;
        let layer0_wgpu = wgpu_model.layers[0].forward(&layer0_wgpu_in, Some(&mask_wgpu), 0)?;
        assert_tensor_close_with_nmse(
            "layer0_input_to_layer1",
            &layer0_wgpu,
            &layer0_cpu,
            5e-2,
            5e-3,
        )?;

        let l1_ln1_cpu = cpu_model.layers[1].ln1.forward(&layer0_cpu)?;
        let l1_ln1_wgpu = wgpu_model.layers[1].ln1.forward(&layer0_wgpu)?;
        assert_tensor_close_with_nmse("layer1_ln1", &l1_ln1_wgpu, &l1_ln1_cpu, 5e-2, 5e-3)?;

        let l1_attn_cpu = cpu_model.layers[1]
            .self_attn
            .forward(&l1_ln1_cpu, Some(&mask_cpu), 0)?;
        let l1_attn_wgpu =
            wgpu_model.layers[1]
                .self_attn
                .forward(&l1_ln1_wgpu, Some(&mask_wgpu), 0)?;
        assert_tensor_close_with_nmse("layer1_attn", &l1_attn_wgpu, &l1_attn_cpu, 5e-2, 5e-3)?;

        let l1_post_attn_cpu = (&layer0_cpu + &l1_attn_cpu)?;
        let l1_post_attn_wgpu = (&layer0_wgpu + &l1_attn_wgpu)?;
        assert_tensor_close_with_nmse(
            "layer1_post_attn",
            &l1_post_attn_wgpu,
            &l1_post_attn_cpu,
            5e-2,
            5e-3,
        )?;

        let l1_ln2_cpu = cpu_model.layers[1].ln2.forward(&l1_post_attn_cpu)?;
        let l1_ln2_wgpu = wgpu_model.layers[1].ln2.forward(&l1_post_attn_wgpu)?;
        assert_tensor_close_with_nmse("layer1_ln2", &l1_ln2_wgpu, &l1_ln2_cpu, 5e-2, 5e-3)?;

        let l1_gate_cpu = cpu_model.layers[1].mlp.gate_proj.forward(&l1_ln2_cpu)?;
        let l1_gate_wgpu = wgpu_model.layers[1].mlp.gate_proj.forward(&l1_ln2_wgpu)?;
        assert_tensor_close_with_nmse("layer1_gate", &l1_gate_wgpu, &l1_gate_cpu, 5e-2, 5e-3)?;

        let l1_up_cpu = cpu_model.layers[1].mlp.up_proj.forward(&l1_ln2_cpu)?;
        let l1_up_wgpu = wgpu_model.layers[1].mlp.up_proj.forward(&l1_ln2_wgpu)?;
        assert_tensor_close_with_nmse("layer1_up", &l1_up_wgpu, &l1_up_cpu, 5e-2, 5e-3)?;

        let l1_gate_act_cpu = l1_gate_cpu.apply(&cpu_model.layers[1].mlp.act_fn)?;
        let l1_gate_act_wgpu = l1_gate_wgpu.apply(&wgpu_model.layers[1].mlp.act_fn)?;
        assert_tensor_close_with_nmse(
            "layer1_gate_act",
            &l1_gate_act_wgpu,
            &l1_gate_act_cpu,
            5e-2,
            5e-3,
        )?;

        let l1_gated_cpu = (&l1_gate_act_cpu * &l1_up_cpu)?;
        let l1_gated_wgpu = (&l1_gate_act_wgpu * &l1_up_wgpu)?;
        assert_tensor_close_with_nmse("layer1_gated", &l1_gated_wgpu, &l1_gated_cpu, 5e-2, 5e-3)?;

        let l1_mlp_cpu = cpu_model.layers[1].mlp.down_proj.forward(&l1_gated_cpu)?;
        let l1_mlp_wgpu = wgpu_model.layers[1].mlp.down_proj.forward(&l1_gated_wgpu)?;
        assert_tensor_close_with_nmse("layer1_mlp", &l1_mlp_wgpu, &l1_mlp_cpu, 5e-2, 5e-3)?;

        let l1_out_cpu = (&l1_post_attn_cpu + &l1_mlp_cpu)?;
        let l1_out_wgpu = (&l1_post_attn_wgpu + &l1_mlp_wgpu)?;
        assert_tensor_close_with_nmse("layer1_out", &l1_out_wgpu, &l1_out_cpu, 5e-2, 5e-3)?;
        assert_no_gpu_fallbacks("qwen3_local_wgpu_layer1_topology_parity", &wgpu)?;
        Ok(())
    }

    fn diff_stats_ab(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len(), "{label}: len {} vs {}", a.len(), b.len());
        let mut max_abs = 0f32;
        let mut sum_err = 0f64;
        let mut sum_abs = 0f64;
        let mut sum_sq = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let err = (*x as f64) - (*y as f64);
            max_abs = max_abs.max(err.abs() as f32);
            sum_err += err;
            sum_abs += err.abs();
            sum_sq += err * err;
        }
        let n = a.len() as f64;
        println!(
            "{label}: n={} max_abs={max_abs:.6} rmse={:.8} mean_err={:.6e} mean_abs={:.6e}",
            a.len(),
            (sum_sq / n).sqrt(),
            sum_err / n,
            sum_abs / n
        );
        Ok(())
    }

    // Op-level wgpu-vs-cuda trace of layer 0 + per-layer hidden-state chain.
    // Compares wgpu and cuda on IDENTICAL inputs at every op inside the first
    // transformer layer, tracking max_abs / rmse / MEAN(error) — a nonzero
    // mean(error) at a single op is the systematic bias that compounds through
    // the residual stream across all 24 layers.
    #[test]
    #[ignore = "requires cuda + wgpu adapters and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH"]
    #[cfg(all(feature = "cuda", feature = "wgpu"))]
    fn qwen3_wgpu_cuda_layer0_op_trace() -> Result<()> {
        let path = qwen3_gguf_path()?;
        let cuda = Device::new_cuda(0)?;
        let wgpu = Device::new_wgpu(0)?;
        reset_gpu_fallback_count(&wgpu);
        let mut cuda_model = load_model(&path, &cuda)?;
        let mut wgpu_model = load_model(&path, &wgpu)?;

        // Prefer the REAL generation prompt (tokenizer.json next to the model);
        // fall back to synthetic ids when only the GGUF is available.
        let tok_path = path
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .filter(|p| p.exists());
        let ids: Vec<u32> = if let Some(tp) = tok_path {
            let tok = tokenizers::Tokenizer::from_file(&tp).map_err(|e| {
                candle::Error::msg(format!("failed to load tokenizer: {e}"))
            })?;
            let prompt = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n";
            let enc = tok
                .encode(prompt, false)
                .map_err(|e| candle::Error::msg(format!("tokenize failed: {e}")))?;
            enc.get_ids().to_vec()
        } else {
            vec![1u32, 2, 3, 4]
        };
        let ids_cuda = Tensor::from_slice(&ids, (1, ids.len()), &cuda)?;
        let ids_wgpu = Tensor::from_slice(&ids, (1, ids.len()), &wgpu)?;

        let emb_cuda = cuda_model.embed_tokens.forward(&ids_cuda)?;
        let emb_wgpu = wgpu_model.embed_tokens.forward(&ids_wgpu)?;
        diff_stats_ab("L0 embed", &emb_wgpu, &emb_cuda)?;

        let mask_cuda = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &cuda,
            cuda_model.dtype,
        )?;
        let mask_wgpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &wgpu,
            wgpu_model.dtype,
        )?;
        diff_stats_ab("L0 mask", &mask_wgpu, &mask_cuda)?;

        // ---- replicate LayerWeights::forward op-by-op ----
        let ac = &mut cuda_model.layers[0];
        let aw = &mut wgpu_model.layers[0];

        let xc = emb_cuda;
        let xw = emb_wgpu;

        let ln1_c = ac.ln1.forward(&xc)?;
        let ln1_w = aw.ln1.forward(&xw)?;
        diff_stats_ab("L0 ln1", &ln1_w, &ln1_c)?;

        let (b, l, _) = xc.dims3()?;

        let q_c = ac.self_attn.q_proj.forward(&ln1_c)?;
        let q_w = aw.self_attn.q_proj.forward(&ln1_w)?;
        diff_stats_ab("L0 q_proj", &q_w, &q_c)?;

        let k_c = ac.self_attn.k_proj.forward(&ln1_c)?;
        let k_w = aw.self_attn.k_proj.forward(&ln1_w)?;
        diff_stats_ab("L0 k_proj", &k_w, &k_c)?;

        // CPU reference for the projections: candle-cuda uses a fast-mmq kernel
        // that quantizes the activation to q8_1 before the dot; candle-cpu uses
        // the same q8_1 activation contract. Comparing all three triangulates
        // which side carries the systematic mean error.
        let cpu = Device::Cpu;
        let cpu_model = load_model(&path, &cpu)?;
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let emb_cpu2 = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let ln1_cpu = cpu_model.layers[0].ln1.forward(&emb_cpu2)?;
        let q_cpu = cpu_model.layers[0].self_attn.q_proj.forward(&ln1_cpu)?;
        diff_stats_ab("L0 q_proj wgpu-vs-cpu", &q_w, &q_cpu.to_device(&wgpu)?)?;
        diff_stats_ab("L0 q_proj cuda-vs-cpu", &q_c, &q_cpu.to_device(&cuda)?)?;
        let k_cpu = cpu_model.layers[0].self_attn.k_proj.forward(&ln1_cpu)?;
        diff_stats_ab("L0 k_proj wgpu-vs-cpu", &k_w, &k_cpu.to_device(&wgpu)?)?;
        diff_stats_ab("L0 k_proj cuda-vs-cpu", &k_c, &k_cpu.to_device(&cuda)?)?;

        let v_c = ac.self_attn.v_proj.forward(&ln1_c)?;
        let v_w = aw.self_attn.v_proj.forward(&ln1_w)?;
        diff_stats_ab("L0 v_proj", &v_w, &v_c)?;

        let q_c = q_c
            .reshape((b, l, ac.self_attn.num_heads, ac.self_attn.head_dim))?
            .transpose(1, 2)?;
        let q_w = q_w
            .reshape((b, l, aw.self_attn.num_heads, aw.self_attn.head_dim))?
            .transpose(1, 2)?;
        diff_stats_ab("L0 q_reshape", &q_w, &q_c)?;

        let k_c = k_c
            .reshape((b, l, ac.self_attn.num_kv_heads, ac.self_attn.head_dim))?
            .transpose(1, 2)?;
        let k_w = k_w
            .reshape((b, l, aw.self_attn.num_kv_heads, aw.self_attn.head_dim))?
            .transpose(1, 2)?;
        diff_stats_ab("L0 k_reshape", &k_w, &k_c)?;

        let v_c = v_c
            .reshape((b, l, ac.self_attn.num_kv_heads, ac.self_attn.head_dim))?
            .transpose(1, 2)?;
        let v_w = v_w
            .reshape((b, l, aw.self_attn.num_kv_heads, aw.self_attn.head_dim))?
            .transpose(1, 2)?;

        let q_flat_c = ac.self_attn.q_norm.forward(&q_c.flatten(0, 2)?)?;
        let q_flat_w = aw.self_attn.q_norm.forward(&q_w.flatten(0, 2)?)?;
        diff_stats_ab("L0 q_norm(flat-stride)", &q_flat_w, &q_flat_c)?;
        let q_c = q_flat_c.reshape((b, ac.self_attn.num_heads, l, ac.self_attn.head_dim))?;
        let q_w = q_flat_w.reshape((b, aw.self_attn.num_heads, l, aw.self_attn.head_dim))?;

        let k_flat_c = ac.self_attn.k_norm.forward(&k_c.flatten(0, 2)?)?;
        let k_flat_w = aw.self_attn.k_norm.forward(&k_w.flatten(0, 2)?)?;
        diff_stats_ab("L0 k_norm(flat-stride)", &k_flat_w, &k_flat_c)?;
        let k_c = k_flat_c.reshape((b, ac.self_attn.num_kv_heads, l, ac.self_attn.head_dim))?;
        let k_w = k_flat_w.reshape((b, aw.self_attn.num_kv_heads, l, aw.self_attn.head_dim))?;

        let (q_rope_c, k_rope_c) = ac.self_attn.rotary_emb.apply(&q_c, &k_c, 0)?;
        let (q_rope_w, k_rope_w) = aw.self_attn.rotary_emb.apply(&q_w, &k_w, 0)?;
        diff_stats_ab("L0 q_rope(fused)", &q_rope_w, &q_rope_c)?;
        diff_stats_ab("L0 k_rope(fused)", &k_rope_w, &k_rope_c)?;
        // A/B: wgpu rope_slow vs wgpu fused (same tables, same tensors).
        {
            let cos = aw
                .self_attn
                .rotary_emb
                .cos
                .narrow(0, 0, l)?
                .to_dtype(q_rope_w.dtype())?;
            let sin = aw
                .self_attn
                .rotary_emb
                .sin
                .narrow(0, 0, l)?
                .to_dtype(q_rope_w.dtype())?;
            let q_slow = candle_nn::rotary_emb::rope_slow(&q_w, &cos, &sin)?;
            diff_stats_ab("L0 q_rope_slow_vs_fused_wgpu", &q_slow, &q_rope_w)?;
        }

        let (k_c, v_c) = ac.self_attn.kv_cache.as_mut().unwrap().append(&k_rope_c, &v_c)?;
        let (k_w, v_w) = aw.self_attn.kv_cache.as_mut().unwrap().append(&k_rope_w, &v_w)?;
        diff_stats_ab("L0 cache_k", &k_w, &k_c)?;
        diff_stats_ab("L0 cache_v", &v_w, &v_c)?;

        let k_c = repeat_kv(k_c, ac.self_attn.num_kv_groups)?.contiguous()?;
        let k_w = repeat_kv(k_w, aw.self_attn.num_kv_groups)?.contiguous()?;
        let v_c = repeat_kv(v_c, ac.self_attn.num_kv_groups)?.contiguous()?;
        let v_w = repeat_kv(v_w, aw.self_attn.num_kv_groups)?.contiguous()?;
        diff_stats_ab("L0 k_repeat_kv", &k_w, &k_c)?;
        diff_stats_ab("L0 v_repeat_kv", &v_w, &v_c)?;

        let scale = 1.0 / (ac.self_attn.head_dim as f64).sqrt();
        let scores_c = (q_rope_c.matmul(&k_c.transpose(2, 3)?)? * scale)?;
        let scores_w = (q_rope_w.matmul(&k_w.transpose(2, 3)?)? * scale)?;
        diff_stats_ab("L0 scores", &scores_w, &scores_c)?;

        // Mirror the model's mask dtype conversion (mask follows scores dtype).
        let mask_c = if mask_cuda.dtype() != scores_c.dtype() {
            mask_cuda.to_dtype(scores_c.dtype())?
        } else {
            mask_cuda.clone()
        };
        let mask_w = if mask_wgpu.dtype() != scores_w.dtype() {
            mask_wgpu.to_dtype(scores_w.dtype())?
        } else {
            mask_wgpu.clone()
        };
        let scores_c = scores_c.broadcast_add(&mask_c)?;
        let scores_w = scores_w.broadcast_add(&mask_w)?;
        diff_stats_ab("L0 scores+mask", &scores_w, &scores_c)?;

        let probs_c = candle_nn::ops::softmax_last_dim(&scores_c)?;
        let probs_w = candle_nn::ops::softmax_last_dim(&scores_w)?;
        diff_stats_ab("L0 softmax", &probs_w, &probs_c)?;

        let ctx_c = probs_c.matmul(&v_c)?;
        let ctx_w = probs_w.matmul(&v_w)?;
        diff_stats_ab("L0 ctx(p@v)", &ctx_w, &ctx_c)?;

        let ctx_c = ctx_c.transpose(1, 2)?.reshape((b, l, ac.self_attn.hidden_size))?;
        let ctx_w = ctx_w.transpose(1, 2)?.reshape((b, l, aw.self_attn.hidden_size))?;
        let attn_c = ac.self_attn.o_proj.forward(&ctx_c)?;
        let attn_w = aw.self_attn.o_proj.forward(&ctx_w)?;
        diff_stats_ab("L0 o_proj", &attn_w, &attn_c)?;

        let post_c = (&xc + &attn_c)?;
        let post_w = (&xw + &attn_w)?;
        diff_stats_ab("L0 x+attn", &post_w, &post_c)?;

        let ln2_c = ac.ln2.forward(&post_c)?;
        let ln2_w = aw.ln2.forward(&post_w)?;
        diff_stats_ab("L0 ln2", &ln2_w, &ln2_c)?;

        let gate_c = ac.mlp.gate_proj.forward(&ln2_c)?;
        let gate_w = aw.mlp.gate_proj.forward(&ln2_w)?;
        diff_stats_ab("L0 gate_proj", &gate_w, &gate_c)?;

        let up_c = ac.mlp.up_proj.forward(&ln2_c)?;
        let up_w = aw.mlp.up_proj.forward(&ln2_w)?;
        diff_stats_ab("L0 up_proj", &up_w, &up_c)?;

        let gate_c = gate_c.apply(&ac.mlp.act_fn)?;
        let gate_w = gate_w.apply(&aw.mlp.act_fn)?;
        diff_stats_ab("L0 silu", &gate_w, &gate_c)?;

        let gated_c = (&gate_c * &up_c)?;
        let gated_w = (&gate_w * &up_w)?;
        diff_stats_ab("L0 gated", &gated_w, &gated_c)?;

        let down_c = ac.mlp.down_proj.forward(&gated_c)?;
        let down_w = aw.mlp.down_proj.forward(&gated_w)?;
        diff_stats_ab("L0 down_proj", &down_w, &down_c)?;

        let out_c = (&post_c + &down_c)?;
        let out_w = (&post_w + &down_w)?;
        diff_stats_ab("L0 output", &out_w, &out_c)?;

        // ---- per-layer chain over the full model ----
        // Start from the op-traced layer-0 output (its KV state already holds
        // the 4-token prefill, exactly the state the chained forward expects).
        let l1_in_c = out_c;
        let l1_in_w = out_w;
        let mut h_c = l1_in_c.clone();
        let mut h_w = l1_in_w.clone();
        for i in 1..cuda_model.layers.len() {
            h_c = cuda_model.layers[i].forward(&h_c, Some(&mask_cuda), 0)?;
            h_w = wgpu_model.layers[i].forward(&h_w, Some(&mask_wgpu), 0)?;
            diff_stats_ab(&format!("L{i} hidden"), &h_w, &h_c)?;
        }

        // ---- layer-1 op drill-down: which op explodes on the F32 path? ----
        {
            let l1c = &mut cuda_model.layers[1];
            let l1w = &mut wgpu_model.layers[1];
            let ln1_c = l1c.ln1.forward(&l1_in_c)?;
            let ln1_w = l1w.ln1.forward(&l1_in_w)?;
            diff_stats_ab("L1 ln1", &ln1_w, &ln1_c)?;
            let q_c = l1c.self_attn.q_proj.forward(&ln1_c)?;
            let q_w = l1w.self_attn.q_proj.forward(&ln1_w)?;
            diff_stats_ab("L1 q_proj", &q_w, &q_c)?;
            let k_c = l1c.self_attn.k_proj.forward(&ln1_c)?;
            let k_w = l1w.self_attn.k_proj.forward(&ln1_w)?;
            diff_stats_ab("L1 k_proj", &k_w, &k_c)?;
            let v_c = l1c.self_attn.v_proj.forward(&ln1_c)?;
            let v_w = l1w.self_attn.v_proj.forward(&ln1_w)?;
            diff_stats_ab("L1 v_proj", &v_w, &v_c)?;
            // Reset cache so the attn drill-down matches the chain's fresh-4-token
            // L1 step (the chain already populated a stale 4-token cache).
            l1c.clear_kv_cache();
            l1w.clear_kv_cache();
            let attn_c = l1c.self_attn.forward(&ln1_c, Some(&mask_cuda), 0)?;
            let attn_w = l1w.self_attn.forward(&ln1_w, Some(&mask_wgpu), 0)?;
            diff_stats_ab("L1 attn", &attn_w, &attn_c)?;
            let post_c = (&l1_in_c + &attn_c)?;
            let post_w = (&l1_in_w + &attn_w)?;
            diff_stats_ab("L1 x+attn", &post_w, &post_c)?;
            let ln2_c = l1c.ln2.forward(&post_c)?;
            let ln2_w = l1w.ln2.forward(&post_w)?;
            diff_stats_ab("L1 ln2", &ln2_w, &ln2_c)?;
            let gate_c = l1c.mlp.gate_proj.forward(&ln2_c)?;
            let gate_w = l1w.mlp.gate_proj.forward(&ln2_w)?;
            diff_stats_ab("L1 gate_proj", &gate_w, &gate_c)?;
            // CPU snapshot of early wgpu tensors to detect in-place corruption of
            // live tensors by later ops (pool aliasing).
            let gate_w_snap = gate_w.flatten_all()?.to_vec1::<f32>()?;
            let ln2_w_snap = ln2_w.flatten_all()?.to_vec1::<f32>()?;
            let up_c = l1c.mlp.up_proj.forward(&ln2_c)?;
            let up_w = l1w.mlp.up_proj.forward(&ln2_w)?;
            diff_stats_ab("L1 up_proj", &up_w, &up_c)?;
            let gate_c = gate_c.apply(&l1c.mlp.act_fn)?;
            let gate_w = gate_w.apply(&l1w.mlp.act_fn)?;
            diff_stats_ab("L1 silu", &gate_w, &gate_c)?;
            let gated_c = (&gate_c * &up_c)?;
            let gated_w = (&gate_w * &up_w)?;
            diff_stats_ab("L1 gated", &gated_w, &gated_c)?;
            let down_c = l1c.mlp.down_proj.forward(&gated_c)?;
            let down_w = l1w.mlp.down_proj.forward(&gated_w)?;
            diff_stats_ab("L1 down_proj", &down_w, &down_c)?;
            let down_w_snap = down_w.flatten_all()?.to_vec1::<f32>()?;
            let up_w_snap = up_w.flatten_all()?.to_vec1::<f32>()?;
            // MICRO-DRIFT: quant matmul repeatedly on the SAME input, interleaved
            // with unrelated quant matmuls, to isolate history-dependent results.
            {
                let a_w = l1w.mlp.gate_proj.forward(&ln2_w)?;
                for _ in 0..8 {
                    let _ = l1w.self_attn.v_proj.forward(&ln2_w)?;
                    let _ = l1w.self_attn.k_proj.forward(&ln2_w)?;
                }
                let b_w = l1w.mlp.gate_proj.forward(&ln2_w)?;
                diff_stats_ab("MICRO gate same-input drift", &b_w, &a_w)?;
                let a_c = l1c.mlp.gate_proj.forward(&ln2_c)?;
                for _ in 0..8 {
                    let _ = l1c.self_attn.v_proj.forward(&ln2_c)?;
                    let _ = l1c.self_attn.k_proj.forward(&ln2_c)?;
                }
                let b_c = l1c.mlp.gate_proj.forward(&ln2_c)?;
                diff_stats_ab("MICRO gate same-input drift cuda", &b_c, &a_c)?;
                // Frozen-input gate: rebuild ln2 from a CPU copy each time so no
                // buffer aliasing exists between the two gate calls.
                {
                    let ln2_cpu = ln2_w.flatten_all()?.to_vec1::<f32>()?;
                    let ln2_shape = ln2_w.dims().to_vec();
                    let f1 = Tensor::from_vec(ln2_cpu.clone(), ln2_shape.clone(), &wgpu)?;
                    let g1 = l1w.mlp.gate_proj.forward(&f1)?;
                    for _ in 0..16 {
                        let f = Tensor::from_vec(ln2_cpu.clone(), ln2_shape.clone(), &wgpu)?;
                        let _ = l1w.mlp.gate_proj.forward(&f)?;
                    }
                    let f2 = Tensor::from_vec(ln2_cpu.clone(), ln2_shape.clone(), &wgpu)?;
                    let g2 = l1w.mlp.gate_proj.forward(&f2)?;
                    diff_stats_ab("FROZEN gate drift wgpu", &g2, &g1)?;
                }
                // Full mlp chain determinism.
                let m1 = l1w.mlp.forward(&ln2_w)?;
                let m2 = l1w.mlp.forward(&ln2_w)?;
                diff_stats_ab("MICRO mlp same-input drift", &m2, &m1)?;
                let silu1 = ln2_w.apply(&l1w.mlp.act_fn)?;
                let silu2 = ln2_w.apply(&l1w.mlp.act_fn)?;
                diff_stats_ab("MICRO silu drift", &silu2, &silu1)?;
            }
            // wgpu self-consistency: manual op-chain output vs full_forward output.
            let trace_w = (&post_w + &down_w)?;
            let trace_c = (&post_c + &down_c)?;
            diff_stats_ab("L1 opchain_out", &trace_w, &trace_c)?;
            l1c.clear_kv_cache();
            l1w.clear_kv_cache();
            let full_c = l1c.forward(&l1_in_c, Some(&mask_cuda), 0)?;
            let full_w = l1w.forward(&l1_in_w, Some(&mask_wgpu), 0)?;
            diff_stats_ab("L1 full_forward", &full_w, &full_c)?;
            // DRIFT check: recompute the same manual mlp late, compare to the
            // early drill result (`down_c`/`down_w`). If the wgpu result drifts
            // while cuda stays bit-stable, a kernel is reading stale pooled
            // buffer content that depends on the intervening allocation history.
            l1w.clear_kv_cache();
            l1c.clear_kv_cache();
            let d_ln1_c = l1c.ln1.forward(&l1_in_c)?;
            let d_ln1_w = l1w.ln1.forward(&l1_in_w)?;
            let d_attn_c = l1c.self_attn.forward(&d_ln1_c, Some(&mask_cuda), 0)?;
            let d_attn_w = l1w.self_attn.forward(&d_ln1_w, Some(&mask_wgpu), 0)?;
            macro_rules! check_gate {
                ($tag:literal) => {{
                    let now = gate_w.flatten_all()?.to_vec1::<f32>()?;
                    let mut m = 0f32;
                    let mut ss = 0f64;
                    let mut nbad = 0usize;
                    for (a, b) in now.iter().zip(gate_w_snap.iter()) {
                        let e = (*a as f64) - (*b as f64);
                        m = m.max(e.abs() as f32);
                        ss += e * e;
                        if e.abs() > 1e-3 {
                            nbad += 1;
                        }
                    }
                    println!(
                        "CKPT {} gate_w: max={:.4} rmse={:.5} nbad={}",
                        $tag,
                        m,
                        (ss / now.len() as f64).sqrt(),
                        nbad
                    );
                }};
            }
            check_gate!("after d_ln1");
            let d_post_c = (&l1_in_c + &d_attn_c)?;
            let d_post_w = (&l1_in_w + &d_attn_w)?;
            check_gate!("after d_attn");
            let d_ln2_c = l1c.ln2.forward(&d_post_c)?;
            let d_ln2_w = l1w.ln2.forward(&d_post_w)?;
            check_gate!("after d_ln2");
            let d_mlp_c = d_ln2_c.apply(&l1c.mlp)?;
            let d_mlp_w = d_ln2_w.apply(&l1w.mlp)?;
            check_gate!("after d_mlp");
            diff_stats_ab("DRIFT ln2 wgpu early-vs-late", &d_ln2_w, &ln2_w)?;
            diff_stats_ab("DRIFT ln2 cuda early-vs-late", &d_ln2_c, &ln2_c)?;
            diff_stats_ab("DRIFT mlp wgpu early-vs-late", &d_mlp_w, &down_w)?;
            diff_stats_ab("DRIFT mlp cuda early-vs-late", &d_mlp_c, &down_c)?;
            diff_stats_ab("DRIFT attn wgpu early-vs-late", &d_attn_w, &attn_w)?;
            diff_stats_ab("DRIFT attn cuda early-vs-late", &d_attn_c, &attn_c)?;
            // Narrow within the mlp: gate / up / silu / down individually.
            let d_gate_w = l1w.mlp.gate_proj.forward(&d_ln2_w)?;
            let d_up_w = l1w.mlp.up_proj.forward(&d_ln2_w)?;
            let d_down_w = l1w.mlp.down_proj.forward(&(&d_gate_w.apply(&l1w.mlp.act_fn)? * &d_up_w)?)?;
            diff_stats_ab("DRIFT gate wgpu early-vs-late", &d_gate_w, &gate_w)?;
            diff_stats_ab("DRIFT up wgpu early-vs-late", &d_up_w, &up_w)?;
            diff_stats_ab("DRIFT down wgpu early-vs-late", &d_down_w, &down_w)?;
            // In-place corruption check on the LIVE early tensor objects.
            {
                let now = gate_w.flatten_all()?.to_vec1::<f32>()?;
                let mut max_abs = 0f32;
                let mut sum_sq = 0f64;
                for (a, b) in now.iter().zip(gate_w_snap.iter()) {
                    let err = (*a as f64) - (*b as f64);
                    max_abs = max_abs.max(err.abs() as f32);
                    sum_sq += err * err;
                }
                println!(
                    "SNAP gate_w live-tensor mutated: max_abs={max_abs:.6} rmse={:.8} (n={})",
                    (sum_sq / now.len() as f64).sqrt(),
                    now.len()
                );
                let now2 = ln2_w.flatten_all()?.to_vec1::<f32>()?;
                let mut max_abs2 = 0f32;
                let mut sum_sq2 = 0f64;
                for (a, b) in now2.iter().zip(ln2_w_snap.iter()) {
                    let err = (*a as f64) - (*b as f64);
                    max_abs2 = max_abs2.max(err.abs() as f32);
                    sum_sq2 += err * err;
                }
                println!(
                    "SNAP ln2_w live-tensor mutated: max_abs={max_abs2:.6} rmse={:.8} (n={})",
                    (sum_sq2 / now2.len() as f64).sqrt(),
                    now2.len()
                );
                let now3 = down_w.flatten_all()?.to_vec1::<f32>()?;
                let mut max_abs3 = 0f32;
                let mut sum_sq3 = 0f64;
                for (a, b) in now3.iter().zip(down_w_snap.iter()) {
                    let err = (*a as f64) - (*b as f64);
                    max_abs3 = max_abs3.max(err.abs() as f32);
                    sum_sq3 += err * err;
                }
                println!(
                    "SNAP down_w live-tensor mutated: max_abs={max_abs3:.6} rmse={:.8} (n={})",
                    (sum_sq3 / now3.len() as f64).sqrt(),
                    now3.len()
                );
                let now4 = up_w.flatten_all()?.to_vec1::<f32>()?;
                let mut max_abs4 = 0f32;
                let mut sum_sq4 = 0f64;
                for (a, b) in now4.iter().zip(up_w_snap.iter()) {
                    let err = (*a as f64) - (*b as f64);
                    max_abs4 = max_abs4.max(err.abs() as f32);
                    sum_sq4 += err * err;
                }
                println!(
                    "SNAP up_w live-tensor mutated: max_abs={max_abs4:.6} rmse={:.8} (n={})",
                    (sum_sq4 / now4.len() as f64).sqrt(),
                    now4.len()
                );
            }
            let d_gate_c = l1c.mlp.gate_proj.forward(&d_ln2_c)?;
            let d_up_c = l1c.mlp.up_proj.forward(&d_ln2_c)?;
            diff_stats_ab("DRIFT gate cuda early-vs-late", &d_gate_c, &gate_c)?;
            diff_stats_ab("DRIFT up cuda early-vs-late", &d_up_c, &up_c)?;
            // Determinism check on wgpu itself: rerun the same layer forward.
            l1w.clear_kv_cache();
            l1c.clear_kv_cache();
            let full_w2 = l1w.forward(&l1_in_w, Some(&mask_wgpu), 0)?;
            let full_c2 = l1c.forward(&l1_in_c, Some(&mask_cuda), 0)?;
            diff_stats_ab("wgpu::L1 run1_vs_run2", &full_w2, &full_w)?;
            diff_stats_ab("cuda::L1 run1_vs_run2", &full_c2, &full_c)?;
            // wgpu-vs-wgpu: op-chain vs full forward (should be ~0 on ONE device;
            // a large delta => the manual replication diverges from model's forward).
            diff_stats_ab("wgpu::L1 trace_vs_full", &full_w, &trace_w)?;
            diff_stats_ab("cuda::L1 trace_vs_full", &full_c, &trace_c)?;
            // Fresh-model (no churn) reference for the same L1 forward.
            {
                let mut c3 = load_model(&path, &cuda)?;
                let mut w3 = load_model(&path, &wgpu)?;
                let ids_c3 = Tensor::from_slice(&ids, (1, ids.len()), &cuda)?;
                let ids_w3 = Tensor::from_slice(&ids, (1, ids.len()), &wgpu)?;
                let mask_c3 = crate::utils::build_additive_causal_mask(
                    ids.len(),
                    0,
                    None,
                    &cuda,
                    c3.dtype,
                )?;
                let mask_w3 = crate::utils::build_additive_causal_mask(
                    ids.len(),
                    0,
                    None,
                    &wgpu,
                    w3.dtype,
                )?;
                let mut hc = c3.embed_tokens.forward(&ids_c3)?;
                let mut hw = w3.embed_tokens.forward(&ids_w3)?;
                hc = c3.layers[0].forward(&hc, Some(&mask_c3), 0)?;
                hw = w3.layers[0].forward(&hw, Some(&mask_w3), 0)?;
                let f0 = c3.layers[1].forward(&hc, Some(&mask_c3), 0)?;
                let f0w = w3.layers[1].forward(&hw, Some(&mask_w3), 0)?;
                diff_stats_ab("L1 fresh_forward wgpu-vs-cuda", &f0w, &f0)?;
                diff_stats_ab("wgpu::L1 fresh_vs_churned_full", &f0w, &full_w)?;
                diff_stats_ab("cuda::L1 fresh_vs_churned_full", &f0, &full_c)?;
            }
            // Locate the worst indices in the L1 full_forward divergence.
            {
                let fw = full_w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let fc = full_c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let mut worst: Vec<(usize, f32)> = fw
                    .iter()
                    .zip(fc.iter())
                    .enumerate()
                    .map(|(i, (a, b))| (i, (a - b).abs()))
                    .collect();
                worst.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("L1 full_forward worst5: (index, |diff|, wgpu_val, cuda_val)");
                for (idx, d) in worst.iter().take(5) {
                    println!(
                        "  idx={idx} diff={d:.4} wgpu={:.4} cuda={:.4}",
                        fw[*idx], fc[*idx]
                    );
                }
                let tw = trace_w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let tc = trace_c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                println!("L1 trace worst5: (index, |diff|, wgpu_val, cuda_val)");
                let mut tworst: Vec<(usize, f32)> = tw
                    .iter()
                    .zip(tc.iter())
                    .enumerate()
                    .map(|(i, (a, b))| (i, (a - b).abs()))
                    .collect();
                tworst.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (idx, d) in tworst.iter().take(5) {
                    println!(
                        "  idx={idx} diff={d:.4} wgpu={:.4} cuda={:.4}",
                        tw[*idx], tc[*idx]
                    );
                }
            }
        }

        let h_c = cuda_model.norm.forward(&h_c)?;
        let h_w = wgpu_model.norm.forward(&h_w)?;
        diff_stats_ab("output_norm", &h_w, &h_c)?;

        let logits_c = cuda_model
            .lm_head
            .forward(&h_c.narrow(1, l - 1, 1)?)?
            .squeeze(1)?;
        let logits_w = wgpu_model
            .lm_head
            .forward(&h_w.narrow(1, l - 1, 1)?)?
            .squeeze(1)?;
        diff_stats_ab("final logits", &logits_w, &logits_c)?;
        assert_no_gpu_fallbacks("qwen3_wgpu_cuda_layer0_op_trace", &wgpu)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a usable Vulkan device and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_stage_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        log_gguf_tensor_dtype(&path, "blk.0.attn_q.weight")?;
        log_gguf_tensor_dtype(&path, "blk.0.ffn_up.weight")?;
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        reset_gpu_fallback_count(&vk);
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

        let mask_cpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &cpu,
            cpu_model.dtype,
        )?;
        let mask_vk = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &vk,
            vk_model.dtype,
        )?;
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
        assert_no_gpu_fallbacks("qwen3_local_vulkan_stage_parity", &vk)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a usable Vulkan device and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_forward_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        log_gguf_tensor_dtype(&path, "blk.1.attn_q.weight")?;
        log_gguf_tensor_dtype(&path, "blk.1.ffn_up.weight")?;
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        reset_gpu_fallback_count(&vk);
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut vk_model = load_model(&path, &vk)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_vk = Tensor::from_slice(&ids, (1, ids.len()), &vk)?;
        let prefill_cpu = cpu_model.forward(&ids_cpu, 0)?;
        let prefill_vk = vk_model.forward(&ids_vk, 0)?;
        assert_logits_close("prefill_logits", &prefill_vk, &prefill_cpu, 5e-2)?;

        let next_cpu = Tensor::from_slice(&[5u32], (1, 1), &cpu)?;
        let next_vk = Tensor::from_slice(&[5u32], (1, 1), &vk)?;
        let decode_cpu = cpu_model.forward(&next_cpu, ids.len())?;
        let decode_vk = vk_model.forward(&next_vk, ids.len())?;
        assert_logits_close("decode_logits", &decode_vk, &decode_cpu, 5e-2)?;
        assert_no_gpu_fallbacks("qwen3_local_vulkan_forward_parity", &vk)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a usable Vulkan device and Qwen3 GGUF from CANDLE_QWEN3_GGUF_PATH or hf-hub"]
    #[cfg(feature = "vulkan")]
    fn qwen3_local_vulkan_layer1_topology_parity() -> Result<()> {
        let path = qwen3_gguf_path()?;
        let cpu = Device::Cpu;
        let vk = Device::new_vulkan(0)?;
        reset_gpu_fallback_count(&vk);
        let mut cpu_model = load_model(&path, &cpu)?;
        let mut vk_model = load_model(&path, &vk)?;

        let ids = [1u32, 2, 3, 4];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_vk = Tensor::from_slice(&ids, (1, ids.len()), &vk)?;
        let mask_cpu = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &cpu,
            cpu_model.dtype,
        )?;
        let mask_vk = crate::utils::build_additive_causal_mask(
            ids.len(),
            0,
            None,
            &vk,
            vk_model.dtype,
        )?;

        let layer0_cpu_in = cpu_model.embed_tokens.forward(&ids_cpu)?;
        let layer0_vk_in = vk_model.embed_tokens.forward(&ids_vk)?;
        let layer0_cpu = cpu_model.layers[0].forward(&layer0_cpu_in, Some(&mask_cpu), 0)?;
        let layer0_vk = vk_model.layers[0].forward(&layer0_vk_in, Some(&mask_vk), 0)?;
        assert_tensor_close_with_nmse(
            "layer0_input_to_layer1",
            &layer0_vk,
            &layer0_cpu,
            5e-2,
            5e-3,
        )?;

        let ln1_cpu = vk_model.layers.len(); // keep borrow scopes simple below
        let _ = ln1_cpu;
        let l1_ln1_cpu = cpu_model.layers[1].ln1.forward(&layer0_cpu)?;
        let l1_ln1_vk = vk_model.layers[1].ln1.forward(&layer0_vk)?;
        assert_tensor_close_with_nmse("layer1_ln1", &l1_ln1_vk, &l1_ln1_cpu, 5e-2, 5e-3)?;

        let l1_attn_cpu = cpu_model.layers[1]
            .self_attn
            .forward(&l1_ln1_cpu, Some(&mask_cpu), 0)?;
        let l1_attn_vk = vk_model.layers[1]
            .self_attn
            .forward(&l1_ln1_vk, Some(&mask_vk), 0)?;
        assert_tensor_close_with_nmse("layer1_attn", &l1_attn_vk, &l1_attn_cpu, 5e-2, 5e-3)?;

        let l1_post_attn_cpu = (&layer0_cpu + &l1_attn_cpu)?;
        let l1_post_attn_vk = (&layer0_vk + &l1_attn_vk)?;
        assert_tensor_close_with_nmse(
            "layer1_post_attn",
            &l1_post_attn_vk,
            &l1_post_attn_cpu,
            5e-2,
            5e-3,
        )?;

        let l1_ln2_cpu = cpu_model.layers[1].ln2.forward(&l1_post_attn_cpu)?;
        let l1_ln2_vk = vk_model.layers[1].ln2.forward(&l1_post_attn_vk)?;
        assert_tensor_close_with_nmse("layer1_ln2", &l1_ln2_vk, &l1_ln2_cpu, 5e-2, 5e-3)?;

        let l1_gate_cpu = cpu_model.layers[1].mlp.gate_proj.forward(&l1_ln2_cpu)?;
        let l1_gate_vk = vk_model.layers[1].mlp.gate_proj.forward(&l1_ln2_vk)?;
        assert_tensor_close_with_nmse("layer1_gate", &l1_gate_vk, &l1_gate_cpu, 5e-2, 5e-3)?;

        let l1_up_cpu = cpu_model.layers[1].mlp.up_proj.forward(&l1_ln2_cpu)?;
        let l1_up_vk = vk_model.layers[1].mlp.up_proj.forward(&l1_ln2_vk)?;
        let l1_ln2_cpu_values = l1_ln2_cpu
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let l1_ln2_cpu_on_vk =
            Tensor::from_vec(l1_ln2_cpu_values, l1_ln2_cpu.shape().clone(), &vk)?;
        let l1_up_vk_exact_input = vk_model.layers[1].mlp.up_proj.forward(&l1_ln2_cpu_on_vk)?;
        assert_tensor_close_with_nmse(
            "layer1_up_exact_input",
            &l1_up_vk_exact_input,
            &l1_up_cpu,
            5e-2,
            5e-3,
        )?;
        assert_tensor_close_with_nmse("layer1_up", &l1_up_vk, &l1_up_cpu, 5e-2, 5e-3)?;

        let l1_gate_act_cpu = l1_gate_cpu.apply(&cpu_model.layers[1].mlp.act_fn)?;
        let l1_gate_act_vk = l1_gate_vk.apply(&vk_model.layers[1].mlp.act_fn)?;
        assert_tensor_close_with_nmse(
            "layer1_gate_act",
            &l1_gate_act_vk,
            &l1_gate_act_cpu,
            5e-2,
            5e-3,
        )?;

        let l1_gated_cpu = (&l1_gate_act_cpu * &l1_up_cpu)?;
        let l1_gated_vk = (&l1_gate_act_vk * &l1_up_vk)?;
        assert_tensor_close_with_nmse("layer1_gated", &l1_gated_vk, &l1_gated_cpu, 5e-2, 5e-3)?;

        let l1_mlp_cpu = cpu_model.layers[1].mlp.down_proj.forward(&l1_gated_cpu)?;
        let l1_mlp_vk = vk_model.layers[1].mlp.down_proj.forward(&l1_gated_vk)?;
        assert_tensor_close_with_nmse("layer1_mlp", &l1_mlp_vk, &l1_mlp_cpu, 5e-2, 5e-3)?;

        let l1_out_cpu = (&l1_post_attn_cpu + &l1_mlp_cpu)?;
        let l1_out_vk = (&l1_post_attn_vk + &l1_mlp_vk)?;
        assert_tensor_close_with_nmse("layer1_out", &l1_out_vk, &l1_out_cpu, 5e-2, 5e-3)?;
        assert_no_gpu_fallbacks("qwen3_local_vulkan_layer1_topology_parity", &vk)?;
        Ok(())
    }

    /// Validates the device-gated expanded (num_heads-wide) KV cache against the
    /// reference `repeat_kv` expansion used by the CPU/CUDA/Metal path. The expansion
    /// is a pure data duplication with the SAME head ordering, so the attention values
    /// must be identical; this guards the pp4096 optimization (w-b14) against any
    /// subtle ordering divergence. Runs on CPU (no GPU needed) — the cache logic is
    /// backend-agnostic.
    #[test]
    fn expanded_kv_cache_matches_repeat_kv_reference() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 2usize;
        let num_kv_groups = 2usize;
        let head_dim = 4usize;
        let mut ec = ExpandedKvCache::new(num_kv_groups);

        // Chunk 1: 2 tokens, 8-head.
        let k1 = Tensor::from_vec(
            (0..num_kv_heads * 2 * head_dim).map(|i| i as f32 + 0.5).collect(),
            (1, num_kv_heads, 2, head_dim),
            &device,
        )?;
        let v1 = Tensor::from_vec(
            (0..num_kv_heads * 2 * head_dim).map(|i| 100.0 + i as f32).collect(),
            (1, num_kv_heads, 2, head_dim),
            &device,
        )?;
        let (kf, vf) = ec.append(&k1, &v1)?;
        let k_ref = repeat_kv(k1.clone(), num_kv_groups)?.contiguous()?;
        let v_ref = repeat_kv(v1.clone(), num_kv_groups)?.contiguous()?;
        assert_eq!(
            kf.flatten_all()?.to_vec1::<f32>()?,
            k_ref.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            vf.flatten_all()?.to_vec1::<f32>()?,
            v_ref.flatten_all()?.to_vec1::<f32>()?
        );

        // Chunk 2: append another 2 tokens. The live cache must equal
        // repeat_kv(cat(k1, k2)) expanded — the same as expanding the whole cache.
        let k2 = Tensor::from_vec(
            (1000..1000 + num_kv_heads * 2 * head_dim)
                .map(|i| i as f32 + 0.25)
                .collect(),
            (1, num_kv_heads, 2, head_dim),
            &device,
        )?;
        let v2 = Tensor::from_vec(
            (2000..2000 + num_kv_heads * 2 * head_dim)
                .map(|i| i as f32 + 0.75)
                .collect(),
            (1, num_kv_heads, 2, head_dim),
            &device,
        )?;
        let (kf, vf) = ec.append(&k2, &v2)?;
        let kcat = Tensor::cat(&[&k1, &k2], 2)?;
        let vcat = Tensor::cat(&[&v1, &v2], 2)?;
        assert_eq!(kcat.dims4()?.2, 4);
        let kcat_ref = repeat_kv(kcat.clone(), num_kv_groups)?.contiguous()?;
        let vcat_ref = repeat_kv(vcat.clone(), num_kv_groups)?.contiguous()?;
        let kf_c = kf.contiguous()?;
        let vf_c = vf.contiguous()?;
        assert_eq!(kf_c.dims(), kcat_ref.dims());
        assert_eq!(vf_c.dims(), vcat_ref.dims());
        assert_eq!(kf_c.flatten_all()?.to_vec1::<f32>()?, kcat_ref.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(vf_c.flatten_all()?.to_vec1::<f32>()?, vcat_ref.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }
}
