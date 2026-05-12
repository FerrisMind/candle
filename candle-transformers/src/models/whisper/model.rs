use super::Config;
use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Conv1d, Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder};

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let q = self.query.forward(x)?;
        let (k, v) = match xa {
            None => {
                let k = self.key.forward(x)?;
                let v = self.value.forward(x)?;
                (k, v)
            }
            Some(x) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = self.key.forward(x)?;
                    let v = self.value.forward(x)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)?.contiguous()? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        if let Some(mask) = mask {
            let mask = mask.i((0..n_ctx, 0..n_ctx))?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(&v)?
        }
        .transpose(1, 2)?
        .contiguous()?
        .flatten_from(2)?;
        Ok(wv)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_attn_ln = layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn = self
            .attn
            .forward(&self.attn_ln.forward(x)?, None, mask, flush_kv_cache)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &mut self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None, flush_kv_cache)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        x + mlp
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.reset_kv_cache();
        }
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(candle::DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x)?.gelu()?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x)?.gelu()?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, None, None, flush_kv_cache)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
    span: tracing::Span,
    span_final: tracing::Span,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            span,
            span_final,
        })
    }

    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let _enter = self.span.enter();
        let last = x.dim(D::Minus1)?;
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, Some(xa), Some(&self.mask), flush_kv_cache)?;
        }
        self.ln.forward(&x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        let logits = {
            let _enter = self.span_final.enter();
            x.matmul(&w.t()?)?
        };
        Ok(logits)
    }

    pub fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        self.encoder
            .blocks
            .iter_mut()
            .for_each(|b| b.reset_kv_cache());
        self.decoder.reset_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::sync::Api;

    fn whisper_fixture() -> Result<(Config, std::path::PathBuf)> {
        let api = Api::new()
            .map_err(|err| candle::Error::msg(format!("failed to initialize hf-hub api: {err}")))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "openai/whisper-tiny.en".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/15".to_string(),
        ));
        let config_path = repo
            .get("config.json")
            .map_err(|err| candle::Error::msg(format!("failed to fetch Whisper config: {err}")))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|err| candle::Error::msg(format!("failed to fetch Whisper weights: {err}")))?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
            .map_err(|err| candle::Error::msg(format!("failed to parse Whisper config: {err}")))?;
        Ok((config, weights_path))
    }

    fn deterministic_f32_data(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bits = ((state >> 41) as u32) | 0x3f80_0000;
                (f32::from_bits(bits) - 1.0) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<(usize, f32, f32, f32)> {
        let actual = actual
            .to_dtype(candle::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(candle::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut best_idx = 0usize;
        let mut best_actual = 0f32;
        let mut best_expected = 0f32;
        let mut best_diff = 0f32;
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            if diff > best_diff {
                best_idx = idx;
                best_actual = *a;
                best_expected = *e;
                best_diff = diff;
            }
        }
        Ok((best_idx, best_actual, best_expected, best_diff))
    }

    fn whisper_encoder_stage_debug(device: &Device) -> Result<()> {
        let (config, weights_path) = whisper_fixture()?;
        let cpu = Device::Cpu;
        let cpu_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], super::super::DTYPE, &cpu)?
        };
        let dev_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], super::super::DTYPE, device)?
        };
        let mut cpu_model = Whisper::load(&cpu_vb, config.clone())?;
        let mut dev_model = Whisper::load(&dev_vb, config.clone())?;

        let mel = deterministic_f32_data(config.num_mel_bins * super::super::N_FRAMES, 0x5151);
        let mel_cpu = Tensor::from_vec(
            mel.clone(),
            (1, config.num_mel_bins, super::super::N_FRAMES),
            &cpu,
        )?;
        let mel_dev = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, super::super::N_FRAMES),
            device,
        )?;
        let (idx, got, expected, diff) = max_abs_diff(&mel_dev, &mel_cpu)?;
        eprintln!("mel_input max diff idx={idx} got={got} expected={expected} diff={diff}");

        let (idx, got, expected, diff) = max_abs_diff(
            dev_model.encoder.conv1.weight(),
            cpu_model.encoder.conv1.weight(),
        )?;
        eprintln!("conv1_weight max diff idx={idx} got={got} expected={expected} diff={diff}");

        let conv1_cfg = *cpu_model.encoder.conv1.config();
        let conv1_cpu_only = mel_cpu.conv1d_with_algo(
            cpu_model.encoder.conv1.weight(),
            conv1_cfg.padding,
            conv1_cfg.stride,
            conv1_cfg.dilation,
            conv1_cfg.groups,
            conv1_cfg.cudnn_fwd_algo,
        )?;
        let conv1_dev_only = mel_dev.conv1d_with_algo(
            dev_model.encoder.conv1.weight(),
            conv1_cfg.padding,
            conv1_cfg.stride,
            conv1_cfg.dilation,
            conv1_cfg.groups,
            conv1_cfg.cudnn_fwd_algo,
        )?;
        let (idx, got, expected, diff) = max_abs_diff(&conv1_dev_only, &conv1_cpu_only)?;
        eprintln!("conv1_only max diff idx={idx} got={got} expected={expected} diff={diff}");
        let conv1_cpu_raw = cpu_model.encoder.conv1.forward(&mel_cpu)?;
        let conv1_dev_raw = dev_model.encoder.conv1.forward(&mel_dev)?;
        let (idx, got, expected, diff) = max_abs_diff(&conv1_dev_raw, &conv1_cpu_raw)?;
        eprintln!("conv1_raw max diff idx={idx} got={got} expected={expected} diff={diff}");
        let conv1_cpu = conv1_cpu_raw.gelu()?;
        let conv1_dev = conv1_dev_raw.gelu()?;
        let (idx, got, expected, diff) = max_abs_diff(&conv1_dev, &conv1_cpu)?;
        eprintln!("conv1_gelu max diff idx={idx} got={got} expected={expected} diff={diff}");

        let conv2_cfg = *cpu_model.encoder.conv2.config();
        let conv2_cpu_only = conv1_cpu.conv1d_with_algo(
            cpu_model.encoder.conv2.weight(),
            conv2_cfg.padding,
            conv2_cfg.stride,
            conv2_cfg.dilation,
            conv2_cfg.groups,
            conv2_cfg.cudnn_fwd_algo,
        )?;
        let conv2_dev_only = conv1_dev.conv1d_with_algo(
            dev_model.encoder.conv2.weight(),
            conv2_cfg.padding,
            conv2_cfg.stride,
            conv2_cfg.dilation,
            conv2_cfg.groups,
            conv2_cfg.cudnn_fwd_algo,
        )?;
        let (idx, got, expected, diff) = max_abs_diff(&conv2_dev_only, &conv2_cpu_only)?;
        eprintln!("conv2_only max diff idx={idx} got={got} expected={expected} diff={diff}");
        let conv2_cpu_raw = cpu_model.encoder.conv2.forward(&conv1_cpu)?;
        let conv2_dev_raw = dev_model.encoder.conv2.forward(&conv1_dev)?;
        let (idx, got, expected, diff) = max_abs_diff(&conv2_dev_raw, &conv2_cpu_raw)?;
        eprintln!("conv2_raw max diff idx={idx} got={got} expected={expected} diff={diff}");
        let conv2_cpu = conv2_cpu_raw.gelu()?;
        let conv2_dev = conv2_dev_raw.gelu()?;
        let (idx, got, expected, diff) = max_abs_diff(&conv2_dev, &conv2_cpu)?;
        eprintln!("conv2_gelu max diff idx={idx} got={got} expected={expected} diff={diff}");

        let x_cpu = conv2_cpu.transpose(1, 2)?;
        let x_dev = conv2_dev.transpose(1, 2)?;
        let (idx, got, expected, diff) = max_abs_diff(&x_dev, &x_cpu)?;
        eprintln!("transpose max diff idx={idx} got={got} expected={expected} diff={diff}");

        let (_bsize, seq_len, _hidden) = x_cpu.dims3()?;
        let pos_cpu = cpu_model
            .encoder
            .positional_embedding
            .narrow(0, 0, seq_len)?;
        let pos_dev = dev_model
            .encoder
            .positional_embedding
            .narrow(0, 0, seq_len)?;
        let (idx, got, expected, diff) = max_abs_diff(&pos_dev, &pos_cpu)?;
        eprintln!("positional max diff idx={idx} got={got} expected={expected} diff={diff}");

        let mut x_cpu = x_cpu.broadcast_add(&pos_cpu)?;
        let mut x_dev = x_dev.broadcast_add(&pos_dev)?;
        let (idx, got, expected, diff) = max_abs_diff(&x_dev, &x_cpu)?;
        eprintln!("input_plus_pos max diff idx={idx} got={got} expected={expected} diff={diff}");

        for (block_idx, (cpu_block, dev_block)) in cpu_model
            .encoder
            .blocks
            .iter_mut()
            .zip(dev_model.encoder.blocks.iter_mut())
            .enumerate()
        {
            if block_idx == 0 {
                let cpu_attn_in = cpu_block.attn_ln.forward(&x_cpu)?;
                let dev_attn_in = dev_block.attn_ln.forward(&x_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_attn_in, &cpu_attn_in)?;
                eprintln!(
                    "block0 attn_ln max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_q = cpu_block.attn.query.forward(&cpu_attn_in)?;
                let dev_q = dev_block.attn.query.forward(&dev_attn_in)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_q, &cpu_q)?;
                eprintln!(
                    "block0 q_proj max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_k = cpu_block.attn.key.forward(&cpu_attn_in)?;
                let dev_k = dev_block.attn.key.forward(&dev_attn_in)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_k, &cpu_k)?;
                eprintln!(
                    "block0 k_proj max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_v = cpu_block.attn.value.forward(&cpu_attn_in)?;
                let dev_v = dev_block.attn.value.forward(&dev_attn_in)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_v, &cpu_v)?;
                eprintln!(
                    "block0 v_proj max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let (_, _n_ctx, n_state) = cpu_q.dims3()?;
                let scale = ((n_state / cpu_block.attn.n_head) as f64).powf(-0.25);
                let cpu_q = (cpu_block.attn.reshape_head(&cpu_q)? * scale)?;
                let dev_q = (dev_block.attn.reshape_head(&dev_q)? * scale)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_q, &cpu_q)?;
                eprintln!(
                    "block0 q_heads max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_k = (cpu_block
                    .attn
                    .reshape_head(&cpu_k)?
                    .transpose(2, 3)?
                    .contiguous()?
                    * scale)?;
                let dev_k = (dev_block
                    .attn
                    .reshape_head(&dev_k)?
                    .transpose(2, 3)?
                    .contiguous()?
                    * scale)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_k, &cpu_k)?;
                eprintln!(
                    "block0 k_heads max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_v = cpu_block.attn.reshape_head(&cpu_v)?.contiguous()?;
                let dev_v = dev_block.attn.reshape_head(&dev_v)?.contiguous()?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_v, &cpu_v)?;
                eprintln!(
                    "block0 v_heads max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_qk = cpu_q.matmul(&cpu_k)?;
                let dev_qk = dev_q.matmul(&dev_k)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_qk, &cpu_qk)?;
                eprintln!("block0 qk max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_w = candle_nn::ops::softmax_last_dim(&cpu_qk)?;
                let dev_w = candle_nn::ops::softmax_last_dim(&dev_qk)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_w, &cpu_w)?;
                eprintln!(
                    "block0 softmax max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_wv = cpu_w
                    .matmul(&cpu_v)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_from(2)?;
                let dev_wv = dev_w
                    .matmul(&dev_v)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_from(2)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_wv, &cpu_wv)?;
                eprintln!("block0 wv max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_attn_out = cpu_block.attn.out.forward(&cpu_wv)?;
                let dev_attn_out = dev_block.attn.out.forward(&dev_wv)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_attn_out, &cpu_attn_out)?;
                eprintln!(
                    "block0 out_proj max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_resid = (&x_cpu + &cpu_attn_out)?;
                let dev_resid = (&x_dev + &dev_attn_out)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_resid, &cpu_resid)?;
                eprintln!("block0 attn_resid max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_mlp_in = cpu_block.mlp_ln.forward(&cpu_resid)?;
                let dev_mlp_in = dev_block.mlp_ln.forward(&dev_resid)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_mlp_in, &cpu_mlp_in)?;
                eprintln!(
                    "block0 mlp_ln max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_mlp1 = cpu_block.mlp_linear1.forward(&cpu_mlp_in)?;
                let dev_mlp1 = dev_block.mlp_linear1.forward(&dev_mlp_in)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_mlp1, &cpu_mlp1)?;
                eprintln!(
                    "block0 mlp_fc1 max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_mlp1 = cpu_mlp1.gelu()?;
                let dev_mlp1 = dev_mlp1.gelu()?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_mlp1, &cpu_mlp1)?;
                eprintln!(
                    "block0 mlp_gelu max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_mlp2 = cpu_block.mlp_linear2.forward(&cpu_mlp1)?;
                let dev_mlp2 = dev_block.mlp_linear2.forward(&dev_mlp1)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_mlp2, &cpu_mlp2)?;
                eprintln!(
                    "block0 mlp_fc2 max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_block_out = (&cpu_resid + &cpu_mlp2)?;
                let dev_block_out = (&dev_resid + &dev_mlp2)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_block_out, &cpu_block_out)?;
                eprintln!(
                    "block0 manual_out max diff idx={idx} got={got} expected={expected} diff={diff}"
                );
            }
            x_cpu = cpu_block.forward(&x_cpu, None, None, true)?;
            x_dev = dev_block.forward(&x_dev, None, None, true)?;
            let (idx, got, expected, diff) = max_abs_diff(&x_dev, &x_cpu)?;
            eprintln!(
                "encoder block {block_idx} max diff idx={idx} got={got} expected={expected} diff={diff}"
            );
        }

        let ln_cpu = cpu_model.encoder.ln_post.forward(&x_cpu)?;
        let ln_dev = dev_model.encoder.ln_post.forward(&x_dev)?;
        let (idx, got, expected, diff) = max_abs_diff(&ln_dev, &ln_cpu)?;
        eprintln!("ln_post max diff idx={idx} got={got} expected={expected} diff={diff}");
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "manual whisper GPU stage diagnostic"]
    fn whisper_encoder_stage_debug_vulkan() -> Result<()> {
        let device = Device::new_vulkan(0)?;
        whisper_encoder_stage_debug(&device)
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "manual whisper GPU stage diagnostic"]
    fn whisper_encoder_stage_debug_wgpu() -> Result<()> {
        let device = Device::new_wgpu(0)?;
        whisper_encoder_stage_debug(&device)
    }
}
