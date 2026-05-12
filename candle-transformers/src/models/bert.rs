//! BERT (Bidirectional Encoder Representations from Transformers)
//!
//! Bert is a general large language model that can be used for various language tasks:
//! - Compute sentence embeddings for a prompt.
//! - Compute similarities between a set of sentences.
//! - [Arxiv](https://arxiv.org/abs/1810.04805) "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
//! - Upstream [GitHub repo](https://github.com/google-research/bert).
//! - See bert in [candle-examples](https://github.com/huggingface/candle/tree/main/candle-examples/) for runnable code
//!
use super::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

#[derive(Clone)]
struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

impl Config {
    fn _all_mini_lm_l6_v2() -> Self {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        Self {
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

#[derive(Clone)]
struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

#[derive(Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let key_layer_t = key_layer.t()?.contiguous()?;
        let attention_scores = query_layer.matmul(&key_layer_t)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, candle::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
#[derive(Clone)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
#[derive(Clone)]
struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
#[derive(Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
#[derive(Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
#[derive(Clone)]
pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(BertEncoder { layers, span })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl BertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                        BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(attention_mask) => attention_mask.clone(),
            None => input_ids.ones_like()?,
        };
        let dtype = embedding_output.dtype();
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, dtype)?;
        let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;
        Ok(sequence_output)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    let min_value = Tensor::new(f32::MIN, attention_mask.device())?.to_dtype(dtype)?;
    // torch.finfo(dtype).min
    (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(&min_value)
}

//https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L752-L766
struct BertPredictionHeadTransform {
    dense: Linear,
    activation: HiddenActLayer,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let activation = HiddenActLayer::new(config.hidden_act);
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            dense,
            activation,
            layer_norm,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .activation
            .forward(&self.dense.forward(hidden_states)?)?;
        self.layer_norm.forward(&hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L769C1-L790C1
pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
}

impl BertLMPredictionHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let transform = BertPredictionHeadTransform::load(vb.pp("transform"), config)?;
        let decoder = linear(config.hidden_size, config.vocab_size, vb.pp("decoder"))?;
        Ok(Self { transform, decoder })
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.decoder
            .forward(&self.transform.forward(hidden_states)?)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L792
pub struct BertOnlyMLMHead {
    predictions: BertLMPredictionHead,
}

impl BertOnlyMLMHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let predictions = BertLMPredictionHead::load(vb.pp("predictions"), config)?;
        Ok(Self { predictions })
    }
}

impl Module for BertOnlyMLMHead {
    fn forward(&self, sequence_output: &Tensor) -> Result<Tensor> {
        self.predictions.forward(sequence_output)
    }
}

pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertOnlyMLMHead,
}

impl BertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), config)?;
        let cls = BertOnlyMLMHead::load(vb.pp("cls"), config)?;
        Ok(Self { bert, cls })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        self.cls.forward(&sequence_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use candle_nn::VarBuilder;
    use hf_hub::api::sync::Api;

    fn minilm_fixture() -> Result<(Config, std::path::PathBuf)> {
        let api = Api::new()
            .map_err(|err| candle::Error::msg(format!("failed to initialize hf-hub api: {err}")))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/21".to_string(),
        ));
        let config_path = repo
            .get("config.json")
            .map_err(|err| candle::Error::msg(format!("failed to fetch MiniLM config: {err}")))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|err| candle::Error::msg(format!("failed to fetch MiniLM weights: {err}")))?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)
            .map_err(|err| candle::Error::msg(format!("failed to parse MiniLM config: {err}")))?;
        Ok((config, weights_path))
    }

    fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<(usize, f32, f32, f32)> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
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

    fn bert_stage_debug(device: &Device) -> Result<()> {
        let (config, weights_path) = minilm_fixture()?;
        let cpu = Device::Cpu;
        let cpu_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DTYPE, &cpu)? };
        let dev_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)? };
        let cpu_model = BertModel::load(cpu_vb.clone(), &config)?;
        let dev_model = BertModel::load(dev_vb.clone(), &config)?;

        let ids = [101u32, 2023, 2003, 1037, 3231, 102, 0, 0];
        let mask = [1u32, 1, 1, 1, 1, 1, 0, 0];
        let token_type_ids = [0u32; 8];
        let ids_cpu = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
        let ids_dev = Tensor::from_slice(&ids, (1, ids.len()), device)?;
        let mask_cpu = Tensor::from_slice(&mask, (1, mask.len()), &cpu)?;
        let mask_dev = Tensor::from_slice(&mask, (1, mask.len()), device)?;
        let tt_cpu = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), &cpu)?;
        let tt_dev = Tensor::from_slice(&token_type_ids, (1, token_type_ids.len()), device)?;

        let emb_cpu = cpu_model.embeddings.forward(&ids_cpu, &tt_cpu)?;
        let emb_dev = dev_model.embeddings.forward(&ids_dev, &tt_dev)?;
        let (idx, got, expected, diff) = max_abs_diff(&emb_dev, &emb_cpu)?;
        eprintln!("embeddings max diff idx={idx} got={got} expected={expected} diff={diff}");

        let ext_cpu = get_extended_attention_mask(&mask_cpu, emb_cpu.dtype())?;
        let ext_dev = get_extended_attention_mask(&mask_dev, emb_dev.dtype())?;
        let (idx, got, expected, diff) = max_abs_diff(&ext_dev, &ext_cpu)?;
        eprintln!("extended_mask max diff idx={idx} got={got} expected={expected} diff={diff}");

        let mut hs_cpu = emb_cpu;
        let mut hs_dev = emb_dev;
        for (layer_idx, (cpu_layer, dev_layer)) in cpu_model
            .encoder
            .layers
            .iter()
            .zip(dev_model.encoder.layers.iter())
            .enumerate()
        {
            if layer_idx == 0 {
                let cpu_query = cpu_layer.attention.self_attention.query.forward(&hs_cpu)?;
                let dev_query = dev_layer.attention.self_attention.query.forward(&hs_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_query, &cpu_query)?;
                eprintln!(
                    "layer0 query max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_key = cpu_layer.attention.self_attention.key.forward(&hs_cpu)?;
                let dev_key = dev_layer.attention.self_attention.key.forward(&hs_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_key, &cpu_key)?;
                eprintln!(
                    "layer0 key max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_value = cpu_layer.attention.self_attention.value.forward(&hs_cpu)?;
                let dev_value = dev_layer.attention.self_attention.value.forward(&hs_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_value, &cpu_value)?;
                eprintln!(
                    "layer0 value max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_query = cpu_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&cpu_query)?;
                let dev_query = dev_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&dev_query)?;
                let cpu_key = cpu_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&cpu_key)?;
                let dev_key = dev_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&dev_key)?;
                let cpu_value = cpu_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&cpu_value)?;
                let dev_value = dev_layer
                    .attention
                    .self_attention
                    .transpose_for_scores(&dev_value)?;

                let cpu_scores = cpu_query.matmul(&cpu_key.t()?.contiguous()?)?;
                let dev_scores = dev_query.matmul(&dev_key.t()?.contiguous()?)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_scores, &cpu_scores)?;
                eprintln!(
                    "layer0 scores_pre_scale max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let scale = (cpu_layer.attention.self_attention.attention_head_size as f64).sqrt();
                let cpu_scores = (&cpu_scores / scale)?;
                let dev_scores = (&dev_scores / scale)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_scores, &cpu_scores)?;
                eprintln!("layer0 scores_scaled max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_scores = cpu_scores.broadcast_add(&ext_cpu)?;
                let dev_scores = dev_scores.broadcast_add(&ext_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_scores, &cpu_scores)?;
                eprintln!("layer0 scores_masked max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_probs = candle_nn::ops::softmax(&cpu_scores, candle::D::Minus1)?;
                let dev_probs = candle_nn::ops::softmax(&dev_scores, candle::D::Minus1)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_probs, &cpu_probs)?;
                eprintln!(
                    "layer0 probs max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_context = cpu_probs.matmul(&cpu_value)?;
                let dev_context = dev_probs.matmul(&dev_value)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_context, &cpu_context)?;
                eprintln!("layer0 context_pre_merge max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_context = cpu_context
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_from(candle::D::Minus2)?;
                let dev_context = dev_context
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_from(candle::D::Minus2)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_context, &cpu_context)?;
                eprintln!(
                    "layer0 context max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_attn_out = cpu_layer
                    .attention
                    .self_output
                    .forward(&cpu_context, &hs_cpu)?;
                let dev_attn_out = dev_layer
                    .attention
                    .self_output
                    .forward(&dev_context, &hs_dev)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_attn_out, &cpu_attn_out)?;
                eprintln!("layer0 attn_output max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_inter = cpu_layer.intermediate.forward(&cpu_attn_out)?;
                let dev_inter = dev_layer.intermediate.forward(&dev_attn_out)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_inter, &cpu_inter)?;
                eprintln!("layer0 intermediate max diff idx={idx} got={got} expected={expected} diff={diff}");

                let cpu_output_dense = cpu_layer.output.dense.forward(&cpu_inter)?;
                let dev_output_dense = dev_layer.output.dense.forward(&dev_inter)?;
                let (idx, got, expected, diff) =
                    max_abs_diff(&dev_output_dense, &cpu_output_dense)?;
                eprintln!(
                    "layer0 output_dense max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_dense_vb = {
                    let root = cpu_vb.pp("encoder").pp("layer.0").pp("output").pp("dense");
                    if root.contains_tensor("weight") {
                        root
                    } else {
                        cpu_vb
                            .pp("bert")
                            .pp("encoder")
                            .pp("layer.0")
                            .pp("output")
                            .pp("dense")
                    }
                };
                let dev_dense_vb = {
                    let root = dev_vb.pp("encoder").pp("layer.0").pp("output").pp("dense");
                    if root.contains_tensor("weight") {
                        root
                    } else {
                        dev_vb
                            .pp("bert")
                            .pp("encoder")
                            .pp("layer.0")
                            .pp("output")
                            .pp("dense")
                    }
                };
                let cpu_dense_w =
                    cpu_dense_vb.get((config.hidden_size, config.intermediate_size), "weight")?;
                let dev_dense_w =
                    dev_dense_vb.get((config.hidden_size, config.intermediate_size), "weight")?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_dense_w, &cpu_dense_w)?;
                eprintln!(
                    "layer0 output_dense_weight max diff idx={idx} got={got} expected={expected} diff={diff}"
                );
                let cpu_dense_b = cpu_dense_vb.get(config.hidden_size, "bias")?;
                let dev_dense_b = dev_dense_vb.get(config.hidden_size, "bias")?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_dense_b, &cpu_dense_b)?;
                eprintln!(
                    "layer0 output_dense_bias max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_output_mm = cpu_inter
                    .reshape((ids.len(), config.intermediate_size))?
                    .matmul(&cpu_dense_w.t()?.contiguous()?)?
                    .reshape((1, ids.len(), config.hidden_size))?;
                let dev_output_mm = dev_inter
                    .reshape((ids.len(), config.intermediate_size))?
                    .matmul(&dev_dense_w.t()?.contiguous()?)?
                    .reshape((1, ids.len(), config.hidden_size))?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_output_mm, &cpu_output_mm)?;
                eprintln!(
                    "layer0 output_dense_manual_matmul max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_output_residual = (&cpu_output_dense + &cpu_attn_out)?;
                let dev_output_residual = (&dev_output_dense + &dev_attn_out)?;
                let (idx, got, expected, diff) =
                    max_abs_diff(&dev_output_residual, &cpu_output_residual)?;
                eprintln!(
                    "layer0 output_residual max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_output_norm = cpu_layer.output.layer_norm.forward(&cpu_output_residual)?;
                let dev_output_norm = dev_layer.output.layer_norm.forward(&dev_output_residual)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_output_norm, &cpu_output_norm)?;
                eprintln!(
                    "layer0 output_norm max diff idx={idx} got={got} expected={expected} diff={diff}"
                );

                let cpu_out = cpu_layer.output.forward(&cpu_inter, &cpu_attn_out)?;
                let dev_out = dev_layer.output.forward(&dev_inter, &dev_attn_out)?;
                let (idx, got, expected, diff) = max_abs_diff(&dev_out, &cpu_out)?;
                eprintln!(
                    "layer0 final max diff idx={idx} got={got} expected={expected} diff={diff}"
                );
            }
            hs_cpu = cpu_layer.forward(&hs_cpu, &ext_cpu)?;
            hs_dev = dev_layer.forward(&hs_dev, &ext_dev)?;
            let (idx, got, expected, diff) = max_abs_diff(&hs_dev, &hs_cpu)?;
            eprintln!(
                "encoder layer {layer_idx} max diff idx={idx} got={got} expected={expected} diff={diff}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "manual bert GPU stage diagnostic"]
    fn bert_stage_debug_vulkan() -> Result<()> {
        let device = Device::new_vulkan(0)?;
        bert_stage_debug(&device)
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "manual bert GPU stage diagnostic"]
    fn bert_stage_debug_wgpu() -> Result<()> {
        let device = Device::new_wgpu(0)?;
        bert_stage_debug(&device)
    }
}
