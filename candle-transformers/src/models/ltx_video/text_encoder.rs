//! T5 text encoder integration for LTX-Video.
//!
//! This module wraps `candle_transformers::models::t5::T5EncoderModel` together with a tokenizer
//! to produce prompt embeddings, attention masks, and support negative prompts for classifier-free
//! guidance. The implementation enforces the 4096-dimensional hidden size required by the
//! Transformer3D cross-attention layers and exposes helpers for loading checkpoint formats.
use candle::{DType, Device, Error, Result, Tensor};
use candle_nn::{activation::Activation, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::models::t5::{self, Config as T5ModelConfig, T5EncoderModel};

/// Configuration shared between the LTX-Video pipeline and the T5 text encoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct T5Config {
    /// Model name or local path (e.g., `PixArt-alpha/PixArt-XL-2-1024-MS`).
    pub model_name: String,
    /// Maximum number of tokens to keep per prompt.
    pub max_length: usize,
    /// Hidden size of the encoder outputs (must be a multiple of 64). Default 4096 = T5-XXL.
    pub hidden_size: usize,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            model_name: "PixArt-alpha/PixArt-XL-2-1024-MS".to_string(),
            max_length: 512,
            hidden_size: 4096,
        }
    }
}

impl T5Config {
    pub(crate) fn to_t5_config(&self) -> Result<T5ModelConfig> {
        if self.hidden_size == 0 || !self.hidden_size.is_multiple_of(64) {
            return Err(Error::Msg(format!(
                "T5 hidden_size must be a positive multiple of 64, got {}",
                self.hidden_size
            )));
        }
        let num_heads = self.hidden_size / 64;
        let d_ff = self.hidden_size * 4;

        Ok(T5ModelConfig {
            vocab_size: 32128,
            d_model: self.hidden_size,
            d_kv: 64,
            d_ff,
            num_layers: 24,
            num_decoder_layers: None,
            num_heads,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: t5::ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: false,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        })
    }
}

/// Result of encoding both conditional and unconditional prompts.
#[derive(Debug)]
pub struct TextConditioning {
    pub prompt_embeds: Tensor,
    pub prompt_attention_mask: Tensor,
    pub negative_prompt_embeds: Tensor,
    pub negative_prompt_attention_mask: Tensor,
}

/// Wraps the Candle T5 encoder and the tokenizer used by LTX-Video.
pub struct T5TextEncoder {
    encoder: T5EncoderModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    max_length: usize,
    pad_token_id: u32,
}

impl T5TextEncoder {
    /// Creates a new text encoder from a tokenizer and pre-filled `VarBuilder`.
    pub fn new(vb: VarBuilder, tokenizer: Tokenizer, cfg: &T5Config) -> Result<Self> {
        let t5_config = cfg.to_t5_config()?;
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let pad_token_id = t5_config.pad_token_id as u32;
        let encoder = T5EncoderModel::load(vb, &t5_config)?;

        Ok(Self {
            encoder,
            tokenizer,
            device,
            dtype,
            max_length: cfg.max_length,
            pad_token_id,
        })
    }

    /// Encode the provided prompts and generate attention masks. Negative prompts are required for
    /// classifier-free guidance and are defaulted to the empty string when not supplied.
    pub fn encode<T, U>(
        &mut self,
        prompts: &[T],
        negative_prompts: Option<&[U]>,
        max_length: Option<usize>,
    ) -> Result<TextConditioning>
    where
        T: AsRef<str>,
        U: AsRef<str>,
    {
        let seq_len = match max_length {
            Some(value) if value > 0 => value,
            None => self.max_length,
            _ => {
                return Err(Error::Msg(
                    "max_length must be a positive number".to_string(),
                ))
            }
        };

        let normalized_prompts = Self::normalize(prompts);
        if normalized_prompts.is_empty() {
            return Err(Error::Msg("At least one prompt is required".to_string()));
        }
        let batch_size = normalized_prompts.len();
        let prompt_input = self.tokenize(&normalized_prompts, seq_len)?;
        let prompt_embeds = self.encoder.forward(&prompt_input.tokens)?;

        let negative_texts = Self::build_negative(batch_size, negative_prompts)?;
        let negative_input = self.tokenize(&negative_texts, seq_len)?;
        let negative_prompt_embeds = self.encoder.forward(&negative_input.tokens)?;

        Ok(TextConditioning {
            prompt_embeds,
            prompt_attention_mask: prompt_input.attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask: negative_input.attention_mask,
        })
    }

    /// Returns the dtype used for embeddings.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the maximum token length enforced by this encoder.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Loads a tokenizer from a local file containing the HuggingFace JSON descriptor.
    pub fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
        Tokenizer::from_file(path)
            .map_err(|e| Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }

    /// Helper that selects the correct VarBuilder loader based on the file extensions.
    pub fn load_var_builder<'a, P: AsRef<Path>>(
        paths: &'a [P],
        dtype: DType,
        device: &'a Device,
    ) -> Result<VarBuilder<'a>> {
        if paths.is_empty() {
            return Err(Error::Msg("No checkpoint files provided".to_string()));
        }

        if paths.iter().all(|path| Self::is_safetensors(path.as_ref())) {
            // SAFETENSORS (preferred format)
            unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
        } else if paths.len() == 1 && Self::is_pytorch(paths[0].as_ref()) {
            let file = paths[0].as_ref();
            VarBuilder::from_pth(file, dtype, device)
        } else {
            Err(Error::Msg(
                "Checkpoint files must be either multiple .safetensors or a single .pth/.bin"
                    .to_string(),
            ))
        }
    }

    fn tokenize(&self, texts: &[String], seq_len: usize) -> Result<TokenizedText> {
        let batch_size = texts.len();
        let mut ids = Vec::with_capacity(batch_size * seq_len);
        let mut mask = Vec::with_capacity(batch_size * seq_len);

        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| Error::Msg(format!("Tokenizer error: {}", e)))?;

            let truncated_len = encoding.get_ids().len().min(seq_len);
            ids.extend(encoding.get_ids().iter().take(truncated_len).copied());
            mask.extend(std::iter::repeat_n(1u32, truncated_len));

            if truncated_len < seq_len {
                let pad_len = seq_len - truncated_len;
                ids.extend(std::iter::repeat_n(self.pad_token_id, pad_len));
                mask.extend(std::iter::repeat_n(0u32, pad_len));
            }
        }

        let tokens = Tensor::new(ids.as_slice(), &self.device)?.reshape((batch_size, seq_len))?;
        let attention_mask =
            Tensor::new(mask.as_slice(), &self.device)?.reshape((batch_size, seq_len))?;

        Ok(TokenizedText {
            tokens,
            attention_mask,
        })
    }

    fn normalize<T: AsRef<str>>(texts: &[T]) -> Vec<String> {
        texts
            .iter()
            .map(|text| text.as_ref().trim().to_string())
            .collect()
    }

    fn build_negative<U: AsRef<str>>(
        batch_size: usize,
        negative_prompts: Option<&[U]>,
    ) -> Result<Vec<String>> {
        match negative_prompts {
            Some(values) if values.len() == 1 => {
                let token = values[0].as_ref().trim().to_string();
                Ok(vec![token; batch_size])
            }
            Some(values) if values.len() == batch_size => Ok(values
                .iter()
                .map(|text| text.as_ref().trim().to_string())
                .collect()),
            Some(values) => Err(Error::Msg(format!(
                "Expected negative prompt length to be 1 or {}, got {}",
                batch_size,
                values.len()
            ))),
            None => Ok(vec![String::new(); batch_size]),
        }
    }

    fn is_safetensors(path: &Path) -> bool {
        path.extension()
            .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
    }

    fn is_pytorch(path: &Path) -> bool {
        path.extension()
            .map(|ext| {
                ext.eq_ignore_ascii_case("pth")
                    || ext.eq_ignore_ascii_case("bin")
                    || ext.eq_ignore_ascii_case("pt")
            })
            .unwrap_or(false)
    }
}

struct TokenizedText {
    tokens: Tensor,
    attention_mask: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_config_default() {
        let cfg = T5Config::default();
        assert_eq!(cfg.max_length, 512);
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn negative_prompt_expansion() {
        let negatives = vec!["foo".to_string()];
        let expanded = T5TextEncoder::build_negative(2, Some(&negatives)).unwrap();
        assert_eq!(expanded, vec!["foo".to_string(), "foo".to_string()]);
    }
}
