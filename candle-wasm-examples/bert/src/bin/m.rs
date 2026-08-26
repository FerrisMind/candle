use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use candle_wasm_device_select::{DeviceMode, ResolvedKind};
use candle_wasm_example_bert::console_log;
use tokenizers::{PaddingParams, Tokenizer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    bert: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    resolved: ResolvedKind,
    adapter_name: Option<String>,
}

#[wasm_bindgen]
impl Model {
    /// Sync CPU-only constructor kept for compatibility. Prefer [`Self::load_with_device`].
    #[wasm_bindgen(constructor)]
    pub fn load(weights: Vec<u8>, tokenizer: Vec<u8>, config: Vec<u8>) -> Result<Model, JsError> {
        Self::load_on_device(
            weights,
            tokenizer,
            config,
            Device::Cpu,
            ResolvedKind::Cpu,
            None,
        )
    }

    /// Async load that resolves `device_mode` (`cpu` | `wgpu` | `auto`) via
    /// `candle_wasm_device_select::DeviceMode::resolve`.
    #[wasm_bindgen]
    pub async fn load_with_device(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        device_mode: String,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        let mode =
            DeviceMode::parse(&device_mode).map_err(|e| JsError::new(&e.to_string()))?;
        let resolved = mode
            .resolve()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        Self::load_on_device(
            weights,
            tokenizer,
            config,
            resolved.device,
            resolved.resolved,
            resolved.adapter_name,
        )
    }

    fn load_on_device(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        device: Device,
        resolved: ResolvedKind,
        adapter_name: Option<String>,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!(
            "loading model on {:?}",
            match resolved {
                ResolvedKind::Cpu => "cpu",
                ResolvedKind::Wgpu => "wgpu",
            }
        );
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let config: Config = serde_json::from_slice(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let bert = BertModel::load(vb, &config)?;

        Ok(Self {
            bert,
            tokenizer,
            device,
            resolved,
            adapter_name,
        })
    }

    #[wasm_bindgen]
    pub fn resolved_device(&self) -> String {
        match self.resolved {
            ResolvedKind::Cpu => "cpu".to_string(),
            ResolvedKind::Wgpu => "wgpu".to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn adapter_name(&self) -> Option<String> {
        self.adapter_name.clone()
    }

    pub fn get_embeddings(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let input: Params =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let sentences = input.sentences;
        let normalize_embeddings = input.normalize_embeddings;

        let device = &self.device;
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(|m| JsError::new(&m.to_string()))?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let attention_mask: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        console_log!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self
            .bert
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        console_log!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if normalize_embeddings {
            embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?
        } else {
            embeddings
        };
        let embeddings_data = embeddings.to_vec2()?;
        Ok(serde_wasm_bindgen::to_value(&Embeddings {
            data: embeddings_data,
        })?)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Embeddings {
    data: Vec<Vec<f32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Params {
    sentences: Vec<String>,
    normalize_embeddings: bool,
}
fn main() {
    console_error_panic_hook::set_once();
}
