use candle_wasm_device_select::DeviceMode;
use candle_wasm_example_whisper::worker::{Decoder as D, ModelData};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Decoder {
    decoder: D,
}

#[wasm_bindgen]
impl Decoder {
    /// Sync constructor: always loads on CPU (compat with existing JS until setDevice migrates).
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        mel_filters: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
        is_multilingual: bool,
        timestamps: bool,
        task: Option<String>,
        language: Option<String>,
    ) -> Result<Decoder, JsError> {
        let decoder = D::load_cpu(ModelData {
            tokenizer,
            mel_filters,
            config,
            quantized,
            weights,
            is_multilingual,
            timestamps,
            task,
            language,
            device_mode: DeviceMode::Cpu,
        });

        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Async factory: resolve `device_mode` (`"cpu"` | `"wgpu"` | `"auto"`) then load weights.
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn load_with_device(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        mel_filters: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
        is_multilingual: bool,
        timestamps: bool,
        task: Option<String>,
        language: Option<String>,
        device_mode: String,
    ) -> Result<Decoder, JsError> {
        let device_mode = DeviceMode::parse(&device_mode)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let decoder = D::load(ModelData {
            tokenizer,
            mel_filters,
            config,
            quantized,
            weights,
            is_multilingual,
            timestamps,
            task,
            language,
            device_mode,
        })
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { decoder })
    }

    /// `"cpu"` or `"wgpu"` after successful load.
    #[wasm_bindgen]
    pub fn resolved_device(&self) -> String {
        self.decoder.resolved_device().to_string()
    }

    /// Adapter display name when resolved to wgpu; `undefined` on CPU.
    #[wasm_bindgen]
    pub fn adapter_name(&self) -> Option<String> {
        self.decoder.adapter_name().map(str::to_string)
    }

    #[wasm_bindgen]
    pub fn decode(&mut self, wav_input: Vec<u8>) -> Result<String, JsError> {
        let segments = self
            .decoder
            .convert_and_run(&wav_input)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let json = serde_json::to_string(&segments)?;
        Ok(json)
    }
}

fn main() {}
