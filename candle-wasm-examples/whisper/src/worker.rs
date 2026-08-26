//! Whisper WASM worker / decoder.
//!
//! **Yew agent limitation (slice 1):** `yew_agent::Worker::handle_input` is synchronous, so the
//! Yew path only loads on CPU (`DeviceMode::Cpu`, and `Auto` ≡ CPU). Explicit `Wgpu` returns
//! [`WorkerOutput::DeviceError`]. Prefer the JS lib worker (`bin/m.rs` async `load_with_device`)
//! for portable WebGPU.

use crate::languages::LANGUAGES;
use anyhow::Error as E;
use candle::{safetensors::Load, DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
pub use candle_transformers::models::whisper::{self as m, Config};
use candle_wasm_device_select::{DeviceMode, ResolvedKind};
use rand::{distr::Distribution, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

pub const DTYPE: DType = DType::F32;

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    language: Option<String>,
    is_multilingual: bool,
    mel_filters: Vec<f32>,
    timestamps: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    /// Device used for all tensors / inference (must match load).
    device: Device,
    resolved: ResolvedKind,
    adapter_name: Option<String>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        device: Device,
        resolved: ResolvedKind,
        adapter_name: Option<String>,
        task: Option<Task>,
        language: Option<String>,
        is_multilingual: bool,
        timestamps: bool,
    ) -> anyhow::Result<Self> {
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        let seed = 299792458;
        Ok(Self {
            model,
            rng: StdRng::seed_from_u64(seed),
            tokenizer,
            mel_filters,
            task,
            timestamps,
            language,
            is_multilingual,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            device,
            resolved,
            adapter_name,
        })
    }

    /// Label for JS / UI: `"cpu"` or `"wgpu"`.
    pub fn resolved_device(&self) -> &'static str {
        match self.resolved {
            ResolvedKind::Cpu => "cpu",
            ResolvedKind::Wgpu => "wgpu",
        }
    }

    pub fn adapter_name(&self) -> Option<&str> {
        self.adapter_name.as_deref()
    }

    pub fn resolved_kind(&self) -> ResolvedKind {
        self.resolved
    }

    async fn decode(&mut self, mel: &Tensor, t: f64) -> anyhow::Result<DecodingResult> {
        let model = &mut self.model;
        let language_token = match (self.is_multilingual, &self.language) {
            (true, None) => Some(detect_language(model, &self.tokenizer, mel).await?),
            (false, None) => None,
            (true, Some(language)) => {
                match token_id(&self.tokenizer, &format!("<|{:?}|>", self.language)) {
                    Ok(token_id) => Some(token_id),
                    Err(_) => anyhow::bail!("language {language} is not supported"),
                }
            }
            (false, Some(_)) => {
                anyhow::bail!("a language cannot be set for non-multilingual models")
            }
        };

        let audio_features = model.encoder_forward(mel, true)?;
        println!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // At each token boundary yield to the JS event loop so the wgpu backend can
            // `on_submitted_work_done` and recycle its storage buffers back into the pool.
            // This is what prevents the wasm wgpu VRAM balloon + garbage output. On the CPU
            // (and dummy-wgpu) path this arm is a no-op.
            if let Device::Wgpu(dev) = &self.device {
                dev.synchronize_async().await?;
            }

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar_async::<f32>().await? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1_async().await?;
                let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1_async().await?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar_async::<f32>().await? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    async fn decode_with_fallback(&mut self, segment: &Tensor) -> anyhow::Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, _> = self.decode(segment, t).await;
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    console_log!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    async fn run(&mut self, mel: &Tensor) -> anyhow::Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment).await?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                console_log!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            console_log!("{seek}: {segment:?}");
            segments.push(segment)
        }
        Ok(segments)
    }

    /// Load weights onto a resolved device (shared by async resolve and Yew CPU path).
    fn load_on_device(
        md: ModelData,
        device: Device,
        resolved: ResolvedKind,
        adapter_name: Option<String>,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_bytes(&md.tokenizer).map_err(E::msg)?;

        let mel_filters = safetensors::tensor::SafeTensors::deserialize(&md.mel_filters)?;
        let mel_filters = mel_filters.tensor("mel_80")?.load(&Device::Cpu)?;
        console_log!("loaded mel filters {:?}", mel_filters.shape());
        let mel_filters = mel_filters.flatten_all()?.to_vec1::<f32>()?;
        let config: Config = serde_json::from_slice(&md.config)?;
        let model = if md.quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
                &md.weights,
                &device,
            )?;
            Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
        } else {
            let vb = VarBuilder::from_buffered_safetensors(md.weights, m::DTYPE, &device)?;
            Model::Normal(m::model::Whisper::load(&vb, config)?)
        };
        console_log!(
            "done loading model on {}{}",
            match resolved {
                ResolvedKind::Cpu => "cpu",
                ResolvedKind::Wgpu => "wgpu",
            },
            adapter_name
                .as_ref()
                .map(|n| format!(" ({n})"))
                .unwrap_or_default()
        );

        let task = match md.task.as_deref() {
            Some("translate") => Some(Task::Translate),
            _ => Some(Task::Transcribe),
        };

        Self::new(
            model,
            tokenizer,
            mel_filters,
            device,
            resolved,
            adapter_name,
            task,
            md.language,
            md.is_multilingual,
            md.timestamps,
        )
    }

    /// Async load: resolve [`ModelData::device_mode`] then build the decoder on that device.
    pub async fn load(md: ModelData) -> anyhow::Result<Self> {
        let resolved = md
            .device_mode
            .resolve()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Self::load_on_device(md, resolved.device, resolved.resolved, resolved.adapter_name)
    }

    /// Synchronous CPU-only load for the Yew agent (cannot await wgpu init).
    pub fn load_cpu(md: ModelData) -> anyhow::Result<Self> {
        Self::load_on_device(md, Device::Cpu, ResolvedKind::Cpu, None)
    }

    pub async fn convert_and_run(&mut self, wav_input: &[u8]) -> anyhow::Result<Vec<Segment>> {
        let device = &self.device;
        let mut wav_input = std::io::Cursor::new(wav_input);
        let wav_reader = hound::WavReader::new(&mut wav_input)?;
        let spec = wav_reader.spec();
        console_log!("loaded wav data: {spec:?}");
        if spec.sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!("wav file must have a {} sampling rate", m::SAMPLE_RATE);
        }
        let mut data = wav_reader.into_samples::<i16>().collect::<Vec<_>>();
        data.truncate(data.len() / spec.channels as usize);
        let mut pcm_data = Vec::with_capacity(data.len());
        for d in data.into_iter() {
            let d = d?;
            pcm_data.push(d as f32 / 32768.)
        }
        console_log!("pcm data loaded {}", pcm_data.len());
        let mel = crate::audio::pcm_to_mel(self.model.config(), &pcm_data, &self.mel_filters)?;
        let mel_len = mel.len();
        let n_mels = self.model.config().num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, n_mels, mel_len / n_mels), device)?;
        console_log!("loaded mel: {:?}", mel.dims());
        let segments = self.run(&mel).await?;
        Ok(segments)
    }
}

/// Returns the token id for the selected language.
pub async fn detect_language(model: &mut Model, tokenizer: &Tokenizer, mel: &Tensor) -> Result<u32, E> {
    console_log!("detecting language");
    let (_bsize, _, seq_len) = mel.dims3()?;
    let mel = mel.narrow(
        2,
        0,
        usize::min(seq_len, model.config().max_source_positions),
    )?;
    let device = mel.device();

    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")))
        .map(|e| e.map_err(E::msg))
        .collect::<Result<Vec<_>, E>>()?;

    let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
    let audio_features = model.encoder_forward(&mel, true)?;
    let tokens = Tensor::new(&[[sot_token]], device)?;
    let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;
    let ys = model.decoder_forward(&tokens, &audio_features, true)?;
    let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
    let logits = logits.index_select(&language_token_ids, 0)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let probs = probs.to_vec1_async::<f32>().await?;
    let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
    probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs.iter().take(5) {
        println!("{language}: {p}")
    }
    let token = &format!("<|{}|>", probs[0].0 .0);
    let language = token_id(tokenizer, token)?;
    console_log!("detected language: {language} {token}");
    Ok(language)
}
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transferred via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub mel_filters: Vec<u8>,
    pub config: Vec<u8>,
    pub quantized: bool,
    pub timestamps: bool,
    pub is_multilingual: bool,
    pub language: Option<String>,
    pub task: Option<String>,
    /// Preferred device for load / setDevice. Inference (`DecodeTask`) does not carry a mode.
    pub device_mode: DeviceMode,
}

pub struct Worker {
    link: WorkerLink<Self>,
    decoder: Rc<RefCell<Option<Decoder>>>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    /// Load (or reload) weights for the given device mode. Yew path: Cpu / Auto→Cpu only.
    SetDevice { mode: DeviceMode, model: ModelData },
    /// Run decode on the currently loaded decoder. Must not include device mode.
    DecodeTask { wav_bytes: Vec<u8> },
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    WeightsLoaded {
        /// `"cpu"` or `"wgpu"`.
        resolved: String,
        adapter_name: Option<String>,
    },
    Decoded(Vec<Segment>),
    DeviceError {
        message: String,
        requested: DeviceMode,
    },
}

fn resolved_label(kind: ResolvedKind) -> String {
    match kind {
        ResolvedKind::Cpu => "cpu".to_string(),
        ResolvedKind::Wgpu => "wgpu".to_string(),
    }
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self {
            link,
            decoder: Rc::new(RefCell::new(None)),
        }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        // `SetDevice` stays fully synchronous (CPU-only Yew path): load + respond inline.
        // `DecodeTask` must drive the now-async decode, so it takes the decoder out and
        // spawns a local future that responds only once the decode completes.
        match msg {
            WorkerInput::SetDevice { mode, mut model } => {
                model.device_mode = mode;
                let output = match mode {
                    // Sync Yew agent cannot await wgpu; Cpu and Auto load on CPU.
                    DeviceMode::Cpu | DeviceMode::Auto => match Decoder::load_cpu(model) {
                        Ok(decoder) => {
                            let resolved = resolved_label(decoder.resolved_kind());
                            let adapter_name = decoder.adapter_name().map(str::to_string);
                            *self.decoder.borrow_mut() = Some(decoder);
                            Ok(WorkerOutput::WeightsLoaded {
                                resolved,
                                adapter_name,
                            })
                        }
                        Err(err) => Err(format!("model creation error {err:?}")),
                    },
                    DeviceMode::Wgpu => Ok(WorkerOutput::DeviceError {
                        message: "Yew agent path is CPU-only in slice 1 (sync handle_input cannot await WebGPU init); use the JS lib worker (m.wasm load_with_device) for wgpu".to_string(),
                        requested: DeviceMode::Wgpu,
                    }),
                };
                self.link.respond(id, output);
            }
            WorkerInput::DecodeTask { wav_bytes } => {
                // Take the decoder out so the spawned future owns it (`'static`). The slot is
                // left `None` for the duration of the decode and restored afterwards, so the
                // `RefCell` is never held borrowed across an await point.
                let decoded = self.decoder.borrow_mut().take();
                let Some(mut decoder) = decoded else {
                    self.link
                        .respond(id, Err("model has not been set".to_string()));
                    return;
                };
                let slot = Rc::clone(&self.decoder);
                let link = self.link.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = decoder.convert_and_run(&wav_bytes).await;
                    *slot.borrow_mut() = Some(decoder);
                    let output = match result {
                        Ok(segments) => Ok(WorkerOutput::Decoded(segments)),
                        Err(e) => Err(e.to_string()),
                    };
                    link.respond(id, output);
                });
            }
        }
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}
