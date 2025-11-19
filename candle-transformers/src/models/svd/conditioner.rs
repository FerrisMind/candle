//! Conditioner helpers for Stable Video Diffusion.
//! The conditioner gathers embeddings from various modalities (image, motion,
//! timestep) and exposes them as `"key -> Tensor"` pairs that mirror the
//! Python `GeneralConditioner` used in `stabilityai/stable-video-diffusion`.
use candle::{bail, DType, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::Arc;

use super::openclip_vision::OpenClipVisionModel;
use super::video_ae::VideoAutoencoder;

type OpenClipVisionRef = Arc<OpenClipVisionModel>;
type VideoAutoencoderRef = Arc<VideoAutoencoder>;

/// Simple holder for conditioning tensors, conceptually similar to `value_dict`.
#[derive(Default)]
pub struct ConditioningBatch {
    values: HashMap<String, Tensor>,
}

impl ConditioningBatch {
    pub fn insert(&mut self, key: impl Into<String>, tensor: Tensor) {
        self.values.insert(key.into(), tensor);
    }

    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.values.get(key)
    }
}

/// Trait implemented by every conditioning head that transforms one tensor into
/// another.
pub trait ConditionEmbedder: Send + Sync {
    fn input_key(&self) -> &str;
    fn output_key(&self) -> &str;
    fn forward(&self, batch: &ConditioningBatch) -> Result<Tensor>;
}

/// Aggregates the outputs from multiple embedders so the VideoUNet can consume
/// them later.
pub struct GeneralConditioner {
    embedders: Vec<Box<dyn ConditionEmbedder>>,
}

impl GeneralConditioner {
    pub fn new(embedders: Vec<Box<dyn ConditionEmbedder>>) -> Self {
        Self { embedders }
    }

    pub fn forward(&self, batch: &ConditioningBatch) -> Result<HashMap<String, Tensor>> {
        let mut output = HashMap::new();
        for embedder in &self.embedders {
            let tensor = embedder.forward(batch)?;
            output.insert(embedder.output_key().to_string(), tensor);
        }
        Ok(output)
    }
}

/// Embeds raw video frames using a shared OpenCLIP Vision encoder.
pub struct OpenClipImageEmbedder {
    input_key: String,
    output_key: String,
    openclip: OpenClipVisionRef,
}

impl OpenClipImageEmbedder {
    pub fn new(
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        openclip: OpenClipVisionRef,
    ) -> Self {
        Self {
            input_key: input_key.into(),
            output_key: output_key.into(),
            openclip,
        }
    }

    fn embed_frames(&self, tensor: &Tensor) -> Result<Tensor> {
        let (flat, batch, frames) = flatten_temporal_tensor(tensor)?;
        let embeddings = self.openclip.forward(&flat)?;
        let embed_dim = embeddings.dim(D::Minus1)?;
        embeddings.reshape((batch, frames, embed_dim))
    }
}

impl ConditionEmbedder for OpenClipImageEmbedder {
    fn input_key(&self) -> &str {
        &self.input_key
    }

    fn output_key(&self) -> &str {
        &self.output_key
    }

    fn forward(&self, batch: &ConditioningBatch) -> Result<Tensor> {
        let tensor = batch
            .get(&self.input_key)
            .ok_or_else(|| candle::Error::msg("missing conditioning input"))?;
        self.embed_frames(tensor)
    }
}

/// Similar to `OpenClipImageEmbedder` but repeats the embedding for several
/// temporal copies so that the prediction path can be blended.
pub struct OpenClipImagePredictionEmbedder {
    input_key: String,
    output_key: String,
    openclip: OpenClipVisionRef,
    copies: usize,
}

impl OpenClipImagePredictionEmbedder {
    pub fn new(
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        openclip: OpenClipVisionRef,
        copies: usize,
    ) -> Self {
        Self {
            input_key: input_key.into(),
            output_key: output_key.into(),
            openclip,
            copies: copies.max(1),
        }
    }

    fn embed_frames(&self, tensor: &Tensor) -> Result<Tensor> {
        let (flat, batch, frames) = flatten_temporal_tensor(tensor)?;
        let embeddings = self.openclip.forward(&flat)?;
        let embed_dim = embeddings.dim(D::Minus1)?;
        embeddings.reshape((batch, frames, embed_dim))
    }
}

impl ConditionEmbedder for OpenClipImagePredictionEmbedder {
    fn input_key(&self) -> &str {
        &self.input_key
    }

    fn output_key(&self) -> &str {
        &self.output_key
    }

    fn forward(&self, batch: &ConditioningBatch) -> Result<Tensor> {
        let tensor = batch
            .get(&self.input_key)
            .ok_or_else(|| candle::Error::msg("missing conditioning input"))?;
        let embedded = self.embed_frames(tensor)?;

        if self.copies == 1 {
            Ok(embedded)
        } else {
            embedded.repeat((self.copies, 1, 1))
        }
    }
}

/// Sinusoidal embedding for discrete motion tokens (fps, motion bucket).
pub struct ConcatTimestepEmbedderNd {
    input_key: String,
    output_key: String,
    outdim: usize,
}

impl ConcatTimestepEmbedderNd {
    pub fn new(input_key: impl Into<String>, output_key: impl Into<String>, outdim: usize) -> Self {
        Self {
            input_key: input_key.into(),
            output_key: output_key.into(),
            outdim,
        }
    }
}

impl ConditionEmbedder for ConcatTimestepEmbedderNd {
    fn input_key(&self) -> &str {
        &self.input_key
    }

    fn output_key(&self) -> &str {
        &self.output_key
    }

    fn forward(&self, batch: &ConditioningBatch) -> Result<Tensor> {
        let tensor = batch
            .get(&self.input_key)
            .ok_or_else(|| candle::Error::msg("missing conditioning input"))?;
        let matrix = ensure_matrix(tensor)?;
        let (batch_size, dims) = matrix.dims2()?;
        let flattened = matrix.reshape((batch_size * dims,))?;
        let embedded = sinusoidal_embedding(&flattened, self.outdim)?;
        embedded.reshape((batch_size, dims * self.outdim))
    }
}

/// Encodes video frames into latents using the SVD first-stage autoencoder.
pub struct VideoPredictionEmbedderWithEncoder {
    input_key: String,
    output_key: String,
    autoencoder: VideoAutoencoderRef,
    copies: usize,
}

impl VideoPredictionEmbedderWithEncoder {
    pub fn new(
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        autoencoder: VideoAutoencoderRef,
        copies: usize,
    ) -> Self {
        Self {
            input_key: input_key.into(),
            output_key: output_key.into(),
            autoencoder,
            copies: copies.max(1),
        }
    }

    fn align_frames(&self, tensor: &Tensor) -> Result<Tensor> {
        let (flat, batch, frames) = flatten_temporal_tensor(tensor)?;
        let total = batch * frames;
        let target = self.autoencoder.config().num_frames;
        if target == 0 {
            bail!("autoencoder configured with zero frames");
        }

        if total % target == 0 {
            return Ok(flat);
        }

        if total < target {
            let repeats = target.div_ceil(total);
            let repeated = flat.repeat((repeats, 1, 1, 1))?;
            return repeated.i((0..target, .., .., ..));
        }

        bail!("conditioning frames ({total}) must be divisible by {target}");
    }

    fn repeat_if_needed(&self, encoded: Tensor) -> Result<Tensor> {
        if self.copies == 1 {
            Ok(encoded)
        } else {
            encoded.repeat((self.copies, 1, 1, 1, 1))
        }
    }
}

impl ConditionEmbedder for VideoPredictionEmbedderWithEncoder {
    fn input_key(&self) -> &str {
        &self.input_key
    }

    fn output_key(&self) -> &str {
        &self.output_key
    }

    fn forward(&self, batch: &ConditioningBatch) -> Result<Tensor> {
        let tensor = batch
            .get(&self.input_key)
            .ok_or_else(|| candle::Error::msg("missing conditioning input"))?;
        let aligned = self.align_frames(tensor)?;
        let encoded = self.autoencoder.encode(&aligned)?;
        self.repeat_if_needed(encoded)
    }
}

fn flatten_temporal_tensor(input: &Tensor) -> Result<(Tensor, usize, usize)> {
    match input.rank() {
        4 => {
            let (batch, _, _, _) = input.dims4()?;
            Ok((input.clone(), batch, 1))
        }
        5 => {
            let (batch, frames, channels, height, width) = input.dims5()?;
            let flat = input.reshape((batch * frames, channels, height, width))?;
            Ok((flat, batch, frames))
        }
        dims => bail!("expected 4D or 5D tensor but got {dims}D"),
    }
}

fn ensure_matrix(tensor: &Tensor) -> Result<Tensor> {
    match tensor.rank() {
        1 => tensor.unsqueeze(1),
        2 => Ok(tensor.clone()),
        dims => bail!("expected 1D or 2D tensor but got {dims}D"),
    }
}

fn sinusoidal_embedding(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    if dim == 0 {
        bail!("outdim must be positive");
    }
    let half = dim / 2;
    let dev = tensor.device();
    let coords = tensor.to_dtype(DType::F32)?;
    let rows = coords.shape().dims().iter().product::<usize>();
    if half == 0 {
        let mut emb = Tensor::zeros((rows, 0), DType::F32, dev)?;
        if dim % 2 == 1 {
            let zeros = Tensor::zeros((rows, 1), DType::F32, dev)?;
            emb = Tensor::cat(&[emb, zeros], D::Minus1)?;
        }
        return Ok(emb);
    }
    let freq_base = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let scale = (-10000f64.ln()) / half as f64;
    let scale_tensor = Tensor::full(scale as f32, (half,), dev)?;
    let freq = freq_base.broadcast_mul(&scale_tensor)?.exp()?;
    let args = coords.unsqueeze(1)?.broadcast_mul(&freq.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let mut emb = Tensor::cat(&[cos, sin], D::Minus1)?;
    if dim % 2 == 1 {
        let (rows, _) = emb.dims2()?;
        let zeros = Tensor::zeros((rows, 1), DType::F32, dev)?;
        emb = Tensor::cat(&[emb, zeros], D::Minus1)?;
    }
    Ok(emb)
}
