//! OpenCLIP Vision helper for Stable Video Diffusion.
//! The goal is to expose the ViT-H-14 vision head that `tp/generative-models`
//! wires into `FrozenOpenCLIPImageEmbedder`, keeping the same preprocessing and
//! normalization pipeline (resize → map [-1, 1] to [0, 1] → apply CLIP mean/std).
use candle::{bail, DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use image::{imageops::FilterType, ImageBuffer, Rgb};

use crate::models::clip::text_model::Activation;
use crate::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};

type Rgb32FImage = ImageBuffer<Rgb<f32>, Vec<f32>>;

/// Configuration that mirrors OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`) vision.
#[derive(Clone, Debug)]
pub struct OpenClipVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
}

impl Default for OpenClipVisionConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            embed_dim: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            projection_dim: 1024,
        }
    }
}

impl OpenClipVisionConfig {
    /// Adapter that reuses `ClipVisionTransformer` configuration.
    fn to_clip_config(&self) -> ClipVisionConfig {
        ClipVisionConfig {
            embed_dim: self.embed_dim,
            activation: Activation::QuickGelu,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            projection_dim: self.projection_dim,
            num_channels: 3,
            image_size: self.image_size,
            patch_size: self.patch_size,
        }
    }
}

/// OpenCLIP ViT-H-14 without the projection head. Matches
/// `tp/generative-models/sgm/modules/encoders/modules.py::FrozenOpenCLIPImageEmbedder`.
pub struct OpenClipVisionModel {
    transformer: ClipVisionTransformer,
    config: OpenClipVisionConfig,
    mean: Tensor,
    std: Tensor,
}

impl OpenClipVisionModel {
    /// Builds a CLIP vision transformer and caches the OpenCLIP mean/std buffers.
    pub fn new(vs: VarBuilder, config: OpenClipVisionConfig) -> Result<Self> {
        let clip_config = config.to_clip_config();
        let transformer = ClipVisionTransformer::new(vs.clone(), &clip_config)?;

        let mean = Tensor::from_slice(&[0.48145466f32, 0.4578275, 0.40821073], (3,), vs.device())?
            .reshape((1, 3, 1, 1))?;
        let std = Tensor::from_slice(
            &[0.26862954f32, 0.2613026f32, 0.2757771f32],
            (3,),
            vs.device(),
        )?
        .reshape((1, 3, 1, 1))?;

        Ok(Self {
            transformer,
            config,
            mean,
            std,
        })
    }

    /// Runs the vision encoder and returns the pooled embedding.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let processed = self.preprocess(input)?;
        self.transformer.forward(&processed)
    }

    /// Access the configuration used to build this model.
    pub fn config(&self) -> &OpenClipVisionConfig {
        &self.config
    }

    /// Preprocess the tensor exactly like the OpenCLIP Python reference:
    /// 1. resize to `image_size`,
    /// 2. shift from [-1, 1] to [0, 1],
    /// 3. normalize with the stored mean/std.
    fn preprocess(&self, input: &Tensor) -> Result<Tensor> {
        let resized = self.ensure_image_size(input)?;
        let cast = resized.to_dtype(DType::F32)?;
        let target_device = cast.device().clone();
        let scaled = (cast + 1.0)?;
        let scaled = (scaled / 2.0)?;
        let mean = self.mean.to_device(&target_device)?;
        let std = self.std.to_device(&target_device)?;
        scaled.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    /// Ensures the incoming image tensor is exactly `image_size` × `image_size`.
    fn ensure_image_size(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = input.dims4()?;
        let target = self.config.image_size;
        if height == target && width == target {
            return Ok(input.clone());
        }

        let original_device = input.device().clone();
        let cpu_input = input.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let data = cpu_input.to_vec1::<f32>()?;
        let resized = Self::resize_image_batch(&data, batch, channels, height, width, target)?;
        let tensor = Tensor::from_vec(resized, (batch, channels, target, target), &Device::Cpu)?;
        tensor.to_device(&original_device)
    }

    fn resize_image_batch(
        data: &[f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        target: usize,
    ) -> Result<Vec<f32>> {
        if channels != 3 {
            bail!("OpenCLIP vision encoder expects 3 channels but got {channels} channels");
        }

        let per_image = channels * height * width;
        let mut resized = Vec::with_capacity(batch * channels * target * target);
        for b in 0..batch {
            let start = b * per_image;
            let slice = &data[start..start + per_image];
            let image = Self::rgb_image_from_slice(slice, channels, height, width);
            let scaled = image::imageops::resize(
                &image,
                target as u32,
                target as u32,
                FilterType::CatmullRom,
            );
            let raw = scaled.into_raw();
            for c in 0..channels {
                for y in 0..target {
                    for x in 0..target {
                        let idx = ((y * target + x) * channels) + c;
                        resized.push(raw[idx]);
                    }
                }
            }
        }

        Ok(resized)
    }

    fn rgb_image_from_slice(
        slice: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Rgb32FImage {
        Rgb32FImage::from_fn(width as u32, height as u32, |x, y| {
            let mut pixel = [0.0f32; 3];
            let x = x as usize;
            let y = y as usize;
            for (c, pixel_value) in pixel.iter_mut().enumerate().take(channels) {
                let idx = c * height * width + y * width + x;
                *pixel_value = slice[idx];
            }
            Rgb(pixel)
        })
    }
}
