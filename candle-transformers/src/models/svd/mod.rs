//! Stable Video Diffusion (SVD) config shared across the Rust pipeline.
//!
//! This module will eventually mirror the structure of
//! `stabilityai/stable-video-diffusion-img2vid-xt-1-1`, exposing a builder-friendly
//! configuration that keeps track of the dimensions, latent channels, and scale
//! factor used across the video UNet, conditioner, and autoencoder.
//!
//! Step 1.1 introduces the `SvdConfig` configuration entry point and the
//! `svd_xt_1_1` constructor. This type carries the metadata for all video
//! components and exposes helper methods that build the corresponding submodules.

pub mod checkpoint;
pub mod conditioner;
pub mod edm;
pub mod openclip_vision;
pub mod video_ae;
pub mod video_attention;
pub mod video_unet;

pub use checkpoint::{SvdCheckpointAnalysis, SvdCheckpointComponent};
pub use video_attention::{
    SpatialVideoTransformer, SpatialVideoTransformerConfig, VideoTransformerBlock,
};
pub use video_unet::{AE3DConv, AlphaBlender, MergeStrategy, TimeConv, VideoResBlock};

use candle::Result;
use candle_nn::VarBuilder;
use std::sync::Arc;

use self::conditioner::{
    ConcatTimestepEmbedderNd, ConditionEmbedder, GeneralConditioner, OpenClipImageEmbedder,
    OpenClipImagePredictionEmbedder, VideoPredictionEmbedderWithEncoder,
};
use self::edm::{EdmDiscretizationConfig, EulerEdmSampler, LinearPredictionGuider};
use self::openclip_vision::{OpenClipVisionConfig, OpenClipVisionModel};
use self::video_ae::{VideoAutoencoder, VideoAutoencoderConfig};
use self::video_unet::{VideoUnet, VideoUnetConfig};

/// Shared configuration for a single SVD variant.
#[derive(Clone, Debug)]
pub struct SvdConfig {
    pub width: usize,
    pub height: usize,
    pub num_frames: usize,
    pub scale_factor: f64,
    pub video_unet: VideoUnetConfig,
    pub video_ae: VideoAutoencoderConfig,
    pub openclip: OpenClipVisionConfig,
    pub edm: EdmDiscretizationConfig,
}

impl SvdConfig {
    const DEFAULT_WIDTH: usize = 1024;
    const DEFAULT_HEIGHT: usize = 576;
    const DEFAULT_NUM_FRAMES: usize = 25;
    const DEFAULT_SCALE_FACTOR: f64 = 0.18215;

    /// Returns the configuration shared by the `svd_xt_1_1` variant documented
    /// in `tp/generative-models/scripts/sampling/configs/svd_xt_1_1.yaml`.
    pub fn svd_xt_1_1(
        width: Option<usize>,
        height: Option<usize>,
        num_frames: Option<usize>,
    ) -> Self {
        let width = width.unwrap_or(Self::DEFAULT_WIDTH);
        let height = height.unwrap_or(Self::DEFAULT_HEIGHT);
        let num_frames = num_frames.unwrap_or(Self::DEFAULT_NUM_FRAMES);
        let video_unet = VideoUnetConfig {
            num_frames,
            ..Default::default()
        };
        let video_ae = VideoAutoencoderConfig {
            num_frames,
            ..Default::default()
        };
        Self {
            width,
            height,
            num_frames,
            scale_factor: Self::DEFAULT_SCALE_FACTOR,
            video_unet,
            video_ae,
            openclip: OpenClipVisionConfig::default(),
            edm: EdmDiscretizationConfig::default(),
        }
    }

    pub fn latent_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.num_frames,
            self.video_ae.z_channels,
            self.height / 8,
            self.width / 8,
        )
    }

    pub fn build_video_unet(&self, vs: VarBuilder) -> Result<VideoUnet> {
        VideoUnet::new(vs, self.video_unet.clone())
    }

    pub fn build_autoencoder(&self, vs: VarBuilder) -> Result<VideoAutoencoder> {
        VideoAutoencoder::new(vs, self.video_ae.clone())
    }

    pub fn build_openclip(&self, vs: VarBuilder) -> Result<OpenClipVisionModel> {
        OpenClipVisionModel::new(vs, self.openclip.clone())
    }

    /// Constructs the `GeneralConditioner` layout defined in
    /// `tp/generative-models/scripts/sampling/configs/svd_xt_1_1.yaml`, mirroring the
    /// OpenCLIP predictor + timestep embedder stack.
    pub fn build_conditioner(
        &self,
        video_autoencoder: std::sync::Arc<VideoAutoencoder>,
        openclip: std::sync::Arc<OpenClipVisionModel>,
    ) -> GeneralConditioner {
        let embedders: Vec<Box<dyn ConditionEmbedder>> = vec![
            Box::new(OpenClipImagePredictionEmbedder::new(
                "cond_frames_without_noise",
                "cond_pred",
                openclip.clone(),
                1,
            )),
            Box::new(ConcatTimestepEmbedderNd::new("fps_id", "fps_vector", 256)),
            Box::new(ConcatTimestepEmbedderNd::new(
                "motion_bucket_id",
                "motion_vector",
                256,
            )),
            Box::new(VideoPredictionEmbedderWithEncoder::new(
                "cond_frames",
                "cond_latents",
                video_autoencoder.clone(),
                1,
            )),
            Box::new(ConcatTimestepEmbedderNd::new(
                "cond_aug",
                "cond_aug_vector",
                256,
            )),
            Box::new(OpenClipImageEmbedder::new(
                "cond_frames",
                "cond_vision",
                openclip.clone(),
            )),
        ];
        GeneralConditioner::new(embedders)
    }

    /// Instantiates the sampler described in the `sampler_config` block of the
    /// upstream `svd_xt_1_1` YAML reference, pairing EDM discretization with the
    /// linear prediction guider.
    pub fn build_edm_sampler(&self) -> EulerEdmSampler {
        let guider = Arc::new(LinearPredictionGuider::new(
            3.0,
            self.num_frames,
            1.5,
            Vec::new(),
        ));
        EulerEdmSampler::new(self.edm.clone(), guider)
    }
}
