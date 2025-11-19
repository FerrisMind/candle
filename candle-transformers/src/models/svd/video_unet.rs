//! Video-aware UNet for Stable Video Diffusion.
//!
//! This module keeps the existing 2D `UNet2DConditionModel` as a backbone and
//! incorporates a lightweight temporal stack that mixes the spatial output along
//! the time axis using 1D convolutions and a learnable gating mechanism.
use candle::{bail, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, layer_norm, linear, ops, Conv1d, Conv1dConfig, Init, LayerNorm, Linear, VarBuilder,
};

use std::collections::HashSet;

use crate::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeStrategy {
    Fixed,
    Learned,
    LearnedWithImages,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::LearnedWithImages
    }
}

#[derive(Debug)]
struct AlphaBlender {
    strategy: MergeStrategy,
    mix_factor: Tensor,
}

impl AlphaBlender {
    fn new(vs: VarBuilder, strategy: MergeStrategy, mix_factor: f64) -> Result<Self> {
        let mix_factor = match strategy {
            MergeStrategy::Fixed => Tensor::full(mix_factor, (1,), vs.device())?,
            MergeStrategy::Learned | MergeStrategy::LearnedWithImages => {
                vs.get_with_hints((1,), "mix_factor", Init::Const(mix_factor))?
            }
        };
        Ok(Self {
            strategy,
            mix_factor,
        })
    }

    fn compute_alpha(&self, image_only_indicator: Option<&Tensor>) -> Result<Tensor> {
        match self.strategy {
            MergeStrategy::Fixed => Ok(self.mix_factor.clone()),
            MergeStrategy::Learned => ops::sigmoid(&self.mix_factor),
            MergeStrategy::LearnedWithImages => {
                let indicator = match image_only_indicator {
                    Some(t) => t,
                    None => {
                        bail!("learned_with_images merge strategy requires image_only_indicator")
                    }
                };
                let dtype = self.mix_factor.dtype();
                let indicator = indicator.to_dtype(dtype)?;
                let device = indicator.device().clone();
                let (batch, time) = indicator.dims2()?;
                let batch_time = batch * time;
                let mask = indicator
                    .reshape((batch_time,))?
                    .gt(0.0)?
                    .to_dtype(dtype)?
                    .reshape((batch_time, 1, 1))?;
                let mix_alpha = ops::sigmoid(&self.mix_factor)?
                    .reshape((1, 1, 1))?
                    .to_device(&device)?
                    .expand(mask.shape().clone())?;
                let ones = Tensor::ones(mask.shape().clone(), dtype, &device)?;
                let one_minus_mask = ones.broadcast_sub(&mask)?;
                mask.broadcast_add(&one_minus_mask.broadcast_mul(&mix_alpha)?)
            }
        }
    }

    fn blend(
        &self,
        x_spatial: &Tensor,
        x_temporal: &Tensor,
        image_only_indicator: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = x_spatial.device().clone();
        let dtype = x_spatial.dtype();
        let base_alpha = self.compute_alpha(image_only_indicator)?;
        let alpha = base_alpha.to_dtype(dtype)?.to_device(&device)?;
        let one = Tensor::ones(alpha.shape().clone(), dtype, &device)?;
        let one_minus_alpha = one.broadcast_sub(&alpha)?;
        let spatial = alpha.broadcast_mul(x_spatial)?;
        let temporal = one_minus_alpha.broadcast_mul(x_temporal)?;
        spatial.broadcast_add(&temporal)
    }
}

#[derive(Debug)]
struct TemporalStack {
    layer_norm: LayerNorm,
    conv1: Conv1d,
    conv2: Conv1d,
    temb_proj: Linear,
}

impl TemporalStack {
    fn new(
        vs: VarBuilder,
        channels: usize,
        temb_channels: usize,
        kernel_size: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        let layer_norm = layer_norm(channels, norm_eps, vs.pp("norm"))?;
        let conv1 = conv1d(
            channels,
            channels,
            kernel_size,
            Conv1dConfig {
                padding: kernel_size / 2,
                ..Default::default()
            },
            vs.pp("conv1"),
        )?;
        let conv2 = conv1d(
            channels,
            channels,
            1,
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
            vs.pp("conv2"),
        )?;
        let temb_proj = linear(temb_channels, channels, vs.pp("temb_proj"))?;
        Ok(Self {
            layer_norm,
            conv1,
            conv2,
            temb_proj,
        })
    }

    fn forward(&self, xs: &Tensor, temb: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (batch_time, channels, height, width) = xs.dims4()?;
        if num_frames == 0 {
            bail!("num_frames must be positive");
        }
        if batch_time % num_frames != 0 {
            bail!("input batch ({batch_time}) is not divisible by num_frames ({num_frames})");
        }
        let batch = batch_time / num_frames;
        let xs = xs.reshape((batch, num_frames, channels, height, width))?;
        let normalized = xs
            .permute((0, 1, 3, 4, 2))?
            .reshape((batch * num_frames * height * width, channels))?;
        let normalized = self.layer_norm.forward(&normalized)?;
        let normalized = normalized
            .reshape((batch, num_frames, height, width, channels))?
            .permute((0, 1, 4, 2, 3))?;
        let temb_proj = self.temb_proj.forward(temb)?;
        let temb_proj = temb_proj
            .reshape((batch, num_frames, channels))?
            .unsqueeze(D::Minus1)?
            .unsqueeze(D::Minus1)?;
        let fused = normalized.broadcast_add(&temb_proj)?;
        let conv_input = fused.permute((0, 3, 4, 2, 1))?.reshape((
            batch * height * width,
            channels,
            num_frames,
        ))?;
        let conv1 = self.conv1.forward(&conv_input)?;
        let activated = ops::silu(&conv1)?;
        let conv2 = self.conv2.forward(&activated)?;
        let conv_output = conv2
            .reshape((batch, height, width, channels, num_frames))?
            .permute((0, 4, 3, 1, 2))?
            .reshape((batch_time, channels, height, width))?;
        Ok(conv_output)
    }
}

fn build_unet_blocks(config: &VideoUnetConfig) -> Vec<BlockConfig> {
    let mut blocks = Vec::with_capacity(config.channel_mult.len());
    let mut ds = 1usize;
    let attention_set = config
        .attention_resolutions
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for (index, &mult) in config.channel_mult.iter().enumerate() {
        let out_channels = config.model_channels * mult;
        let use_cross_attn = if config.use_cross_attn && attention_set.contains(&ds) {
            Some(config.transformer_layers_per_block)
        } else {
            None
        };
        blocks.push(BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim: config.attention_head_dim,
        });
        if index < config.channel_mult.len() - 1 {
            ds *= 2;
        }
    }
    blocks
}

#[derive(Clone, Debug)]
pub struct VideoUnetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub channel_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attention_resolutions: Vec<usize>,
    pub use_cross_attn: bool,
    pub attention_head_dim: usize,
    pub transformer_layers_per_block: usize,
    pub use_linear_projection: bool,
    pub cross_attention_dim: usize,
    pub norm_eps: f64,
    pub norm_num_groups: usize,
    pub use_flash_attn: bool,
    pub sliced_attention_size: Option<usize>,
    pub num_frames: usize,
    pub temporal_kernel_size: usize,
    pub temporal_norm_eps: f64,
    pub merge_factor: f64,
    pub merge_strategy: MergeStrategy,
}

impl Default for VideoUnetConfig {
    fn default() -> Self {
        Self {
            in_channels: 8,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attention_resolutions: vec![4, 2, 1],
            use_cross_attn: true,
            attention_head_dim: 64,
            transformer_layers_per_block: 1,
            use_linear_projection: true,
            cross_attention_dim: 1024,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            use_flash_attn: false,
            sliced_attention_size: None,
            num_frames: 25,
            temporal_kernel_size: 3,
            temporal_norm_eps: 1e-5,
            merge_factor: 0.5,
            merge_strategy: MergeStrategy::LearnedWithImages,
        }
    }
}

pub struct VideoUnet {
    inner: UNet2DConditionModel,
    config: VideoUnetConfig,
    temporal_stack: TemporalStack,
    temporal_mixer: AlphaBlender,
}

impl VideoUnet {
    pub fn new(vs: VarBuilder, config: VideoUnetConfig) -> Result<Self> {
        let blocks = build_unet_blocks(&config);
        let unet_config = UNet2DConditionModelConfig {
            blocks,
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            layers_per_block: config.num_res_blocks,
            downsample_padding: 1,
            mid_block_scale_factor: 1.0,
            norm_eps: config.norm_eps,
            norm_num_groups: config.norm_num_groups,
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
            cross_attention_dim: config.cross_attention_dim,
        };
        let inner = UNet2DConditionModel::new(
            vs.clone(),
            config.in_channels,
            config.out_channels,
            config.use_flash_attn,
            unet_config,
        )?;
        let time_embed_dim = config.model_channels * 4;
        let temporal_vs = vs.pp("temporal");
        let temporal_stack = TemporalStack::new(
            temporal_vs.pp("stack"),
            config.out_channels,
            time_embed_dim,
            config.temporal_kernel_size,
            config.temporal_norm_eps,
        )?;
        let temporal_mixer = AlphaBlender::new(
            temporal_vs.pp("mixer"),
            config.merge_strategy,
            config.merge_factor,
        )?;
        Ok(Self {
            inner,
            config,
            temporal_stack,
            temporal_mixer,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        image_only_indicator: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_time, _, _, _) = input.dims4()?;
        let num_frames = self.config.num_frames;
        if num_frames == 0 {
            bail!("video UNet configured with zero frames");
        }
        if batch_time % num_frames != 0 {
            bail!("input batch ({batch_time}) is not divisible by num_frames ({num_frames})");
        }
        let temb = self.inner.compute_time_embedding(
            batch_time,
            timestep,
            input.dtype(),
            input.device().clone(),
        )?;
        let spatial_output = self.inner.forward(input, timestep, encoder_hidden_states)?;
        let temporal_output = self
            .temporal_stack
            .forward(&spatial_output, &temb, num_frames)?;
        self.temporal_mixer
            .blend(&spatial_output, &temporal_output, image_only_indicator)
    }

    pub fn config(&self) -> &VideoUnetConfig {
        &self.config
    }
}
