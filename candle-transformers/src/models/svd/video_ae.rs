//! Video-first-stage autoencoder for Stable Video Diffusion.
//! The architecture parallels `stabilityai/stable-video-diffusion-img2vid-xt-1-1` by:
//!  1. Reusing the image autoencoder to encode individual frames.
//!  2. Adding a video-aware decoder that blends spatial features with temporal
//!     convolutions via gated mixing layers.
//!
//! Temporal mixing is controlled through `VideoResBlock` and the final `AE3DConv`.
use crate::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use crate::models::svd::video_unet::MergeStrategy;
use candle::{bail, Module, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::ops;
use candle_nn::{
    conv1d, conv2d, group_norm, linear, Conv1d, Conv2d, GroupNorm, Init, Linear, VarBuilder,
};
use std::collections::HashSet;

const DEFAULT_NORM_EPS: f64 = 1e-6;

/// Simple scaled dot-product attention used by `AttnBlock`.
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / f64::sqrt(dim as f64);
    let attn = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn = ops::softmax(&attn, D::Minus1)?;
    attn.matmul(v)
}

#[derive(Debug)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    dropout: f64,
    temb_proj: Option<Linear>,
    nin_shortcut: Option<Conv2d>,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(
        vs: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        dropout: f64,
        use_conv_shortcut: bool,
    ) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(32, in_channels, DEFAULT_NORM_EPS, vs.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg, vs.pp("conv1"))?;
        let norm2 = group_norm(32, out_channels, DEFAULT_NORM_EPS, vs.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg, vs.pp("conv2"))?;
        let temb_proj = if temb_channels > 0 {
            Some(linear(temb_channels, out_channels, vs.pp("temb_proj"))?)
        } else {
            None
        };
        let nin_shortcut = if in_channels != out_channels && !use_conv_shortcut {
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                Default::default(),
                vs.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        let conv_shortcut = if use_conv_shortcut && in_channels != out_channels {
            Some(conv2d(
                in_channels,
                out_channels,
                3,
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            dropout,
            temb_proj,
            nin_shortcut,
            conv_shortcut,
        })
    }

    fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let shortcut = if let Some(conv_shortcut) = &self.conv_shortcut {
            conv_shortcut.forward(xs)?
        } else if let Some(conv) = &self.nin_shortcut {
            conv.forward(xs)?
        } else {
            xs.clone()
        };
        let xs = self.norm1.forward(xs)?;
        let xs = self.conv1.forward(&ops::silu(&xs)?)?;
        let xs = if let (Some(temb), Some(time_proj)) = (temb, &self.temb_proj) {
            let emb = time_proj.forward(&ops::silu(temb)?)?;
            emb.unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
                .broadcast_add(&xs)?
        } else {
            xs
        };
        let xs = self.norm2.forward(&xs)?;
        let xs = self.conv2.forward(&ops::silu(&xs)?)?;
        let drop = if self.dropout > 0.0 {
            Some(ops::dropout(&xs, self.dropout as f32)?)
        } else {
            None
        };
        let sum = if let Some(drop) = drop {
            drop.broadcast_add(&shortcut)?
        } else {
            xs.broadcast_add(&shortcut)?
        };
        Ok(sum)
    }
}

#[derive(Debug)]
struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    fn new(vs: VarBuilder, channels: usize) -> Result<Self> {
        let norm = group_norm(32, channels, DEFAULT_NORM_EPS, vs.pp("norm"))?;
        let q = conv2d(channels, channels, 1, Default::default(), vs.pp("q"))?;
        let k = conv2d(channels, channels, 1, Default::default(), vs.pp("k"))?;
        let v = conv2d(channels, channels, 1, Default::default(), vs.pp("v"))?;
        let proj_out = conv2d(channels, channels, 1, Default::default(), vs.pp("proj_out"))?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let res = xs.clone();
        let xs = self.norm.forward(xs)?;
        let q = self.q.forward(&xs)?;
        let k = self.k.forward(&xs)?;
        let v = self.v.forward(&xs)?;
        let (batch, channels, height, width) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        let attn = attn
            .squeeze(1)?
            .t()?
            .reshape((batch, channels, height, width))?;
        let projected = self.proj_out.forward(&attn)?;
        let sum = projected.broadcast_add(&res)?;
        Ok(sum)
    }
}

#[derive(Debug)]
struct Upsample {
    conv: Option<Conv2d>,
}

impl Upsample {
    fn new(vs: VarBuilder, in_channels: usize, with_conv: bool) -> Result<Self> {
        let conv = if with_conv {
            Some(conv2d(
                in_channels,
                in_channels,
                3,
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("conv"),
            )?)
        } else {
            None
        };
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = xs.dims4()?;
        let mut xs = xs.upsample_nearest2d(2 * height, 2 * width)?;
        if let Some(conv) = &self.conv {
            xs = conv.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
struct TimeConv {
    conv: Conv1d,
}

impl TimeConv {
    fn new(vs: VarBuilder, channels: usize, kernel_size: usize) -> Result<Self> {
        let cfg = nn::Conv1dConfig {
            padding: kernel_size / 2,
            groups: channels,
            ..Default::default()
        };
        let conv = conv1d(channels, channels, kernel_size, cfg, vs)?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (batch_time, channels, height, width) = xs.dims4()?;
        if num_frames == 0 {
            bail!("num_frames cannot be zero");
        }
        let batch = batch_time / num_frames;
        let xs = xs.reshape((batch, num_frames, channels, height, width))?;
        let xs = xs.permute((0, 3, 4, 2, 1))?; // (batch, h, w, channels, frames)
        let xs = xs.reshape((batch * height * width, channels, num_frames))?;
        let xs = self.conv.forward(&xs)?;
        let xs = xs.reshape((batch, height, width, channels, num_frames))?;
        let xs = xs.permute((0, 4, 3, 1, 2))?; // (batch, frames, channels, h, w)
        let xs = xs.reshape((batch * num_frames, channels, height, width))?;
        Ok(xs)
    }
}

#[derive(Debug, Clone, Copy)]
enum MergePolicy {
    Fixed,
    Learned,
}

impl MergePolicy {
    fn compute_alpha(&self, factor: &Tensor) -> Result<Tensor> {
        match self {
            MergePolicy::Fixed => Ok(factor.clone()),
            MergePolicy::Learned => ops::sigmoid(factor),
        }
    }
}

#[derive(Debug)]
struct VideoResBlock {
    spatial: ResnetBlock,
    time_conv: TimeConv,
    alpha: Tensor,
    merge_policy: MergePolicy,
}

impl VideoResBlock {
    fn new(
        vs: VarBuilder,
        channels: usize,
        video_kernel: usize,
        merge_policy: MergePolicy,
        dropout: f64,
        alpha_init: f64,
    ) -> Result<Self> {
        let spatial = ResnetBlock::new(vs.pp("spatial"), channels, channels, 0, dropout, false)?;
        let time_conv = TimeConv::new(vs.pp("time"), channels, video_kernel)?;
        let alpha = vs.get_with_hints((1,), "alpha", Init::Const(alpha_init))?;
        Ok(Self {
            spatial,
            time_conv,
            alpha,
            merge_policy,
        })
    }

    fn forward(&self, xs: &Tensor, num_frames: usize) -> Result<Tensor> {
        let spatial = self.spatial.forward(xs, None)?;
        let temporal = self.time_conv.forward(&spatial, num_frames)?;
        let alpha = self.merge_policy.compute_alpha(&self.alpha)?;
        let device = alpha.device();
        let one = Tensor::ones(alpha.shape().clone(), alpha.dtype(), device)?;
        let one_minus_alpha = one.broadcast_sub(&alpha)?;
        let alpha_term = alpha.broadcast_mul(&temporal)?;
        let spatial_term = one_minus_alpha.broadcast_mul(&spatial)?;
        let output = alpha_term.broadcast_add(&spatial_term)?;
        Ok(output)
    }
}

#[derive(Debug)]
struct AE3DConv {
    conv: Conv2d,
    time_conv: TimeConv,
}

impl AE3DConv {
    fn new(vs: VarBuilder, channels: usize, kernel: usize) -> Result<Self> {
        let conv = conv2d(
            channels,
            channels,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv"),
        )?;
        let time_conv = TimeConv::new(vs.pp("time"), channels, kernel)?;
        Ok(Self { conv, time_conv })
    }

    fn forward(&self, xs: &Tensor, num_frames: usize) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        self.time_conv.forward(&xs, num_frames)
    }
}

#[derive(Debug)]
struct VideoUpBlock {
    blocks: Vec<VideoResBlock>,
    attns: Vec<AttnBlock>,
    upsample: Option<Upsample>,
    add_attn: bool,
}

#[derive(Debug, Clone, Copy)]
struct VideoUpBlockConfig {
    num_blocks: usize,
    in_channels: usize,
    out_channels: usize,
    add_attn: bool,
    upsample: bool,
    video_kernel: usize,
    merge_policy: MergePolicy,
    dropout: f64,
}

impl VideoUpBlock {
    fn new(vs: VarBuilder, config: VideoUpBlockConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.num_blocks);
        let vs_blocks = vs.pp("blocks");
        for idx in 0..config.num_blocks {
            blocks.push(VideoResBlock::new(
                vs_blocks.pp(idx.to_string()),
                if idx == 0 {
                    config.in_channels
                } else {
                    config.out_channels
                },
                config.video_kernel,
                config.merge_policy,
                config.dropout,
                0.0,
            )?);
        }
        let attns = if config.add_attn {
            (0..config.num_blocks)
                .map(|idx| AttnBlock::new(vs.pp("attns").pp(idx.to_string()), config.out_channels))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let upsample = if config.upsample {
            Some(Upsample::new(vs.pp("upsample"), config.out_channels, true)?)
        } else {
            None
        };
        Ok(Self {
            blocks,
            attns,
            upsample,
            add_attn: config.add_attn,
        })
    }
}

#[derive(Debug)]
pub enum TimeMode {
    ConvOnly,
    AttnOnly,
    All,
}

#[derive(Debug)]
pub struct VideoDecoderConfig {
    pub ch: usize,
    pub out_ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub dropout: f64,
    pub resolution: usize,
    pub z_channels: usize,
    pub video_kernel_size: usize,
    pub alpha: f64,
    pub merge_strategy: MergeStrategy,
    pub time_mode: TimeMode,
    pub num_frames: usize,
}

impl Default for VideoDecoderConfig {
    fn default() -> Self {
        Self {
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attn_resolutions: vec![],
            dropout: 0.0,
            resolution: 256,
            z_channels: 4,
            video_kernel_size: 3,
            alpha: 0.0,
            merge_strategy: MergeStrategy::Learned,
            time_mode: TimeMode::ConvOnly,
            num_frames: 25,
        }
    }
}

#[derive(Debug)]
pub struct VideoDecoder {
    conv_in: Conv2d,
    mid_block_1: VideoResBlock,
    mid_attn: AttnBlock,
    mid_block_2: VideoResBlock,
    up_blocks: Vec<VideoUpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: AE3DConv,
    config: VideoDecoderConfig,
}

impl VideoDecoder {
    pub fn new(vs: VarBuilder, config: VideoDecoderConfig) -> Result<Self> {
        let block_in = config.ch * config.ch_mult.last().copied().unwrap_or(1);
        let conv_in = conv2d(
            config.z_channels,
            block_in,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;
        let merge_policy = match config.merge_strategy {
            MergeStrategy::Fixed => MergePolicy::Fixed,
            MergeStrategy::Learned => MergePolicy::Learned,
            MergeStrategy::LearnedWithImages => MergePolicy::Learned,
        };
        let mid_block_1 = VideoResBlock::new(
            vs.pp("mid").pp("block_1"),
            block_in,
            config.video_kernel_size,
            merge_policy,
            config.dropout,
            config.alpha,
        )?;
        let mid_attn = AttnBlock::new(vs.pp("mid").pp("attn_1"), block_in)?;
        let mid_block_2 = VideoResBlock::new(
            vs.pp("mid").pp("block_2"),
            block_in,
            config.video_kernel_size,
            merge_policy,
            config.dropout,
            config.alpha,
        )?;
        let mut up_blocks = Vec::new();
        let mut curr_prev = block_in;
        let mut curr_res = config.resolution / (2usize.pow((config.ch_mult.len() - 1) as u32));
        let attn_set: HashSet<_> = config.attn_resolutions.iter().copied().collect();
        for i_level in (0..config.ch_mult.len()).rev() {
            let mult = config.ch_mult[i_level];
            let out_channels = config.ch * mult;
            let add_attn = attn_set.contains(&curr_res);
            let block_vs = vs.pp("up").pp(i_level.to_string());
            let block = VideoUpBlock::new(
                block_vs,
                VideoUpBlockConfig {
                    num_blocks: config.num_res_blocks + 1,
                    in_channels: curr_prev,
                    out_channels,
                    add_attn,
                    upsample: i_level != 0,
                    video_kernel: config.video_kernel_size,
                    merge_policy,
                    dropout: config.dropout,
                },
            )?;
            up_blocks.insert(0, block);
            curr_prev = out_channels;
            curr_res *= 2;
        }
        let conv_norm_out = group_norm(
            32,
            config.ch * config.ch_mult[0],
            DEFAULT_NORM_EPS,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out = AE3DConv::new(
            vs.pp("conv_out"),
            config.ch * config.ch_mult[0],
            config.video_kernel_size,
        )?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn,
            mid_block_2,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        })
    }

    pub fn config(&self) -> &VideoDecoderConfig {
        &self.config
    }

    pub fn forward(&self, zs: &Tensor) -> Result<Tensor> {
        let num_frames = self.config.num_frames;
        let z = self.conv_in.forward(zs)?;
        let mut h = self.mid_block_1.forward(&z, num_frames)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h, num_frames)?;
        for block in self.up_blocks.iter() {
            for (i, resnet) in block.blocks.iter().enumerate() {
                h = resnet.forward(&h, num_frames)?;
                if block.add_attn && i < block.attns.len() {
                    h = block.attns[i].forward(&h)?;
                }
            }
            if let Some(upsample) = &block.upsample {
                h = upsample.forward(&h)?;
            }
        }
        let h = self.conv_norm_out.forward(&h)?;
        let h = ops::silu(&h)?;
        self.conv_out.forward(&h, num_frames)
    }
}

#[derive(Debug, Clone)]
pub struct VideoAutoencoderConfig {
    pub num_frames: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub dropout: f64,
    pub resolution: usize,
    pub z_channels: usize,
    pub video_kernel_size: usize,
    pub alpha: f64,
    pub merge_strategy: MergeStrategy,
}

impl Default for VideoAutoencoderConfig {
    fn default() -> Self {
        Self {
            num_frames: 25,
            in_channels: 3,
            out_channels: 3,
            ch: 128,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attn_resolutions: vec![],
            dropout: 0.0,
            resolution: 256,
            z_channels: 4,
            video_kernel_size: 3,
            alpha: 0.0,
            merge_strategy: MergeStrategy::Learned,
        }
    }
}

pub struct VideoAutoencoder {
    encoder: AutoEncoderKL,
    decoder: VideoDecoder,
    config: VideoAutoencoderConfig,
}

impl VideoAutoencoder {
    pub fn new(vs: VarBuilder, config: VideoAutoencoderConfig) -> Result<Self> {
        let encoder = AutoEncoderKL::new(
            vs.pp("encoder"),
            config.in_channels,
            config.out_channels,
            AutoEncoderKLConfig {
                block_out_channels: config.ch_mult.iter().map(|mult| config.ch * mult).collect(),
                layers_per_block: config.num_res_blocks,
                latent_channels: config.z_channels,
                norm_num_groups: 32,
                use_quant_conv: true,
                use_post_quant_conv: true,
            },
        )?;
        let decoder_config = VideoDecoderConfig {
            ch: config.ch,
            out_ch: config.out_channels,
            ch_mult: config.ch_mult.clone(),
            num_res_blocks: config.num_res_blocks,
            attn_resolutions: config.attn_resolutions.clone(),
            dropout: config.dropout,
            resolution: config.resolution,
            z_channels: config.z_channels,
            video_kernel_size: config.video_kernel_size,
            alpha: config.alpha,
            merge_strategy: config.merge_strategy,
            time_mode: TimeMode::ConvOnly,
            num_frames: config.num_frames,
        };
        let decoder = VideoDecoder::new(vs.pp("decoder"), decoder_config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_time, _, _, _) = xs.dims4()?;
        if batch_time % self.config.num_frames != 0 {
            bail!("input batch must be divisible by num_frames");
        }
        let distribution = self.encoder.encode(xs)?;
        let latents = distribution.sample()?;
        let (latent_batch, channels, height, width) = latents.dims4()?;
        let batch = latent_batch / self.config.num_frames;
        let reshaped = latents.reshape((batch, self.config.num_frames, channels, height, width))?;
        Ok(reshaped)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, num_frames, channels, height, width) = latents.dims5()?;
        if num_frames != self.config.num_frames {
            bail!("unexpected num_frames");
        }
        let z = latents.reshape((batch * num_frames, channels, height, width))?;
        self.decoder.forward(&z)
    }

    pub fn config(&self) -> &VideoAutoencoderConfig {
        &self.config
    }
}
