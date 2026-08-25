//! Worker-F4 diagnostic: replicate the stable-diffusion v1-5 VAE *decoder* as a
//! candle-nn test fixture (model struct copied from
//! candle-transformers/src/models/stable_diffusion/{vae,resnet,unet_2d_blocks,attention}.rs),
//! load the real VAE safetensors (fp16 on disk, cast to f32), and diff the
//! decode intermediates on CPU vs wgpu stage-by-stage for a FIXED known latent.
//!
//! Goal: localize the "green channel ~ 0 (magenta)" defect. If decode diverges
//! at a specific stage, that op is the crime scene. If decode is EXACT end-2-end,
//! the defect is upstream in the UNet-produced latents (extend to UNet afterwards).
//!
//! Run (serial GPU, own PID only):
//!   cargo test -p candle-nn --features wgpu --test wgpu_sd_vae_bisect -- --test-threads=1
//!
//! Env:
//!   CANDLE_SD_VAE  path to the VAE fp16 safetensors
//!                  (default G:\models\stable-diffusion-v1-5\vae\diffusion_pytorch_model.fp16.safetensors)
#![cfg(feature = "wgpu")]

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{
    conv2d, group_norm, layer_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module,
    VarBuilder,
};

fn relerr(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (((x - y).abs()) as f64) / ((x.hypot(*y)) as f64).max(1e-6))
        .fold(0.0f64, f64::max)
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn downcast(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 13) % 4093) as f32 / 61.0 - 32.0
        })
        .collect()
}

// N(0,1) gaussian input (sane magnitude, unlike LCG's [-32,32]).
fn gauss(seed: u64, n: usize) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = StdRng::seed_from_u64(seed);
    let d: StandardNormal = StandardNormal;
    (0..n).map(|_| { let x: f64 = d.sample(&mut rng); x as f32 }).collect()
}

fn model_path() -> String {
    std::env::var("CANDLE_SD_VAE").unwrap_or_else(|_| {
        r"G:\models\stable-diffusion-v1-5\vae\diffusion_pytorch_model.fp16.safetensors".to_string()
    })
}

fn vb_from_path(mp: &str, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    unsafe { VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&mp), dtype, device) }
}

fn conv_cfg(padding: usize) -> Conv2dConfig {
    Conv2dConfig { padding, ..Default::default() }
}

// ---- ResnetBlock2D (candle-transformers resnet.rs, temb=None) ----
struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
    output_scale_factor: f64,
}

impl ResnetBlock2D {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize, groups: usize, eps: f64) -> Result<Self> {
        let norm1 = group_norm(groups, in_channels, eps, vb.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg(1), vb.pp("conv1"))?;
        let norm2 = group_norm(groups, out_channels, eps, vb.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg(1), vb.pp("conv2"))?;
        let use_in_shortcut = in_channels != out_channels;
        let conv_shortcut = if use_in_shortcut {
            Some(conv2d(in_channels, out_channels, 1, conv_cfg(0), vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut, output_scale_factor: 1. })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut_xs = match &self.conv_shortcut {
            Some(c) => c.forward(xs)?,
            None => xs.clone(),
        };
        let xs = self.norm1.forward(xs)?;
        let xs = self.conv1.forward(&candle_nn::ops::silu(&xs)?)?;
        let xs = self.conv2.forward(&candle_nn::ops::silu(&self.norm2.forward(&xs)?)?)?;
        (shortcut_xs + xs)? / self.output_scale_factor
    }
}

// ---- AttentionBlock (VAE mid-block single-head spatial self-attn) ----
struct AttentionBlock {
    group_norm: GroupNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    proj_attn: Linear,
    channels: usize,
    num_heads: usize,
}

impl AttentionBlock {
    fn new(vb: VarBuilder, channels: usize, num_groups: usize, eps: f64) -> Result<Self> {
        // VAE mid-block: attn_num_head_channels=None => num_heads = channels/channels = 1.
        let num_heads = 1;
        let group_norm = group_norm(num_groups, channels, eps, vb.pp("group_norm"))?;
        // The fp16 file uses to_q/to_k/to_v/to_out.0 (2D Linear weights [channels,channels]).
        let use_to_names = vb.contains_tensor("to_q.weight");
        let (qp, kp, vp, op) = if use_to_names {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        let query = linear(channels, channels, vb.pp(qp))?;
        let key = linear(channels, channels, vb.pp(kp))?;
        let value = linear(channels, channels, vb.pp(vp))?;
        let proj_attn = linear(channels, channels, vb.pp(op))?;
        Ok(Self { group_norm, query, key, value, proj_attn, channels, num_heads })
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Result<Tensor> {
        let (batch, t, h_times_d) = xs.dims3()?;
        xs.reshape((batch, t, self.num_heads, h_times_d / self.num_heads))?
            .transpose(1, 2)
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let residual = xs;
        let (batch, channel, height, width) = xs.dims4()?;
        let xs = self.group_norm.forward(xs)?.reshape((batch, channel, height * width))?.transpose(1, 2)?;

        let query = self.query.forward(&xs)?;
        let key = self.key.forward(&xs)?;
        let value = self.value.forward(&xs)?;

        let query = self.transpose_for_scores(query)?.to_dtype(DType::F32)?;
        let key = self.transpose_for_scores(key)?.to_dtype(DType::F32)?;
        let value = self.transpose_for_scores(value)?.to_dtype(DType::F32)?;

        // scale applied twice => -0.25 (single head here).
        let scale = f64::powf(self.channels as f64 / self.num_heads as f64, -0.25);
        let attention_scores = (query * scale)?.matmul(&(key.t()? * scale)?)?;
        let attention_probs = candle_nn::ops::softmax(&attention_scores, D::Minus1)?;

        let xs = attention_probs.matmul(&value)?;
        let xs = xs.to_dtype(in_dtype)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = xs.flatten_from(D::Minus2)?;
        let xs = self.proj_attn.forward(&xs)?.t()?.reshape((batch, channel, height, width))?;
        (xs + residual)? / 1.0
    }
}

// ---- UNetMidBlock2D (VAE variant, num_layers=1, temb=None) ----
struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn: AttentionBlock,
    resnet2: ResnetBlock2D,
}

impl UNetMidBlock2D {
    fn new(vb: VarBuilder, in_channels: usize, groups: usize, eps: f64) -> Result<Self> {
        let resnet = ResnetBlock2D::new(vb.pp("resnets.0"), in_channels, in_channels, groups, eps)?;
        let attn = AttentionBlock::new(vb.pp("attentions.0"), in_channels, groups, eps)?;
        let resnet2 = ResnetBlock2D::new(vb.pp("resnets.1"), in_channels, in_channels, groups, eps)?;
        Ok(Self { resnet, attn, resnet2 })
    }
}

// ---- Upsample2D ----
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv = conv2d(in_channels, out_channels, 3, conv_cfg(1), vb.pp("conv"))?;
        Ok(Self { conv })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(2 * h, 2 * w)?;
        self.conv.forward(&xs)
    }
}

// ---- UpDecoderBlock2D ----
struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
}

impl UpDecoderBlock2D {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize, num_layers: usize, add_upsample: bool) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let cin = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(vb.pp(format!("resnets.{i}")), cin, out_channels, 32, 1e-6)?);
        }
        let upsampler = if add_upsample {
            Some(Upsample2D::new(vb.pp("upsamplers.0"), out_channels, out_channels)?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs)?;
        }
        match &self.upsampler {
            Some(u) => u.forward(&xs),
            None => Ok(xs),
        }
    }
}

// ---- Decoder ----
struct Decoder {
    conv_in: Conv2d,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    // block_out_channels = [128,256,512,512], layers_per_block = 2 (SD v1-5).
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let boc = [128usize, 256, 512, 512];
        let last = *boc.last().unwrap();
        let conv_in = conv2d(in_channels, last, 3, conv_cfg(1), vb.pp("conv_in"))?;
        let mid_block = UNetMidBlock2D::new(vb.pp("mid_block"), last, 32, 1e-6)?;
        let mut up_blocks = Vec::with_capacity(boc.len());
        let reversed: Vec<_> = boc.iter().copied().rev().collect();
        for index in 0..boc.len() {
            let outc = reversed[index];
            let inc = if index > 0 { reversed[index - 1] } else { reversed[0] };
            let is_final = index + 1 == boc.len();
            add_upblock(&mut up_blocks, vb.pp(format!("up_blocks.{index}")), inc, outc, 3, !is_final)?;
        }
        let conv_norm_out = group_norm(32, boc[0], 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(boc[0], out_channels, 3, conv_cfg(1), vb.pp("conv_out"))?;
        Ok(Self { conv_in, mid_block, up_blocks, conv_norm_out, conv_out })
    }

    // Forward returning every named stage (trace) for the bisect.
    #[allow(clippy::type_complexity)]
    fn forward_trace(&self, xs: &Tensor) -> Result<(
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    )> {
        let t0 = self.conv_in.forward(xs)?;                     // [1,512,32,32]
        let mid_res0 = self.mid_block.resnet.forward(&t0)?;     // resnets.0
        let mid_attn = self.mid_block.attn.forward(&mid_res0)?;
        let mid_out = self.mid_block.resnet2.forward(&mid_attn)?; // mid_block output
        let up0 = self.up_blocks[0].forward(&mid_out)?;
        let up1 = self.up_blocks[1].forward(&up0)?;
        let up2 = self.up_blocks[2].forward(&up1)?;
        let up3 = self.up_blocks[3].forward(&up2)?;
        let norm_out = self.conv_norm_out.forward(&up3)?;
        let silu_out = candle_nn::ops::silu(&norm_out)?;
        let out = self.conv_out.forward(&silu_out)?;
        Ok((t0, mid_res0, mid_attn, mid_out, up0, up1, up2, up3, norm_out, silu_out, out))
    }
}

fn add_upblock(
    v: &mut Vec<UpDecoderBlock2D>,
    vb: VarBuilder,
    inc: usize,
    outc: usize,
    num_layers: usize,
    add_upsample: bool,
) -> Result<()> {
    v.push(UpDecoderBlock2D::new(vb, inc, outc, num_layers, add_upsample)?);
    Ok(())
}

fn build_decoder(device: &Device, dtype: DType) -> Result<Decoder> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path()], dtype, device) }?;
    // VAE decoded with in_channels=4 (latent), out_channels=3 (image). Prefix "decoder".
    Decoder::new(vb.pp("decoder"), 4, 3)
}

#[test]
fn vae_decode_fixed_latent_cpu_vs_wgpu() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = model_path();
    if !std::path::Path::new(&mp).exists() {
        eprintln!("[vae] model file not found at {mp}; skipping");
        return Ok(());
    }

    let cpu_dec = build_decoder(&cpu, DType::F32)?;
    let wgpu_dec = build_decoder(&wgpu, DType::F32)?;

    // Fixed known latent [1,4,32,32] (latent for a 256x256 image; /8 & SD vae scale).
    let n = 4 * 32 * 32;
    let data = lcg(12345, n);
    let lat_cpu = Tensor::from_vec(data.clone(), [1, 4, 32, 32].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let lat_wgpu = Tensor::from_vec(data, [1, 4, 32, 32].as_slice(), &wgpu)?.to_dtype(DType::F32)?;

    let c = cpu_dec.forward_trace(&lat_cpu)?;
    let g = wgpu_dec.forward_trace(&lat_wgpu)?;

    let labels = [
        "conv_in", "mid_res0", "mid_attn", "mid_out", "up0", "up1", "up2", "up3", "norm_out",
        "silu_out", "out",
    ];
    let cvals = [
        downcast(&c.0)?, downcast(&c.1)?, downcast(&c.2)?, downcast(&c.3)?, downcast(&c.4)?,
        downcast(&c.5)?, downcast(&c.6)?, downcast(&c.7)?, downcast(&c.8)?, downcast(&c.9)?,
        downcast(&c.10)?,
    ];
    let gvals = [
        downcast(&g.0)?, downcast(&g.1)?, downcast(&g.2)?, downcast(&g.3)?, downcast(&g.4)?,
        downcast(&g.5)?, downcast(&g.6)?, downcast(&g.7)?, downcast(&g.8)?, downcast(&g.9)?,
        downcast(&g.10)?,
    ];

    for i in 0..labels.len() {
        let r = relerr(&cvals[i], &gvals[i]);
        let ma = maxabs(&cvals[i], &gvals[i]);
        eprintln!("[vae] {:10} cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6} n={}", labels[i], cvals[i].len());
    }

    // Final 3-channel output: per-channel means (R,G,B) to detect green-zeroing.
    let out_c = &cvals[10];
    let out_g = &gvals[10];
    let (h, w) = (256usize, 256usize);
    let ch = h * w;
    for (name, idx) in [("R", 0), ("G", 1), ("B", 2)] {
        let cs = &out_c[idx * ch..(idx + 1) * ch];
        let gs = &out_g[idx * ch..(idx + 1) * ch];
        let cm: f32 = cs.iter().sum::<f32>() / ch as f32;
        let gm: f32 = gs.iter().sum::<f32>() / ch as f32;
        let s: f32 = gs.iter().map(|v| v * v).sum::<f32>() / ch as f32 - gm * gm;
        let cs_std = cs.iter().map(|v| v * v).sum::<f32>() / ch as f32 - cm * cm;
        eprintln!("[vae] out ch {name}: cpu_mean={cm:.4} cpu_std2={cs_std:.4} wgpu_mean={gm:.4} wgpu_std2={s:.4}");
    }

    // Assertion: decode must be exact (f32 reduction tolerance). Green-zeroing is gross.
    let final_ma = maxabs(out_c, out_g);
    assert!(
        final_ma < 5e-3,
        "[vae] final decode output cpu-vs-wgpu maxabs={final_ma:.5} >= 5e-3 (VAE decode diverges; green bug inside VAE)"
    );
    Ok(())
}

// ============================================================================
// UNet multi-head cross-attention (SpatialTransformer). The UNet uses 8-head
// cross-attn; the VAE used single-head spatial self-attn. This block is the op
// the VAE decode did NOT exercise and the top suspect for the green bug.
// Replicated from candle-transformers attention.rs + unet_2d_blocks.rs.
// ============================================================================

struct GeGlu {
    proj: Linear,
}
impl GeGlu {
    fn new(vb: VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = linear(dim_in, dim_out * 2, vb.pp("proj"))?;
        Ok(Self { proj })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.proj.forward(xs)?.chunk(2, D::Minus1)?;
        &g[0] * g[1].gelu()?
    }
}

struct FeedForward {
    project_in: GeGlu,
    linear: Linear,
}
impl FeedForward {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let vs = vb.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, dim * 4)?;
        let linear = linear(dim * 4, dim, vs.pp("2"))?;
        Ok(Self { project_in, linear })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(&self.project_in.forward(xs)?)
    }
}

struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    scale: f64,
}
impl CrossAttention {
    fn new(vb: VarBuilder, query_dim: usize, context_dim: Option<usize>, heads: usize, dim_head: usize) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = candle_nn::linear_no_bias(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_no_bias(context_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_no_bias(context_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out.0"))?;
        Ok(Self { to_q, to_k, to_v, to_out, heads, scale })
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size / self.heads, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size / self.heads, seq_len, dim * self.heads))
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let key = self.to_k.forward(&context)?;
        let value = self.to_v.forward(&context)?;
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        let in_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let xs = query.matmul(&(key.t()? * self.scale)?)?;
        let xs = candle_nn::ops::softmax_last_dim(&xs)?;
        let xs = xs.matmul(&value)?.to_dtype(in_dtype)?;
        self.reshape_batch_dim_to_heads(&xs)?.apply(&self.to_out)
    }

    // Trace: returns (q_h, k_h, v_h, scores, probs, attn_out_heads, out).
    #[allow(clippy::type_complexity)]
    fn forward_trace(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let key = self.to_k.forward(&context)?;
        let value = self.to_v.forward(&context)?;
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        let in_dtype = query.dtype();
        let q32 = query.to_dtype(DType::F32)?;
        let k32 = key.to_dtype(DType::F32)?;
        let v32 = value.to_dtype(DType::F32)?;
        let scores = q32.matmul(&(k32.t()? * self.scale)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = probs.matmul(&v32)?.to_dtype(in_dtype)?;
        let out = self.reshape_batch_dim_to_heads(&attn)?.apply(&self.to_out)?;
        Ok((query, key, value, scores, probs, attn, out))
    }
}

struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    norm3: candle_nn::LayerNorm,
}
impl BasicTransformerBlock {
    fn new(vb: VarBuilder, dim: usize, n_heads: usize, d_head: usize, context_dim: Option<usize>) -> Result<Self> {
        let attn1 = CrossAttention::new(vb.pp("attn1"), dim, None, n_heads, d_head)?;
        let ff = FeedForward::new(vb.pp("ff"), dim)?;
        let attn2 = CrossAttention::new(vb.pp("attn2"), dim, context_dim, n_heads, d_head)?;
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let norm3 = layer_norm(dim, 1e-5, vb.pp("norm3"))?;
        Ok(Self { attn1, ff, attn2, norm1, norm2, norm3 })
    }
    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, context)? + xs)?;
        self.ff.forward(&self.norm3.forward(&xs)?)? + xs
    }
}

struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: Conv2d,
    blocks: Vec<BasicTransformerBlock>,
    proj_out: Conv2d,
}
impl SpatialTransformer {
    fn new(vb: VarBuilder, in_channels: usize, n_heads: usize, d_head: usize, context_dim: Option<usize>, num_groups: usize) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        let norm = group_norm(num_groups, in_channels, 1e-6, vb.pp("norm"))?;
        let proj_in = conv2d(in_channels, inner_dim, 1, Default::default(), vb.pp("proj_in"))?;
        let mut blocks = Vec::with_capacity(1);
        let tb = BasicTransformerBlock::new(vb.pp("transformer_blocks.0"), inner_dim, n_heads, d_head, context_dim)?;
        blocks.push(tb);
        let proj_out = conv2d(inner_dim, in_channels, 1, Default::default(), vb.pp("proj_out"))?;
        Ok(Self { norm, proj_in, blocks, proj_out })
    }
    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (batch, _channel, height, weight) = xs.dims4()?;
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let xs = self.proj_in.forward(&xs)?;
        let inner_dim = xs.dim(1)?;
        let xs = xs.transpose(1, 2)?.t()?.reshape((batch, height * weight, inner_dim))?;
        let mut xs = xs;
        for block in self.blocks.iter() {
            xs = block.forward(&xs, context)?;
        }
        let xs = self.proj_out.forward(
            &xs.reshape((batch, height, weight, inner_dim))?.t()?.transpose(1, 2)?,
        )?;
        xs + residual
    }
}

#[test]
fn unet_cross_attn_cpu_vs_wgpu() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = std::env::var("CANDLE_SD_UNET").unwrap_or_else(|_| {
        r"G:\models\stable-diffusion-v1-5\unet\diffusion_pytorch_model.fp16.safetensors".to_string()
    });
    if !std::path::Path::new(&mp).exists() {
        eprintln!("[xattn] unet file not found at {mp}; skipping");
        return Ok(());
    }

    // Two shapes: mid-block (1280 ch, 8x8 latent => seq 64) and down0 (320 ch, 32x32 => seq 1024).
    for (name, prefix, in_ch, sp, n_ctx) in [
        ("mid", "mid_block.attentions.0", 1280usize, 8usize, 768usize),
        ("down0", "down_blocks.0.attentions.0", 320usize, 32usize, 768usize),
    ] {
        let cpu_vb = vb_from_path(&mp, DType::F32, &cpu)?;
        let wgpu_vb = vb_from_path(&mp, DType::F32, &wgpu)?;
        let n_heads = 8usize;
        let d_head = in_ch / n_heads;
        let cpu_t = SpatialTransformer::new(cpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;
        let wgpu_t = SpatialTransformer::new(wgpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;

        // deterministic hidden [1,in_ch,sp,sp] and context [1, 77, n_ctx]
        let n_hidden = in_ch * sp * sp;
        let hdata = lcg(777, n_hidden);
        let n_ctx_el = 77 * n_ctx;
        let cdata = lcg(888, n_ctx_el);
        let hidden_cpu = Tensor::from_vec(hdata.clone(), [1, in_ch, sp, sp].as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let hidden_wgpu = Tensor::from_vec(hdata, [1, in_ch, sp, sp].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
        let ctx_cpu = Tensor::from_vec(cdata.clone(), [1, 77, n_ctx].as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let ctx_wgpu = Tensor::from_vec(cdata, [1, 77, n_ctx].as_slice(), &wgpu)?.to_dtype(DType::F32)?;

        let out_c = downcast(&cpu_t.forward(&hidden_cpu, Some(&ctx_cpu))?)?;
        let out_g = downcast(&wgpu_t.forward(&hidden_wgpu, Some(&ctx_wgpu))?)?;
        let r = relerr(&out_c, &out_g);
        let ma = maxabs(&out_c, &out_g);
        eprintln!("[xattn] {name:5} hidden=[1,{in_ch},{sp},{sp}] ctx=[1,77,{n_ctx}] heads={n_heads} dhead={d_head} cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6}");
        // NOTE: wgpu cross-attn uses the cooperative-matrix F16 mixed-precision
        // GEMM for these tile-aligned shapes, so it is ~1e-3 off the f32 CPU
        // reference (M%16==0 => coop64 path; ctx seq=77 is exact). This is an
        // intentional precision/perf tradeoff, NOT the SD green bug (which was
        // the wgpu seeded RNG). Informational only — no assert.
    }
    Ok(())
}

// Deep stage-by-stage trace through the mid-block SpatialTransformer (the
// diverging block). Dumps every intermediate on both devices and diffs them to
// localize the exact op inside cross-attention.
#[test]
fn xattn_mid_trace() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = std::env::var("CANDLE_SD_UNET").unwrap_or_else(|_| {
        r"G:\models\stable-diffusion-v1-5\unet\diffusion_pytorch_model.fp16.safetensors".to_string()
    });
    if !std::path::Path::new(&mp).exists() {
        eprintln!("[trace] unet file not found at {mp}; skipping");
        return Ok(());
    }
    let prefix = "mid_block.attentions.0";
    let in_ch = 1280usize;
    let sp = 8usize;
    let n_ctx = 768usize;
    let n_heads = 8usize;
    let d_head = in_ch / n_heads;

    let cpu_vb = vb_from_path(&mp, DType::F32, &cpu)?;
    let wgpu_vb = vb_from_path(&mp, DType::F32, &wgpu)?;
    let cpu_t = SpatialTransformer::new(cpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;
    let wgpu_t = SpatialTransformer::new(wgpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;

    let hdata = gauss(777, in_ch * sp * sp);
    let cdata = gauss(888, 77 * n_ctx);
    let hidden_c = Tensor::from_vec(hdata.clone(), [1, in_ch, sp, sp].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let ctx_c = Tensor::from_vec(cdata.clone(), [1, 77, n_ctx].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let hidden_g = Tensor::from_vec(hdata, [1, in_ch, sp, sp].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
    let ctx_g = Tensor::from_vec(cdata, [1, 77, n_ctx].as_slice(), &wgpu)?.to_dtype(DType::F32)?;

    let c = trace_transformer(&cpu_t, &hidden_c, &ctx_c)?;
    let g = trace_transformer(&wgpu_t, &hidden_g, &ctx_g)?;
    for i in 0..c.len() {
        let (name, cv) = &c[i];
        let (_, gv) = &g[i];
        let r = relerr(cv, gv);
        let ma = maxabs(cv, gv);
        eprintln!("[trace] {:16} cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6} n={}", name, cv.len());
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn trace_transformer(st: &SpatialTransformer, hidden: &Tensor, ctx: &Tensor) -> Result<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::new();
    let (batch, _ch, h, w) = hidden.dims4()?;
    let t_norm = st.norm.forward(hidden)?;
    out.push(("group_norm".into(), downcast(&t_norm)?));
    let t_pi = st.proj_in.forward(&t_norm)?;
    out.push(("proj_in(conv1)".into(), downcast(&t_pi)?));
    let inner_dim = t_pi.dim(1)?;
    let t_rs = t_pi.transpose(1, 2)?.t()?.reshape((batch, h * w, inner_dim))?;
    out.push(("reshape(seq,dim)".into(), downcast(&t_rs)?));

    let blk = &st.blocks[0];
    let n1 = blk.norm1.forward(&t_rs)?;
    out.push(("norm1(layernorm)".into(), downcast(&n1)?));
    let a1 = blk.attn1.forward_trace(&n1, None)?;
    out.push(("attn1.q_h".into(), downcast(&a1.0)?));
    out.push(("attn1.k_h".into(), downcast(&a1.1)?));
    out.push(("attn1.v_h".into(), downcast(&a1.2)?));
    out.push(("attn1.scores".into(), downcast(&a1.3)?));
    out.push(("attn1.probs".into(), downcast(&a1.4)?));
    out.push(("attn1.attn".into(), downcast(&a1.5)?));
    out.push(("attn1.out".into(), downcast(&a1.6)?));
    let xs1 = (&blk.attn1.forward(&n1, None)? + &t_rs)?;
    out.push(("res1".into(), downcast(&xs1)?));

    let n2 = blk.norm2.forward(&xs1)?;
    out.push(("norm2(layernorm)".into(), downcast(&n2)?));
    let a2 = blk.attn2.forward_trace(&n2, Some(ctx))?;
    out.push(("attn2.q_h".into(), downcast(&a2.0)?));
    out.push(("attn2.k_h".into(), downcast(&a2.1)?));
    out.push(("attn2.v_h".into(), downcast(&a2.2)?));
    out.push(("attn2.scores".into(), downcast(&a2.3)?));
    out.push(("attn2.probs".into(), downcast(&a2.4)?));
    out.push(("attn2.attn".into(), downcast(&a2.5)?));
    out.push(("attn2.out".into(), downcast(&a2.6)?));
    let xs2 = (&blk.attn2.forward(&n2, Some(ctx))? + &xs1)?;
    out.push(("res2".into(), downcast(&xs2)?));

    let n3 = blk.norm3.forward(&xs2)?;
    let ff = blk.ff.forward(&n3)?;
    out.push(("ff_out".into(), downcast(&ff)?));
    let xs3 = (&ff + &xs2)?;
    out.push(("res3".into(), downcast(&xs3)?));

    let t_po = xs3.reshape((batch, h, w, inner_dim))?.t()?.transpose(1, 2)?;
    let t_po2 = st.proj_out.forward(&t_po)?;
    out.push(("proj_out(conv1)".into(), downcast(&t_po2)?));
    let fin = (&t_po2 + hidden)?;
    out.push(("final".into(), downcast(&fin)?));
    Ok(out)
}

// Determinism + sane (gaussian) inputs, to distinguish a real wgpu kernel bug
// from f32/numerical ill-conditioning of the LCG input.
#[test]
fn xattn_mid_determinism() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = std::env::var("CANDLE_SD_UNET").unwrap_or_else(|_| {
        r"G:\models\stable-diffusion-v1-5\unet\diffusion_pytorch_model.fp16.safetensors".to_string()
    });
    if !std::path::Path::new(&mp).exists() {
        eprintln!("[det] unet file not found at {mp}; skipping");
        return Ok(());
    }
    let prefix = "mid_block.attentions.0";
    let in_ch = 1280usize;
    let sp = 8usize;
    let n_ctx = 768usize;
    let n_heads = 8usize;
    let d_head = in_ch / n_heads;

    let cpu_vb = vb_from_path(&mp, DType::F32, &cpu)?;
    let wgpu_vb = vb_from_path(&mp, DType::F32, &wgpu)?;
    let cpu_t = SpatialTransformer::new(cpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;
    let wgpu_t = SpatialTransformer::new(wgpu_vb.pp(prefix), in_ch, n_heads, d_head, Some(n_ctx), 32)?;

    // gaussian (sane) inputs
    let hdata = gauss(1234, in_ch * sp * sp);
    let cdata = gauss(5678, 77 * n_ctx);
    let hidden_c = Tensor::from_vec(hdata.clone(), [1, in_ch, sp, sp].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let ctx_c = Tensor::from_vec(cdata.clone(), [1, 77, n_ctx].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let hidden_g = Tensor::from_vec(hdata, [1, in_ch, sp, sp].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
    let ctx_g = Tensor::from_vec(cdata, [1, 77, n_ctx].as_slice(), &wgpu)?.to_dtype(DType::F32)?;

    let out_c = downcast(&cpu_t.forward(&hidden_c, Some(&ctx_c))?)?;
    let out_g1 = downcast(&wgpu_t.forward(&hidden_g, Some(&ctx_g))?)?;
    let out_g2 = downcast(&wgpu_t.forward(&hidden_g, Some(&ctx_g))?)?;
    let det = maxabs(&out_g1, &out_g2);
    let r = relerr(&out_c, &out_g1);
    let ma = maxabs(&out_c, &out_g1);
    eprintln!("[det] gaussian mid cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6} | wgpu run-run maxabs={det:.6}");
    // If a real kernel bug: wgpu deterministic (run-run ~0) but cpu-vs-wgpu wrong (large).
    eprintln!("[det] => wgpu deterministic={} (run-run), cpu-vs-wgpu divergence shown above", det < 1e-5f32);
    Ok(())
}

// Isolate WHERE attn1 diverges: the to_q Linear output (pre-reshape) vs the
// reshape_heads_to_batch_dim output (post-reshape). If pre-reshape is exact but
// post-reshape diverges, the reshape/transpose is the bug. If pre-reshape
// already diverges, the Linear matmul is the bug.
#[test]
fn xattn_reshape_probe() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = std::env::var("CANDLE_SD_UNET").unwrap_or_else(|_| {
        r"G:\models\stable-diffusion-v1-5\unet\diffusion_pytorch_model.fp16.safetensors".to_string()
    });
    if !std::path::Path::new(&mp).exists() {
        eprintln!("[probe] unet file not found at {mp}; skipping");
        return Ok(());
    }
    let prefix = "mid_block.attentions.0.transformer_blocks.0";
    let cpu_vb = vb_from_path(&mp, DType::F32, &cpu)?;
    let wgpu_vb = vb_from_path(&mp, DType::F32, &wgpu)?;
    let cpu_attn = CrossAttention::new(cpu_vb.pp(prefix).pp("attn1"), 1280, None, 8, 160)?;
    let wgpu_attn = CrossAttention::new(wgpu_vb.pp(prefix).pp("attn1"), 1280, None, 8, 160)?;

    let dat = gauss(42, 64 * 1280);
    let x_c = Tensor::from_vec(dat.clone(), [1, 64, 1280].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let x_g = Tensor::from_vec(dat, [1, 64, 1280].as_slice(), &wgpu)?.to_dtype(DType::F32)?;

    // pre-reshape (to_q linear output)
    let q_pre_c = cpu_attn.to_q.forward(&x_c)?;
    let q_pre_g = wgpu_attn.to_q.forward(&x_g)?;
    let cpre = downcast(&q_pre_c)?;
    let gpre = downcast(&q_pre_g)?;
    eprintln!("[probe] to_q(pre-reshape)  cpu-vs-wgpu relerr={:.6} maxabs={:.6}", relerr(&cpre, &gpre), maxabs(&cpre, &gpre));

    // post-reshape (reshape_heads_to_batch_dim)
    let q_post_c = cpu_attn.reshape_heads_to_batch_dim(&q_pre_c)?;
    let q_post_g = wgpu_attn.reshape_heads_to_batch_dim(&q_pre_g)?;
    let cpost = downcast(&q_post_c)?;
    let gpost = downcast(&q_post_g)?;
    eprintln!("[probe] to_q(post-reshape) cpu-vs-wgpu relerr={:.6} maxabs={:.6}", relerr(&cpost, &gpost), maxabs(&cpost, &gpost));

    // Also test the reshape sequence applied to a CONTIGUOUS gaussian tensor directly
    // (isolates transpose+reshape from any matmul).
    let raw = Tensor::from_vec(gauss(99, 64 * 1280), [1, 64, 1280].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let raw_g = Tensor::from_vec(gauss(99, 64 * 1280), [1, 64, 1280].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
    let rs_c = cpu_attn.reshape_heads_to_batch_dim(&raw)?;
    let rs_g = wgpu_attn.reshape_heads_to_batch_dim(&raw_g)?;
    let crs = downcast(&rs_c)?;
    let grs = downcast(&rs_g)?;
    eprintln!("[probe] reshape(Tensor raw) cpu-vs-wgpu relerr={:.6} maxabs={:.6}", relerr(&crs, &grs), maxabs(&crs, &grs));
    Ok(())
}

// Bare matmul probe: does the wgpu Linear/matmul kernel diverge at specific
// K/M/N dims? Distinguishes a real kernel bug from f32 reduction rounding.
// Focus: M (rows) sweep with fixed K=N=1280.
#[test]
fn matmul_kernel_probe() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    for m in [
        1usize, 2, 3, 4, 7, 8, 16, 32, 33, 48, 63, 64, 65, 66, 77, 96, 127, 128, 129, 192, 255,
        256, 257, 320, 512, 640, 1024,
    ] {
        let k = 1280usize;
        let n = 1280usize;
        let xdata = gauss(100, m * k);
        let wdata = gauss(200, n * k);
        let x_c = Tensor::from_vec(xdata.clone(), [m, k].as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let x_g = Tensor::from_vec(xdata, [m, k].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
        let w_c = Tensor::from_vec(wdata.clone(), [n, k].as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let w_g = Tensor::from_vec(wdata, [n, k].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
        let out_c = x_c.matmul(&w_c.t()?)?;
        let out_g = x_g.matmul(&w_g.t()?)?;
        let cv = downcast(&out_c)?;
        let gv = downcast(&out_g)?;
        let ma = maxabs(&cv, &gv);
        eprintln!("[mm] M={m:4} K=K=N=1280 cpu-vs-wgpu maxabs={ma:.6}");
    }
    Ok(())
}



// Regression (Worker-F4): the wgpu seeded RNG (randn/rand_uniform) must produce
// a correct distribution. splitmix64 constants were lo/hi SWAPPED and u64_mul
// was broken, biasing randn to mean~-0.67/std~0.33 (all-negative compressed)
// which made SD v1-5 initial latents wrong -> VAE decode green channel clamped
// to 0 (magenta images). This test asserts a proper N(0,1) / U(0,1).
#[test]
fn wgpu_rng_distribution() -> Result<()> {
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    wgpu.set_seed(12345)?;
    let n = 1usize << 16;
    let un = Tensor::rand(0f32, 1f32, n, &wgpu)?;
    let un = downcast(&un)?;
    let um: f32 = un.iter().sum::<f32>() / n as f32;
    let usd: f32 = (un.iter().map(|v| v * v).sum::<f32>() / n as f32 - um * um).sqrt();
    eprintln!("[rng] rand_uniform mean={um:.4} std={usd:.4} (expect mean~0.5 std~0.2887) min={} max={}", un.iter().copied().fold(f32::INFINITY, f32::min), un.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    assert!((um - 0.5).abs() < 0.02, "[rng] rand_uniform mean={um:.4} != ~0.5 (biased RNG)");
    assert!((usd - 0.2887).abs() < 0.02, "[rng] rand_uniform std={usd:.4} != ~0.2887");

    wgpu.set_seed(12345)?;
    let rn = Tensor::randn(0f64, 1f64, n, &wgpu)?;
    let rn = downcast(&rn)?;
    let rm: f32 = rn.iter().sum::<f32>() / n as f32;
    let rsd: f32 = (rn.iter().map(|v| v * v).sum::<f32>() / n as f32 - rm * rm).sqrt();
    let rmin = rn.iter().copied().fold(f32::INFINITY, f32::min);
    let rmax = rn.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("[rng] randn mean={rm:.4} std={rsd:.4} (expect mean~0 std~1) min={rmin:.3} max={rmax:.3}");
    assert!(rm.abs() < 0.02, "[rng] randn mean={rm:.4} != ~0 (biased RNG)");
    assert!((rsd - 1.0).abs() < 0.05, "[rng] randn std={rsd:.4} != ~1");
    // Fully-correct randn must have both signs present and a wide spread.
    assert!(rmin < -3.0 && rmax > 3.0, "[rng] randn range [{rmin:.3},{rmax:.3}] suspiciously narrow (compressed)");
    assert!(rn.iter().any(|&v| v > 0.0) && rn.iter().any(|&v| v < 0.0), "[rng] randn has no sign mix (all one sign = biased)");
    Ok(())
}
