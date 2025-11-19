//! Video-aware UNet for Stable Video Diffusion.
//!
//! This implementation follows `stabilityai/stable-video-diffusion-img2vid-xt-1-1`
//! and mirrors the layout defined in
//! `tp/generative-models/sgm/modules/diffusionmodules/video_model.py`.
//! The architecture reuses the Candle `UNet2DConditionModel` backbone for the
//! spatial processing and adds temporal blending layers inspired by the Python
//! reference.

use candle::{bail, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, conv2d, layer_norm, linear, ops, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Init,
    LayerNorm, Linear, VarBuilder,
};

use std::collections::HashSet;

use crate::models::stable_diffusion::embeddings::LabelEmbedding;
use crate::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};

/// Strategy for blending spatial and temporal processing branches.
///
/// This enum determines how the `AlphaBlender` combines feature maps from the
/// spatial (2D) and temporal (3D) processing pathways.
///
/// **Reference**: `tp/generative-models/sgm/modules/diffusionmodules/util.py`, class `AlphaBlender`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Fixed blending weight, not trainable. Alpha remains constant.
    Fixed,
    /// Learnable blending weight. Alpha = sigmoid(mix_factor).
    Learned,
    /// Learnable with image-only indicator. When processing images, alpha = 1.0;
    /// when processing video frames, alpha = sigmoid(mix_factor).
    LearnedWithImages,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::LearnedWithImages
    }
}

/// Blends spatial and temporal feature branches with learnable or fixed weights.
///
/// Blends spatial and temporal feature branches with learnable or fixed weights.
///
/// AlphaBlender implements the strategy for combining outputs from spatial (2D convolution)
/// and temporal (3D convolution/attention) processing paths in a video UNet.
/// It supports three strategies: Fixed, Learned, and LearnedWithImages.
///
/// The blending formula is:
/// ```ignore
/// output = alpha * x_spatial + (1 - alpha) * x_temporal
/// ```
///
/// where alpha is computed based on the merge strategy and optionally the image_only_indicator.
/// Alpha is automatically reshaped via `rearrange_pattern` to enable correct broadcasting with
/// feature tensors of shape `(b, c, t, h, w)` or `(b*t, c, h, w)`.
///
/// **Reference**: `tp/generative-models/sgm/modules/diffusionmodules/util.py`, class `AlphaBlender` (lines 342-397)
#[derive(Debug)]
pub struct AlphaBlender {
    strategy: MergeStrategy,
    mix_factor: Tensor,
    /// Rearrange pattern to reshape alpha for proper broadcasting.
    /// For video: "b t -> b 1 t 1 1" reshapes (b, t) → (b, 1, t, 1, 1)
    /// For flat batch: "... -> 1" reshapes scalar/vector to broadcastable shape
    rearrange_pattern: String,
}

impl AlphaBlender {
    /// Creates a new AlphaBlender with the given strategy and initial mix factor.
    ///
    /// # Arguments
    /// * `vs` - VarBuilder for parameter management
    /// * `strategy` - The blending strategy (Fixed, Learned, or LearnedWithImages)
    /// * `mix_factor` - Initial blending factor (usually in range [0.0, 1.0])
    /// * `rearrange_pattern` - Pattern to reshape alpha for broadcasting.
    ///   Default: "b t -> b 1 t 1 1" for video tensors of shape (b, c, t, h, w)
    ///
    /// # Notes
    /// - For `Fixed` strategy, the mix_factor is stored as a constant tensor.
    /// - For `Learned` and `LearnedWithImages` strategies, mix_factor becomes a trainable parameter.
    /// - The rearrange pattern must accommodate both the batch/time dimensions and the output shape.
    pub fn new(
        vs: VarBuilder,
        strategy: MergeStrategy,
        mix_factor: f64,
        rearrange_pattern: String,
    ) -> Result<Self> {
        let mix_factor = match strategy {
            MergeStrategy::Fixed => Tensor::full(mix_factor, (1,), vs.device())?,
            MergeStrategy::Learned | MergeStrategy::LearnedWithImages => {
                vs.get_with_hints((1,), "mix_factor", Init::Const(mix_factor))?
            }
        };
        Ok(Self {
            strategy,
            mix_factor,
            rearrange_pattern,
        })
    }

    /// Computes the alpha blending weight based on the merge strategy.
    ///
    /// # Arguments
    /// * `image_only_indicator` - Optional tensor indicating which samples are image-only.
    ///   Format: `(batch, time)` where value > 0 means the sample is image-only.
    ///   Required for `LearnedWithImages` strategy.
    ///
    /// # Returns
    /// Alpha blending weights reshaped according to `rearrange_pattern`.
    /// Suitable for broadcasting with feature tensors.
    ///
    /// # Behavior by Strategy
    /// - **Fixed**: Returns the constant mix_factor
    /// - **Learned**: Returns sigmoid(mix_factor), a value in (0, 1)
    /// - **LearnedWithImages**: Returns 1.0 for image-only samples, sigmoid(mix_factor) for video frames,
    ///   then applies rearrange_pattern for proper broadcasting
    ///
    /// # Errors
    /// - If `LearnedWithImages` strategy is used without providing `image_only_indicator`
    /// - If `image_only_indicator` does not have exactly 2 dimensions (batch, time)
    fn compute_alpha(&self, image_only_indicator: Option<&Tensor>) -> Result<Tensor> {
        let alpha = match self.strategy {
            MergeStrategy::Fixed => self.mix_factor.clone(),
            MergeStrategy::Learned => ops::sigmoid(&self.mix_factor)?,
            MergeStrategy::LearnedWithImages => {
                let indicator = match image_only_indicator {
                    Some(t) => t,
                    None => {
                        bail!("LearnedWithImages merge strategy requires image_only_indicator")
                    }
                };

                // Validate indicator shape: must be (batch, time)
                if indicator.dims().len() != 2 {
                    bail!(
                        "image_only_indicator must have shape (batch, time), got {:?}",
                        indicator.dims()
                    );
                }

                let dtype = self.mix_factor.dtype();
                let device = indicator.device().clone();
                let (batch, time) = indicator.dims2()?;

                // Convert indicator to boolean mask: 1.0 where indicator > 0, 0.0 elsewhere
                let indicator_bool = indicator.to_dtype(dtype)?.gt(0.0)?.to_dtype(dtype)?;

                // Get sigmoid of learnable mix_factor
                let mix_alpha = ops::sigmoid(&self.mix_factor)?;

                // Create ones tensor with shape (batch, time)
                let ones = Tensor::ones((batch, time), dtype, &device)?;

                // alpha = where(image_only_indicator > 0, 1.0, sigmoid(mix_factor))
                let expanded_mix = mix_alpha.expand((batch, time))?;
                let one_minus_indicator = ones.broadcast_sub(&indicator_bool)?;
                indicator_bool.broadcast_add(&one_minus_indicator.broadcast_mul(&expanded_mix)?)?
            }
        };

        // Apply rearrange pattern to reshape alpha for proper broadcasting
        self.apply_rearrange_pattern(&alpha)
    }

    /// Applies rearrange pattern to reshape alpha tensor.
    ///
    /// Converts patterns like "b t -> b 1 t 1 1" by manually reshaping tensors.
    /// Currently supports video pattern: (batch, time) → (batch, 1, time, 1, 1)
    fn apply_rearrange_pattern(&self, alpha: &Tensor) -> Result<Tensor> {
        // Parse simple rearrange patterns
        // Pattern format: "source_dims -> target_shape"
        // Example: "b t -> b 1 t 1 1" means (batch, time) → (batch, 1, time, 1, 1)

        if self.rearrange_pattern.contains("b t -> b 1 t 1 1") {
            // Input: (batch, time), Output: (batch, 1, time, 1, 1)
            let (batch, time) = alpha.dims2()?;
            alpha
                .reshape((batch, 1, time, 1, 1))
                .map_err(|e| candle::Error::Msg(format!("Failed to reshape alpha: {}", e)))
        } else if self.rearrange_pattern.contains("b -> b 1 1 1 1") {
            // Input: scalar or (1,), Output: (b, 1, 1, 1, 1)
            let b = alpha.dims()[0];
            alpha
                .reshape((b, 1, 1, 1, 1))
                .map_err(|e| candle::Error::Msg(format!("Failed to reshape alpha: {}", e)))
        } else if self.rearrange_pattern.contains("-> 1 1 1") {
            // Scalar pattern: expand to (1, 1, 1)
            alpha
                .reshape((1, 1, 1))
                .map_err(|e| candle::Error::Msg(format!("Failed to reshape alpha: {}", e)))
        } else {
            // If pattern doesn't match known patterns, return alpha as-is with warning
            // This allows flexibility for custom patterns
            Ok(alpha.clone())
        }
    }

    /// Blends spatial and temporal feature tensors using the computed alpha weight.
    ///
    /// # Arguments
    /// * `x_spatial` - Spatial feature maps from 2D convolution branch. Shape: `(b, c, t, h, w)` or `(b*t, c, h, w)`
    /// * `x_temporal` - Temporal feature maps from 3D convolution/attention branch. Same shape as `x_spatial`
    /// * `image_only_indicator` - Optional indicator for image-only samples. Shape: `(batch, time)`
    ///
    /// # Returns
    /// Blended output: `alpha * x_spatial + (1 - alpha) * x_temporal`
    /// Same shape as input feature tensors
    ///
    /// # Broadcasting
    /// Alpha is automatically reshaped via `rearrange_pattern` to broadcast correctly with the feature tensors.
    pub fn blend(
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

/// Temporal 1D convolution with group-wise processing support.
///
/// Implements 1D convolution along the temporal dimension for video frames.
/// Supports learnable filtering with configurable kernel size and group convolution.
/// When applied to video data rearranged as `(b*h*w, c, t)`, processes temporal sequences
/// with adaptive padding to preserve temporal resolution.
///
/// # Equivalence to Conv3d
/// This is functionally equivalent to PyTorch's `conv_nd(dims=1, ...)` when used as part of
/// a VideoResBlock rearrangement pattern: (b*t, c, h, w) → (b*h*w, c, t) → TimeConv → ...
/// With kernel_size=[k], this replicates Conv3d with kernel_size=[k, 1, 1].
///
/// # Implementation Reference
/// - Standard 1D convolution (PyTorch `Conv1d`)
/// - Supports configurable number of groups for grouped convolution
/// - Padding calculated as `kernel_size // 2` to preserve temporal dimension
/// - Used in VideoResBlock and VideoAutoencoder for temporal coherence
///
/// **Reference**: `tp/generative-models/sgm/modules/diffusionmodules/openaimodel.py` (line ~150)
/// Mirrors PyTorch `conv_nd(dims=1, ...)` for temporal processing with optional grouping.
///
/// # Validation
/// - Kernel size must be odd and positive (for centered receptive field)
/// - In/out channels must be divisible by groups (standard grouped conv requirement)
#[derive(Debug)]
pub struct TimeConv {
    conv: Conv1d,
    #[allow(dead_code)]
    kernel_size: usize,
}

impl TimeConv {
    /// Creates a new TimeConv layer with configurable temporal convolution.
    ///
    /// # Arguments
    /// * `vs` - VarBuilder for parameter management
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the temporal kernel (typically odd: 3, 5, etc.)
    /// * `groups` - Number of groups for grouped convolution (default: 1)
    ///
    /// # Behavior
    /// - Padding is automatically set to `kernel_size / 2` to preserve temporal dimension
    /// - Supports both standard (`groups=1`) and grouped convolution
    /// - Bias is included by default
    ///
    /// # Validation Rules
    /// - `kernel_size` must be odd and positive (bail otherwise)
    /// - `in_channels` and `out_channels` must be divisible by `groups`
    ///
    /// # Rearrangement in Context
    /// Used as part of VideoResBlock with this pattern:
    /// 1. Input: `(b*t, c, h, w)`
    /// 2. Reshape: `(b, t, c, h, w)` → `(b, c, t, h, w)` → `(b*h*w, c, t)`
    /// 3. Apply TimeConv: `(b*h*w, c, t)` → `(b*h*w, c_out, t)`
    /// 4. Reshape back: `(b*t, c_out, h, w)`
    ///
    /// This effectively applies independent temporal convolution to each spatial position (h, w).
    ///
    /// # Examples
    /// ```ignore
    /// // Standard temporal convolution with kernel size 3
    /// let time_conv = TimeConv::new(vs, 256, 256, 3, 1)?;
    /// // Grouped convolution with 16 groups
    /// let grouped_time_conv = TimeConv::new(vs, 256, 256, 3, 16)?;
    /// ```
    pub fn new(
        vs: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        groups: usize,
    ) -> Result<Self> {
        if kernel_size == 0 || kernel_size.is_multiple_of(2) {
            bail!("kernel_size must be odd and positive");
        }
        if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
            bail!("in_channels and out_channels must be divisible by groups");
        }

        let padding = kernel_size / 2;
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel_size,
            Conv1dConfig {
                padding,
                groups,
                ..Default::default()
            },
            vs.pp("conv"),
        )?;
        Ok(Self { conv, kernel_size })
    }

    /// Applies temporal 1D convolution to rearranged video tensors.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `(batch, channels, time)` or `(b*h*w, channels, num_frames)`
    ///   where the last dimension represents the temporal axis
    ///
    /// # Returns
    /// Convolved tensor with shape `(batch, out_channels, time)` preserving temporal length
    ///
    /// # Errors
    /// Returns error if tensor dimensions are invalid for convolution
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Anisotropic 3D convolution for temporal processing in autoencoders.
///
/// Implements full anisotropic (non-uniform) 3D convolution with separate kernels
/// for temporal and spatial dimensions. Supports flexible receptive fields for video
/// frame encoding with learned temporal and spatial coherence.
///
/// # Architecture
///
/// The convolution is applied in sequence:
/// 1. **Spatial stage** (if h_kernel > 1 or w_kernel > 1): 2D convolution preserving temporal dim
///    - Input: `(b, c_in, t, h, w)` → Output: `(b, intermediate_channels, t, h', w')`
///    - Uses kernel `[h_kernel, w_kernel]` with appropriate padding
/// 2. **Temporal stage**: 1D convolution on each spatial position
///    - Rearrange: `(b, c, t, h, w)` → `(b*h*w, c, t)`
///    - Apply Conv1d: `(b*h*w, c, t)` → `(b*h*w, c_out, t)`
///    - Reshape back: `(b*h*w, c_out, t)` → `(b, c_out, t, h, w)`
///
/// # Behavior
///
/// With `video_kernel_size=[3, 1, 1]` from SVD XT 1.1:
/// - temporal kernel: 3, spatial kernels: 1 → only temporal convolution applied
/// - Functionally equivalent to `Conv3d(kernel_size=[3, 1, 1])` in PyTorch
///
/// With `video_kernel_size=[3, 3, 3]`:
/// - temporal kernel: 3, spatial kernels: 3 → Conv2d(3,3) then Conv1d(3) applied sequentially
/// - Full anisotropic 3D processing with spatial+temporal receptive fields
///
/// **Reference**: `tp/generative-models/sgm/modules/autoencoding/temporal_ae.py`, class `AE3DConv`
#[derive(Debug)]
pub struct AE3DConv {
    spatial_conv: Option<Conv2d>,
    temporal_conv: Conv1d,
    #[allow(dead_code)]
    kernel_sizes: Vec<usize>,
    has_spatial: bool,
}

impl AE3DConv {
    /// Creates an anisotropic 3D convolution layer with per-dimension kernel sizes.
    ///
    /// # Arguments
    /// * `vs` - VarBuilder for parameter management
    /// * `in_channels` - Number of input channels (first stage input)
    /// * `out_channels` - Number of output channels (final stage output)
    /// * `kernel_size` - Vector of kernel sizes: `[temporal, height, width]` or `[kernel_size]`
    ///
    /// # Behavior
    /// - Accepts `[k]` (uniform) or `[k_t, k_h, k_w]` (anisotropic)
    /// - If spatial kernels (h, w) are both 1: skip spatial stage, apply only temporal Conv1d
    /// - If spatial kernels > 1: apply Conv2d first, then temporal Conv1d
    /// - Padding is automatically computed as `(kernel_size - 1) / 2` for each dimension
    pub fn new(
        vs: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: Vec<usize>,
    ) -> Result<Self> {
        if kernel_size.is_empty() {
            bail!("kernel_size must not be empty");
        }

        let kernel_sizes = if kernel_size.len() == 1 {
            vec![kernel_size[0]; 3]
        } else if kernel_size.len() == 3 {
            kernel_size.clone()
        } else {
            bail!(
                "kernel_size must have length 1 or 3, got {}",
                kernel_size.len()
            );
        };

        let temporal_kernel = kernel_sizes[0];
        let spatial_h_kernel = kernel_sizes[1];
        let spatial_w_kernel = kernel_sizes[2];
        let has_spatial = spatial_h_kernel > 1 || spatial_w_kernel > 1;

        // Stage 1: Spatial convolution (optional, if h or w kernel > 1)
        let (spatial_conv, intermediate_channels) = if has_spatial {
            let spatial_padding = spatial_h_kernel / 2;

            let conv = conv2d(
                in_channels,
                in_channels,
                spatial_h_kernel,
                Conv2dConfig {
                    padding: spatial_padding,
                    ..Default::default()
                },
                vs.pp("spatial_conv"),
            )?;
            (Some(conv), in_channels)
        } else {
            (None, in_channels)
        };

        // Stage 2: Temporal convolution (always applied)
        let temporal_padding = temporal_kernel / 2;
        let temporal_conv = conv1d(
            intermediate_channels,
            out_channels,
            temporal_kernel,
            Conv1dConfig {
                padding: temporal_padding,
                groups: 1,
                ..Default::default()
            },
            vs.pp("temporal_conv"),
        )?;

        Ok(Self {
            spatial_conv,
            temporal_conv,
            kernel_sizes,
            has_spatial,
        })
    }

    /// Applies full anisotropic 3D convolution to video tensors.
    ///
    /// # Arguments
    /// * `x` - Input: `(batch*time, channels, height, width)`
    /// * `timesteps` - Number of video frames
    ///
    /// # Returns
    /// Output: `(batch*time, out_channels, height, width)` with spatial+temporal coherence
    ///
    /// # Implementation Notes
    ///
    /// 1. If spatial kernels are 1: direct temporal processing
    ///    - Rearrange: (b*t, c, h, w) → (b*h*w, c, t) → Conv1d → (b*h*w, c_out, t) → (b*t, c_out, h, w)
    ///
    /// 2. If spatial kernels > 1: sequential Conv2d then Conv1d
    ///    - Reshape: (b*t, c, h, w) → (b, t, c, h, w)
    ///    - Apply Conv2d to each frame: (b*t, c, h, w) → (b*t, c, h', w')
    ///    - Rearrange: (b*t, c, h', w') → (b*h'*w', c, t) → Conv1d → (b*h'*w', c_out, t)
    ///    - Reshape: (b*h'*w', c_out, t) → (b*t, c_out, h', w')
    pub fn forward(&self, x: &Tensor, timesteps: usize) -> Result<Tensor> {
        let (batch_time, _in_channels, height, width) = x.dims4()?;
        if timesteps == 0 {
            bail!("timesteps must be positive");
        }
        if batch_time % timesteps != 0 {
            bail!(
                "batch_time ({}) must be divisible by timesteps ({})",
                batch_time,
                timesteps
            );
        }

        let batch = batch_time / timesteps;
        let mut x = x.clone();
        let mut current_height = height;
        let mut current_width = width;

        // Stage 1: Apply spatial Conv2d if spatial kernels > 1
        if self.has_spatial {
            if let Some(spatial_conv) = &self.spatial_conv {
                x = spatial_conv.forward(&x)?;
                current_height = x.dim(2)?;
                current_width = x.dim(3)?;
            }
        }

        // Stage 2: Apply temporal Conv1d
        let in_channels_temp = x.dim(1)?;

        // Rearrange: (b*t, c, h, w) → (b, c, t, h, w) → (b*h*w, c, t)
        let x_rearranged = x
            .reshape((
                batch,
                timesteps,
                in_channels_temp,
                current_height,
                current_width,
            ))?
            .permute((0, 2, 1, 3, 4))?; // (b, c, t, h, w)

        let x_temporal_input = x_rearranged.permute((0, 3, 4, 1, 2))?.reshape((
            batch * current_height * current_width,
            in_channels_temp,
            timesteps,
        ))?;

        // Apply temporal 1D convolution
        let x_temporal = self.temporal_conv.forward(&x_temporal_input)?;
        let out_channels = x_temporal.dim(1)?;

        // Rearrange back: (b*h*w, c_out, t) → (b, h, w, c_out, t) → (b, c_out, t, h, w) → (b*t, c_out, h, w)
        let output = x_temporal
            .reshape((
                batch,
                current_height,
                current_width,
                out_channels,
                timesteps,
            ))?
            .permute((0, 4, 3, 1, 2))?
            .reshape((batch_time, out_channels, current_height, current_width))?;

        Ok(output)
    }
}

/// 3D Residual block for video processing.
///
/// Combines spatial 2D residual processing with learnable temporal convolutions
/// and AlphaBlender for mixing. This is the core building block for VideoUNet.
///
/// The forward pass:
/// 1. Apply spatial ResBlock (2D) to preserve spatial features
/// 2. Rearrange from `(b*t, c, h, w)` → `(b, c, t, h, w)` for temporal processing
/// 3. Apply temporal 3D ResBlock (via TimeConv implementation)
/// 4. Blend spatial and temporal outputs using AlphaBlender
/// 5. Rearrange back to `(b*t, c, h, w)` for downstream layers
///
/// **Reference**: `tp/generative-models/sgm/modules/diffusionmodules/video_model.py`, class `VideoResBlock`
#[derive(Debug)]
pub struct VideoResBlock {
    time_stack: TimeConv,
    time_mixer: AlphaBlender,
}

impl VideoResBlock {
    /// Creates a new VideoResBlock.
    ///
    /// # Arguments
    /// * `vs` - VarBuilder for parameter management
    /// * `channels` - Number of channels (in and out)
    /// * `kernel_size` - Temporal convolution kernel size
    /// * `merge_strategy` - Strategy for blending spatial and temporal branches
    /// * `merge_factor` - Initial blending factor
    ///
    /// # Notes
    /// - The spatial ResBlock is handled by the parent UNet's ResBlock
    /// - This structure handles temporal mixing and convolution
    /// - Uses standard convolution (groups=1) for temporal processing
    pub fn new(
        vs: VarBuilder,
        channels: usize,
        kernel_size: usize,
        merge_strategy: MergeStrategy,
        merge_factor: f64,
    ) -> Result<Self> {
        let time_stack = TimeConv::new(
            vs.pp("time_stack"),
            channels,
            channels,
            kernel_size,
            1, // groups=1 for standard convolution
        )?;
        let time_mixer = AlphaBlender::new(
            vs.pp("time_mixer"),
            merge_strategy,
            merge_factor,
            "b t -> b 1 t 1 1".to_string(),
        )?;
        Ok(Self {
            time_stack,
            time_mixer,
        })
    }

    /// Forward pass combining spatial and temporal processing.
    ///
    /// Integrates spatial ResBlock output with temporal processing.
    ///
    /// # Arguments
    /// * `x_spatial` - Output from spatial ResBlock, shape: `(b*t, c, h, w)`
    /// * `num_frames` - Number of frames in the batch
    /// * `image_only_indicator` - Optional indicator for image-only samples
    ///
    /// # Returns
    /// Blended output with same shape as input: `(b*t, c, h, w)`
    pub fn forward(
        &self,
        x_spatial: &Tensor,
        num_frames: usize,
        image_only_indicator: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_time, channels, height, width) = x_spatial.dims4()?;
        if num_frames == 0 {
            bail!("num_frames must be positive");
        }
        let batch = batch_time / num_frames;

        // Prepare spatial tensor for mixing: (b*t, c, h, w) -> (b, c, t, h, w)
        let x_spatial_mix = x_spatial
            .reshape((batch, num_frames, channels, height, width))?
            .permute((0, 2, 1, 3, 4))?; // (b, c, t, h, w)

        // Rearrange for temporal processing: (b*t, c, h, w) -> (b, c, t, h, w)
        let x_rearranged = x_spatial
            .reshape((batch, num_frames, channels, height, width))?
            .permute((0, 2, 1, 3, 4))?; // (b, c, t, h, w)

        // Reshape for temporal conv: (b*h*w, c, t)
        let x_conv_input = x_rearranged.permute((0, 3, 4, 1, 2))?.reshape((
            batch * height * width,
            channels,
            num_frames,
        ))?;

        // Apply temporal convolution
        let x_temporal = self.time_stack.forward(&x_conv_input)?;

        // Reshape back: (b*h*w, c, t) -> (b, h, w, c, t)
        let x_temporal = x_temporal
            .reshape((batch, height, width, channels, num_frames))?
            .permute((0, 3, 4, 1, 2))?; // (b, c, t, h, w)

        // Blend spatial and temporal using AlphaBlender
        let blended = self
            .time_mixer
            .blend(&x_spatial_mix, &x_temporal, image_only_indicator)?;

        // Reshape back to (b*t, c, h, w)
        let output = blended
            .permute((0, 2, 1, 3, 4))?
            .reshape((batch_time, channels, height, width))?;

        Ok(output)
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
        let depth = config
            .transformer_depths
            .get(index)
            .copied()
            .unwrap_or(config.transformer_layers_per_block);
        let use_cross_attn = if config.use_cross_attn && attention_set.contains(&ds) {
            Some(depth)
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
    pub transformer_depths: Vec<usize>,
    pub use_linear_projection: bool,
    pub cross_attention_dim: usize,
    pub context_dim: Option<usize>,
    pub norm_eps: f64,
    pub norm_num_groups: usize,
    pub use_flash_attn: bool,
    pub sliced_attention_size: Option<usize>,
    pub num_frames: usize,
    pub temporal_kernel_size: usize,
    pub temporal_norm_eps: f64,
    pub merge_factor: f64,
    pub merge_strategy: MergeStrategy,
    pub use_linear_in_transformer: bool,
    pub extra_ff_mix_layer: bool,
    pub use_spatial_context: bool,
    pub spatial_transformer_attn_type: String,
    pub disable_temporal_crossattention: bool,
    pub max_ddpm_temb_period: usize,
    /// Dimension of ADM (Adapter Diffusion Model) class conditioning input.
    /// If Some(dim), class embeddings are added to time embeddings.
    pub adm_in_channels: Option<usize>,
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
            transformer_depths: vec![1, 1, 1, 1],
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
            context_dim: Some(1024),
            use_linear_in_transformer: true,
            extra_ff_mix_layer: true,
            use_spatial_context: true,
            spatial_transformer_attn_type: "softmax-xformers".to_owned(),
            disable_temporal_crossattention: false,
            max_ddpm_temb_period: 10000,
            adm_in_channels: None,
        }
    }
}

pub struct VideoUnet {
    inner: UNet2DConditionModel,
    config: VideoUnetConfig,
    temporal_stack: TemporalStack,
    temporal_mixer: AlphaBlender,
    /// Optional class embedding for ADM-style conditioning
    class_embed: Option<LabelEmbedding>,
}

impl VideoUnet {
    pub fn new(vs: VarBuilder, config: VideoUnetConfig) -> Result<Self> {
        // Ensure the transformer depth vector spans every block.
        let transformer_depths = if config.transformer_depths.len() == config.channel_mult.len() {
            config.transformer_depths.clone()
        } else if config.transformer_depths.is_empty() {
            vec![config.transformer_layers_per_block; config.channel_mult.len()]
        } else {
            config
                .transformer_depths
                .iter()
                .cloned()
                .cycle()
                .take(config.channel_mult.len())
                .collect()
        };
        let config = VideoUnetConfig {
            transformer_depths,
            ..config
        };
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
            "b t -> b 1 t 1 1".to_string(),
        )?;

        // Initialize optional ADM class embedding
        let class_embed = if let Some(adm_in_channels) = config.adm_in_channels {
            let time_embed_dim = config.model_channels * 4;
            Some(LabelEmbedding::new(
                vs.pp("class_embed"),
                adm_in_channels,
                time_embed_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            inner,
            config,
            temporal_stack,
            temporal_mixer,
            class_embed,
        })
    }

    /// Forward pass of the VideoUNet.
    ///
    /// Processes video frames through spatial and temporal pathways, blending their outputs
    /// using learnable AlphaBlender weights.
    ///
    /// # Arguments
    /// * `input` - Flattened batch of video frames: `(batch*num_frames, in_channels, height, width)`
    /// * `timestep` - Diffusion timestep (scalar f64) used for time embeddings
    /// * `encoder_hidden_states` - Cross-attention context: `(batch*num_frames, seq_len, context_dim)`
    /// * `image_only_indicator` - Optional indicator for image-only samples
    ///
    /// # image_only_indicator Format
    /// Must have shape `(batch, num_frames)` where:
    /// - Value > 0 indicates the sample is image-only (single frame, no motion)
    /// - Value ≤ 0 indicates a video frame
    ///
    /// This is used by `LearnedWithImages` merge strategy to set alpha=1.0 for image-only samples
    /// (fully spatial) and alpha=sigmoid(mix_factor) for video frames (spatial+temporal mix).
    ///
    /// # class_embedding
    /// Optional class conditioning embedding. If `adm_in_channels` is configured,
    /// this embedding is added to the time embedding before passing to the UNet.
    /// Shape: `(batch, adm_in_channels)` or `(batch*num_frames, adm_in_channels)`
    ///
    /// # Example
    /// ```ignore
    /// // For batch_size=2, num_frames=4, all video:
    /// let image_only_indicator = Tensor::zeros((2, 4), dtype, device)?;
    /// // For batch_size=2, num_frames=4, first sample is image-only:
    /// let image_only_indicator = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 4), device)?;
    /// ```
    ///
    /// # Returns
    /// Denoised video latents: `(batch*num_frames, out_channels, height, width)`
    ///
    /// # Errors
    /// - If `input` batch size is not divisible by `num_frames`
    /// - If `image_only_indicator` shape is invalid when using `LearnedWithImages` strategy
    /// - If `adm_in_channels` is configured but `class_embedding` is not provided
    pub fn forward(
        &self,
        input: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        image_only_indicator: Option<&Tensor>,
        class_embedding: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_time, _, _, _) = input.dims4()?;
        let num_frames = self.config.num_frames;
        if num_frames == 0 {
            bail!("video UNet configured with zero frames");
        }
        if batch_time % num_frames != 0 {
            bail!("input batch ({batch_time}) is not divisible by num_frames ({num_frames})");
        }
        let mut temb = self.inner.compute_time_embedding(
            batch_time,
            timestep,
            input.dtype(),
            input.device().clone(),
        )?;

        // Add class embedding to time embedding if configured
        if let Some(class_embed) = &self.class_embed {
            if let Some(class_emb) = class_embedding {
                let class_emb_proj = class_embed.forward(class_emb)?;
                // Ensure shapes are compatible for broadcasting
                let class_emb_expanded = if class_emb_proj.dim(0)? == batch_time / num_frames {
                    // If class_emb is per-batch, expand to per-frame
                    class_emb_proj
                        .unsqueeze(1)?
                        .broadcast_as(temb.shape())?
                        .reshape(temb.shape().clone())?
                } else if class_emb_proj.dim(0)? == batch_time {
                    // Already per-frame
                    class_emb_proj
                } else {
                    bail!(
                        "class_embedding batch size ({}) doesn't match input batch ({}) or per-frame batch ({})",
                        class_emb_proj.dim(0)?,
                        batch_time,
                        batch_time / num_frames
                    );
                };
                temb = temb.broadcast_add(&class_emb_expanded)?;
            } else {
                bail!("class_embed is configured but class_embedding is not provided");
            }
        }

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
