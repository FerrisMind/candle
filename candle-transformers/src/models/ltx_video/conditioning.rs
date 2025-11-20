use super::vae::CausalVideoAutoencoder;
use candle::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};

/// Structure representing a conditioning item for video generation
#[derive(Clone, Debug)]
pub struct ConditioningItem {
    /// The latent representation of the conditioning frame(s)
    /// Shape: (B, C, F, H, W)
    pub latents: Tensor,
    /// The frame index (in pixel space) where this conditioning starts
    pub frame_idx: usize,
    /// The strength of the conditioning (0.0 to 1.0)
    pub scale: f64,
}

/// Load an image from path, resize, crop and normalize it
pub fn load_image(
    path: impl AsRef<std::path::Path>,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let img = image::open(path).map_err(|e| candle::Error::Msg(e.to_string()))?;
    preprocess_image(&img, height, width)
}

/// Preprocess an image: resize, center crop, and normalize to [-1, 1]
pub fn preprocess_image(img: &DynamicImage, height: usize, width: usize) -> Result<Tensor> {
    let (w, h) = img.dimensions();
    let aspect_ratio = w as f64 / h as f64;
    let target_aspect_ratio = width as f64 / height as f64;

    // Resize logic to cover the target dimensions
    let (nw, nh) = if aspect_ratio > target_aspect_ratio {
        // Image is wider than target
        let scale = height as f64 / h as f64;
        ((w as f64 * scale).round() as u32, height as u32)
    } else {
        // Image is taller than target
        let scale = width as f64 / w as f64;
        (width as u32, (h as f64 * scale).round() as u32)
    };

    let img_resized = img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3);

    // Center crop
    let crop_x = (nw - width as u32) / 2;
    let crop_y = (nh - height as u32) / 2;
    let img_cropped = img_resized.crop_imm(crop_x, crop_y, width as u32, height as u32);

    // Convert to tensor and normalize to [-1, 1]
    let img_rgb = img_cropped.to_rgb8();
    let raw = img_rgb.into_raw();
    let tensor = Tensor::from_vec(raw, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))? // (C, H, W)
        .to_dtype(candle::DType::F32)?;

    // Normalize from [0, 255] to [-1, 1]
    let tensor = ((tensor / 127.5)? - 1.0)?;

    Ok(tensor)
}

/// Apply Gaussian blur to a tensor (C, H, W)
/// Note: This is a simplified implementation. For production, use a proper kernel.
pub fn apply_gaussian_blur(tensor: &Tensor, _sigma: f64) -> Result<Tensor> {
    // TODO: Implement Gaussian blur using conv2d
    // For now, we return the tensor as is.
    Ok(tensor.clone())
}

/// Prepare conditioning latents from a list of images
pub fn prepare_conditioning_latents(
    vae: &CausalVideoAutoencoder,
    images: &[Tensor],
    indices: &[usize],
    device: &Device,
    dtype: candle::DType,
) -> Result<Vec<ConditioningItem>> {
    if images.len() != indices.len() {
        return Err(candle::Error::Msg(
            "Number of images must match number of indices".to_string(),
        ));
    }

    let mut items = Vec::new();

    for (img, &idx) in images.iter().zip(indices.iter()) {
        let img_dims = img.dims();
        let img_processed = if img_dims.len() == 3 {
            // (C, H, W) -> (1, C, 1, H, W)
            img.unsqueeze(0)?.unsqueeze(2)?
        } else if img_dims.len() == 4 {
            // (C, F, H, W) -> (1, C, F, H, W)
            img.unsqueeze(0)?
        } else {
            return Err(candle::Error::Msg(format!(
                "Invalid image tensor shape: {:?}. Expected (C, H, W) or (C, F, H, W)",
                img_dims
            )));
        };

        let img_processed = img_processed.to_device(device)?.to_dtype(dtype)?;

        // Encode to latents
        // VAE encode expects (B, C, F, H, W)
        // Returns (B, C_lat, F_lat, H_lat, W_lat)
        let latents = vae.encode(&img_processed)?;

        items.push(ConditioningItem {
            latents,
            frame_idx: idx,
            scale: 1.0, // Default scale
        });
    }

    Ok(items)
}

/// Blend conditioning latents into the generated latents
pub fn inject_conditioning(
    generated_latents: &Tensor,
    conditioning_items: &[ConditioningItem],
    temporal_compression: usize,
) -> Result<Tensor> {
    let mut output = generated_latents.clone();
    let (b, c, f_lat, h, w) = output.dims5()?;

    for item in conditioning_items {
        let start_latent_idx = item.frame_idx / temporal_compression;
        let (_, _, cond_f, _, _) = item.latents.dims5()?;

        if start_latent_idx + cond_f > f_lat {
            return Err(candle::Error::Msg(format!(
                "Conditioning frames out of bounds: start={}, len={}, total={}",
                start_latent_idx, cond_f, f_lat
            )));
        }

        // Slice assign
        // We need to construct ranges for all dimensions
        let ranges = [
            0..b,
            0..c,
            start_latent_idx..start_latent_idx + cond_f,
            0..h,
            0..w,
        ];

        // For blending, we first extract, blend, then assign.
        if (item.scale - 1.0).abs() < 1e-6 {
            // Hard replacement
            output = output.slice_assign(&ranges, &item.latents)?;
        } else {
            let current_slice = output.narrow(2, start_latent_idx, cond_f)?;
            let blended = ((&current_slice * (1.0 - item.scale))? + (&item.latents * item.scale)?)?;
            output = output.slice_assign(&ranges, &blended)?;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_inject_conditioning_hard() -> Result<()> {
        let device = Device::Cpu;
        // Create dummy generated latents: (1, 1, 4, 4, 4)
        let generated = Tensor::zeros((1, 1, 4, 4, 4), candle::DType::F32, &device)?;

        // Create conditioning latent: (1, 1, 1, 4, 4)
        let cond_latent = Tensor::ones((1, 1, 1, 4, 4), candle::DType::F32, &device)?;

        let item = ConditioningItem {
            latents: cond_latent.clone(),
            frame_idx: 0, // latent idx 0
            scale: 1.0,
        };

        let output = inject_conditioning(&generated, &[item], 1)?;

        // Check that the first frame is replaced with ones
        let first_frame = output.narrow(2, 0, 1)?;
        let diff = (first_frame - cond_latent)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5);

        // Check that other frames are still zeros
        let second_frame = output.narrow(2, 1, 1)?;
        let sum = second_frame.sum_all()?.to_scalar::<f32>()?;
        assert!(sum < 1e-5);

        Ok(())
    }

    #[test]
    fn test_inject_conditioning_blend() -> Result<()> {
        let device = Device::Cpu;
        let generated = Tensor::zeros((1, 1, 4, 4, 4), candle::DType::F32, &device)?;
        let cond_latent = Tensor::ones((1, 1, 1, 4, 4), candle::DType::F32, &device)?;

        let item = ConditioningItem {
            latents: cond_latent.clone(),
            frame_idx: 4, // latent idx 1 (if compression is 4)
            scale: 0.5,
        };

        // Use compression 4
        let output = inject_conditioning(&generated, &[item], 4)?;

        // Frame idx 4 -> latent idx 1
        let target_frame = output.narrow(2, 1, 1)?;

        // Should be 0.5 * 0 + 0.5 * 1 = 0.5
        let mean = target_frame.mean_all()?.to_scalar::<f32>()?;
        assert!((mean - 0.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_inject_conditioning_bounds() -> Result<()> {
        let device = Device::Cpu;
        let generated = Tensor::zeros((1, 1, 4, 4, 4), candle::DType::F32, &device)?;
        let cond_latent = Tensor::ones((1, 1, 1, 4, 4), candle::DType::F32, &device)?;

        let item = ConditioningItem {
            latents: cond_latent,
            frame_idx: 16, // latent idx 4 (out of bounds, size is 4, indices 0-3)
            scale: 1.0,
        };

        let result = inject_conditioning(&generated, &[item], 4);
        assert!(result.is_err());

        Ok(())
    }
}
