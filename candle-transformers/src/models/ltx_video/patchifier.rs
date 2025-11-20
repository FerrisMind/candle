use candle::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct Patchifier {
    patch_size: (usize, usize, usize),
}

impl Patchifier {
    pub fn new(patch_size: (usize, usize, usize)) -> Self {
        Self { patch_size }
    }

    pub fn patchify(&self, video: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = video.dims5()?;
        let (pt, ph, pw) = self.patch_size;

        if f % pt != 0 || h % ph != 0 || w % pw != 0 {
            return Err(candle::Error::Msg(format!(
                "Video dimensions (f={}, h={}, w={}) must be divisible by patch size (pt={}, ph={}, pw={})",
                f, h, w, pt, ph, pw
            )));
        }

        let num_patches_f = f / pt;
        let num_patches_h = h / ph;
        let num_patches_w = w / pw;

        // (B, C, F, H, W) -> (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
        let reshaped = video.reshape(vec![
            b,
            c,
            num_patches_f,
            pt,
            num_patches_h,
            ph,
            num_patches_w,
            pw,
        ])?;

        // Permute to group patches together
        // (B, C, F/pt, pt, H/ph, ph, W/pw, pw) -> (B, F/pt, H/ph, W/pw, C, pt, ph, pw)
        let permuted = reshaped.permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?;

        // The above flatten_from(1) makes it (B, F/pt * H/ph * W/pw * C * pt * ph * pw) which is wrong.
        // We want (B, N, D).
        // permuted shape is (B, num_patches_f, num_patches_h, num_patches_w, C, pt, ph, pw)

        // Let's reshape explicitly to be safe.
        let n = num_patches_f * num_patches_h * num_patches_w;
        let d = c * pt * ph * pw;

        permuted.reshape((b, n, d))
    }

    pub fn unpatchify(
        &self,
        patches: &Tensor,
        shape: (usize, usize, usize, usize, usize), // (B, C, F, H, W)
    ) -> Result<Tensor> {
        let (b, c, f, h, w) = shape;
        let (pt, ph, pw) = self.patch_size;

        let (batch_size, num_patches, dim) = patches.dims3()?;

        if batch_size != b {
            return Err(candle::Error::Msg(format!(
                "Batch size mismatch: expected {}, got {}",
                b, batch_size
            )));
        }

        let expected_n = (f / pt) * (h / ph) * (w / pw);
        let expected_d = c * pt * ph * pw;

        if num_patches != expected_n {
            return Err(candle::Error::Msg(format!(
                "Number of patches mismatch: expected {}, got {}",
                expected_n, num_patches
            )));
        }

        if dim != expected_d {
            return Err(candle::Error::Msg(format!(
                "Patch dimension mismatch: expected {}, got {}",
                expected_d, dim
            )));
        }

        let num_patches_f = f / pt;
        let num_patches_h = h / ph;
        let num_patches_w = w / pw;

        // (B, N, D) -> (B, F/pt, H/ph, W/pw, C, pt, ph, pw)
        let reshaped = patches.reshape(vec![
            b,
            num_patches_f,
            num_patches_h,
            num_patches_w,
            c,
            pt,
            ph,
            pw,
        ])?;

        // Permute back to (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
        // Current: (0, 1, 2, 3, 4, 5, 6, 7) -> (B, nf, nh, nw, c, pt, ph, pw)
        // Target:  (B, c, nf, pt, nh, ph, nw, pw)
        // Indices: (0, 4, 1, 5, 2, 6, 3, 7)
        let permuted = reshaped.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?;

        // Reshape to (B, C, F, H, W)
        permuted.reshape((b, c, f, h, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_patchify_unpatchify() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let c = 128;
        let f = 8;
        let h = 32;
        let w = 32;

        let patch_size = (1, 2, 2);
        let patchifier = Patchifier::new(patch_size);

        // Create a random tensor
        let video = Tensor::randn(0f32, 1f32, (b, c, f, h, w), &device)?;

        // Patchify
        let patches = patchifier.patchify(&video)?;

        // Check dimensions
        // N = (F/pt) * (H/ph) * (W/pw) = 8/1 * 32/2 * 32/2 = 8 * 16 * 16 = 2048
        // D = C * pt * ph * pw = 128 * 1 * 2 * 2 = 512
        let (batch, n, d) = patches.dims3()?;
        assert_eq!(batch, b);
        assert_eq!(n, 8 * 16 * 16);
        assert_eq!(d, 128 * 1 * 2 * 2);

        // Unpatchify
        let reconstructed = patchifier.unpatchify(&patches, (b, c, f, h, w))?;

        // Check dimensions
        assert_eq!(reconstructed.dims(), &[b, c, f, h, w]);

        // Check values
        let diff = (video - reconstructed)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5);

        Ok(())
    }

    #[test]
    fn test_patchify_invalid_dimensions() -> Result<()> {
        let device = Device::Cpu;
        let patch_size = (1, 2, 2);
        let patchifier = Patchifier::new(patch_size);

        // Width 33 is not divisible by 2
        let video = Tensor::zeros((1, 128, 8, 32, 33), DType::F32, &device)?;

        let result = patchifier.patchify(&video);
        assert!(result.is_err());

        Ok(())
    }
}
