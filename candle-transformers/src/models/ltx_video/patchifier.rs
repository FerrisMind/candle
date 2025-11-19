//! Patchifier for LTX-Video
//!
//! This module implements the patchify and unpatchify operations
//! for converting between video tensors and patch representations.

use candle::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// Patchifier for converting video to patches and vice versa
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patchifier {
    pub patch_size: (usize, usize, usize), // (t, h, w)
}

impl Patchifier {
    /// Create a new Patchifier
    pub fn new(patch_size: (usize, usize, usize)) -> Self {
        Self { patch_size }
    }

    /// Validate patch size
    pub fn validate(&self) -> Result<()> {
        if self.patch_size.0 == 0 || self.patch_size.1 == 0 || self.patch_size.2 == 0 {
            return Err(candle::Error::Msg(
                "Patch size dimensions must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert video tensor to patches
    /// Input: (B, C, F, H, W)
    /// Output: (B, num_patches, C*pt*ph*pw)
    pub fn patchify(&self, _video: &Tensor) -> Result<Tensor> {
        // Implementation will be added in task 8.1
        Err(candle::Error::Msg(
            "Patchifier patchify not yet implemented".to_string(),
        ))
    }

    /// Convert patches back to video tensor
    /// Input: (B, num_patches, C*pt*ph*pw)
    /// Output: (B, C, F, H, W)
    pub fn unpatchify(&self, _patches: &Tensor, _shape: (usize, usize, usize)) -> Result<Tensor> {
        // Implementation will be added in task 8.1
        Err(candle::Error::Msg(
            "Patchifier unpatchify not yet implemented".to_string(),
        ))
    }

    /// Calculate the number of patches for given dimensions
    pub fn num_patches(&self, frames: usize, height: usize, width: usize) -> Result<usize> {
        if !frames.is_multiple_of(self.patch_size.0) {
            return Err(candle::Error::Msg(format!(
                "Frames {} not divisible by patch size {}",
                frames, self.patch_size.0
            )));
        }
        if !height.is_multiple_of(self.patch_size.1) {
            return Err(candle::Error::Msg(format!(
                "Height {} not divisible by patch size {}",
                height, self.patch_size.1
            )));
        }
        if !width.is_multiple_of(self.patch_size.2) {
            return Err(candle::Error::Msg(format!(
                "Width {} not divisible by patch size {}",
                width, self.patch_size.2
            )));
        }
        Ok((frames / self.patch_size.0)
            * (height / self.patch_size.1)
            * (width / self.patch_size.2))
    }

    /// Calculate patch dimension given number of channels
    pub fn patch_dim(&self, channels: usize) -> usize {
        channels * self.patch_size.0 * self.patch_size.1 * self.patch_size.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchifier_creation() {
        let patchifier = Patchifier::new((1, 2, 2));
        assert_eq!(patchifier.patch_size, (1, 2, 2));
    }

    #[test]
    fn test_patchifier_validation() {
        let patchifier = Patchifier::new((1, 2, 2));
        assert!(patchifier.validate().is_ok());
    }

    #[test]
    fn test_patchifier_validation_invalid() {
        let patchifier = Patchifier::new((0, 2, 2));
        assert!(patchifier.validate().is_err());
    }

    #[test]
    fn test_num_patches() {
        let patchifier = Patchifier::new((1, 2, 2));
        let num = patchifier.num_patches(121, 64, 96).unwrap();
        assert_eq!(num, 121 * 32 * 48);
    }

    #[test]
    fn test_num_patches_invalid_frames() {
        let patchifier = Patchifier::new((2, 2, 2));
        assert!(patchifier.num_patches(121, 64, 96).is_err());
    }

    #[test]
    fn test_patch_dim() {
        let patchifier = Patchifier::new((1, 2, 2));
        let dim = patchifier.patch_dim(128);
        assert_eq!(dim, 128 * 1 * 2 * 2);
    }
}
