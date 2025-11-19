//! Tests for SVD Step 1.2: Temporal convolution blocks
#[cfg(test)]
mod tests {
    use candle::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::svd::video_unet::{MergeStrategy, TimeConv, VideoResBlock};

    #[test]
    fn test_timeconv_kernel_validation() {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle::DType::F32, &Device::Cpu);
        let result = TimeConv::new(vs.pp("test"), 256, 256, 4, 1);
        assert!(result.is_err(), "TimeConv should reject even kernel sizes");

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle::DType::F32, &Device::Cpu);
        let result = TimeConv::new(vs.pp("test"), 256, 256, 3, 1);
        assert!(result.is_ok(), "TimeConv should accept odd kernel sizes");
    }

    #[test]
    fn test_timeconv_forward_shape() -> candle::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle::DType::F32, &device);

        let time_conv = TimeConv::new(vs, 64, 64, 3, 1)?;
        let input = Tensor::randn(0f32, 1f32, (2, 64, 25), &device)?;
        let output = time_conv.forward(&input)?;

        assert_eq!(output.dims(), &[2, 64, 25]);
        Ok(())
    }

    #[test]
    fn test_videoresblock_forward() -> candle::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle::DType::F32, &device);

        let channels = 64;
        let video_res_block =
            VideoResBlock::new(vs, channels, 3, MergeStrategy::LearnedWithImages, 0.5)?;

        let x_spatial = Tensor::randn(0f32, 1f32, (8, channels, 32, 32), &device)?;
        let indicator =
            Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 4), &device)?;
        let output = video_res_block.forward(&x_spatial, 4, Some(&indicator))?;

        assert_eq!(output.dims(), &[8, channels, 32, 32]);
        Ok(())
    }
}
