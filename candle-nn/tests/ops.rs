#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, test_utils::to_vec3_round, Device, IndexOp, Result, Tensor};

fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round(&t0, 4)?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.3636], [0.1111, 0.7143, 0.5294]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t1, 4)?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.1667, 0.3077], [0.25, 0.8333, 0.6923]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    Ok(())
}

fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(to_vec3_round(&t, 2)?, to_vec3_round(&t2, 2)?);
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

#[test]
fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1::<f32>()?, &[1f32, 0.]);
    Ok(())
}

fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope_i(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(0..1)?, &cos, &sin)?
    };
    let rope2 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(1..2)?, &cos2, &sin2)?
    };

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &both_cos, &both_sin)?
    };
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(ropei, ropei_cpu, ropei_gpu, ropei_metal);
test_device!(rope, rope_cpu, rope_gpu, rope_metal);
test_device!(rope_thd, rope_thd_cpu, rope_thd_gpu, rope_thd_metal);
test_device!(softmax, softmax_cpu, softmax_gpu, softmax_metal);
test_device!(rms_norm, rms_norm_cpu, rms_norm_gpu, rms_norm_metal);
test_device!(rms_norml, rms_norml_cpu, rms_norml_gpu, rms_norml_metal);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal);
test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal);
test_device!(sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal);

#[cfg(feature = "vulkan")]
fn vulkan_device_or_skip(test_name: &str) -> Result<Option<Device>> {
    match Device::new_vulkan(0) {
        Ok(device) => Ok(Some(device)),
        Err(err) => {
            eprintln!("skipping {test_name}: Vulkan device unavailable: {err}");
            Ok(None)
        }
    }
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn softmax_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    softmax_last_dim_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn rms_norm_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    rms_norm_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn sigmoid_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    sigmoid_backend(&device)
}

#[test]
#[cfg(feature = "vulkan")]
fn softmax_vulkan() -> Result<()> {
    let Some(device) = vulkan_device_or_skip("softmax_vulkan")? else {
        return Ok(());
    };
    softmax_last_dim_backend(&device)
}

#[test]
#[cfg(feature = "vulkan")]
fn rms_norm_vulkan() -> Result<()> {
    let Some(device) = vulkan_device_or_skip("rms_norm_vulkan")? else {
        return Ok(());
    };
    rms_norm_backend(&device)
}

#[test]
#[cfg(feature = "vulkan")]
fn sigmoid_vulkan() -> Result<()> {
    let Some(device) = vulkan_device_or_skip("sigmoid_vulkan")? else {
        return Ok(());
    };
    sigmoid_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn layer_norm_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    layer_norm_backend(&device)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn layer_norm_vulkan() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    layer_norm_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn rope_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    rope_backend(&device)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn rope_vulkan() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    rope_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn sdpa_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    sdpa_backend(&device)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn sdpa_vulkan() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    sdpa_backend(&device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn softmax_last_dim_backend(device: &Device) -> Result<()> {
    let tensor = Tensor::new(
        &[
            [3f32.ln(), 1f32.ln(), 4f32.ln()],
            [1f32.ln(), 5f32.ln(), 9f32.ln()],
        ],
        device,
    )?;
    let t = candle_nn::ops::softmax_last_dim(&tensor)?;
    assert_eq!(
        candle::test_utils::to_vec2_round(&t, 4)?,
        &[[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn sigmoid_backend(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t = candle_nn::ops::sigmoid(&tensor)?;
    assert_eq!(
        candle::test_utils::to_vec3_round(&t, 4)?,
        &[
            [[0.9526, 0.7311, 0.982], [0.7311, 0.9933, 0.9999]],
            [[0.8808, 0.7311, 0.9991], [0.9997, 0.8808, 0.9997]]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn rms_norm_backend(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        candle::test_utils::to_vec3_round(&t, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn layer_norm_backend(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let fast = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let slow = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (fast - slow)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-4, "layer_norm backend diff too large: {diff}");
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn rope_backend(device: &Device) -> Result<()> {
    let src_values = [&[
        1.0f32, 0.5, -1.0, 2.0, 3.0, -0.5, 0.25, 1.5, //
        -2.0, 4.0, 1.25, -0.75, 0.0, 2.5, -1.5, 3.5, //
    ]];
    let cos_values = [&[
        1.0f32, 0.5, 0.25, 0.75, //
        0.9, 0.4, 0.3, 0.8, //
    ]];
    let sin_values = [&[
        0.0f32, 0.2, 0.3, 0.4, //
        0.1, 0.25, 0.35, 0.45, //
    ]];

    let src = Tensor::from_slice(src_values[0], (1, 1, 2, 8), device)?;
    let cos = Tensor::from_slice(cos_values[0], (2, 4), device)?;
    let sin = Tensor::from_slice(sin_values[0], (2, 4), device)?;

    let cpu = Device::Cpu;
    let src_cpu = Tensor::from_slice(src_values[0], (1, 1, 2, 8), &cpu)?;
    let cos_cpu = Tensor::from_slice(cos_values[0], (2, 4), &cpu)?;
    let sin_cpu = Tensor::from_slice(sin_values[0], (2, 4), &cpu)?;

    let rope_fast = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope_slow = candle_nn::rotary_emb::rope_slow(&src_cpu, &cos_cpu, &sin_cpu)?;
    let diff = (rope_fast
        .to_dtype(candle::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(rope_slow.flatten_all()?.to_vec1::<f32>()?.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>())
        / rope_fast.elem_count() as f32;
    assert!(diff < 1e-4, "rope backend diff too large: {diff}");

    let rope_i_fast = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope_i_slow = candle_nn::rotary_emb::rope_i_slow(&src_cpu, &cos_cpu, &sin_cpu)?;
    let diff = (rope_i_fast
        .to_dtype(candle::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(rope_i_slow.flatten_all()?.to_vec1::<f32>()?.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>())
        / rope_i_fast.elem_count() as f32;
    assert!(diff < 1e-4, "rope_i backend diff too large: {diff}");
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn sdpa_backend(device: &Device) -> Result<()> {
    let q = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
            0.7, 0.1, 0.2, //
            0.3, 0.4, 0.5, //
        ],
        (1, 2, 2, 3),
        device,
    )?;
    let k = Tensor::from_slice(
        &[
            0.2f32, 0.1, 0.3, //
            0.6, 0.4, 0.2, //
            0.5, 0.7, 0.8, //
        ],
        (1, 1, 3, 3),
        device,
    )?;
    let v = Tensor::from_slice(
        &[
            1.0f32, 0.0, //
            0.5, 2.0, //
            1.5, 1.0, //
        ],
        (1, 1, 3, 2),
        device,
    )?;
    let scale = (3f32).sqrt().recip();
    let fast = candle_nn::ops::sdpa(&q, &k, &v, None, true, scale, 1.0)?;

    let ids = Tensor::from_slice(&[0u32, 0], (2,), device)?;
    let k_exp = k.index_select(&ids, 1)?;
    let v_exp = v.index_select(&ids, 1)?;
    let mut att = q.to_dtype(candle::DType::F32)?.matmul(
        &k_exp
            .to_dtype(candle::DType::F32)?
            .transpose(2, 3)?
            .contiguous()?,
    )?;
    att = (att * scale as f64)?;
    let mask = Tensor::from_slice(
        &[0.0f32, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0],
        (1, 1, 2, 3),
        device,
    )?;
    let probs = candle_nn::ops::softmax_last_dim(&att.broadcast_add(&mask)?)?;
    let reference = probs.matmul(&v_exp.to_dtype(candle::DType::F32)?)?;

    let diff = (fast.to_dtype(candle::DType::F32)? - reference)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert!(diff < 2e-4, "sdpa backend diff too large: {diff}");
    Ok(())
}
