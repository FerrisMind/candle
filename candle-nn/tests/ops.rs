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

fn rms_norm_large_magnitude(device: &Device) -> Result<()> {
    let (rows, hidden) = (4usize, 6912usize);
    let data: Vec<f32> = (0..rows * hidden)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((i as f32 * 0.17).sin().abs() * 7e9)
        })
        .collect();
    let tensor = Tensor::from_vec(data, (rows, hidden), device)?;
    let alpha = Tensor::ones(hidden, candle::DType::F32, device)?;

    let fused = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let slow = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;

    let fused_v = fused.flatten_all()?.to_vec1::<f32>()?;
    let slow_v = slow.flatten_all()?.to_vec1::<f32>()?;
    for &v in &fused_v {
        assert!(
            v.is_finite(),
            "rms_norm produced a non-finite value for large-magnitude input"
        );
    }
    for &v in &slow_v {
        assert!(
            v.is_finite(),
            "rms_norm_slow produced a non-finite value for large-magnitude input"
        );
    }
    let diff = fused_v
        .iter()
        .zip(slow_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 5e-3,
        "rms_norm and rms_norm_slow disagree: max |Δ| = {diff}"
    );
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
test_device!(
    rms_norm_large_magnitude,
    rms_norm_large_magnitude_cpu,
    rms_norm_large_magnitude_gpu,
    rms_norm_large_magnitude_metal
);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal);
test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal);
test_device!(sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal);

#[cfg(feature = "wgpu")]
fn wgpu_device_or_skip(test_name: &str) -> Result<Option<Device>> {
    match Device::new_wgpu(0) {
        Ok(device) => Ok(Some(device)),
        Err(err) if std::env::var_os("CANDLE_REQUIRE_WGPU_TEST_DEVICE").is_some() => Err(err),
        Err(err) => {
            eprintln!("skipping {test_name}: wgpu device unavailable: {err}");
            Ok(None)
        }
    }
}

#[cfg(feature = "vulkan")]
fn vulkan_device_or_skip(test_name: &str) -> Result<Option<Device>> {
    match Device::new_vulkan(0) {
        Ok(device) => Ok(Some(device)),
        Err(err) if std::env::var_os("CANDLE_REQUIRE_VULKAN_TEST_DEVICE").is_some() => Err(err),
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
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("softmax_wgpu")? else {
        return Ok(());
    };
    run_native_backend_case("softmax_wgpu", &device, softmax_last_dim_backend)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn rms_norm_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("rms_norm_wgpu")? else {
        return Ok(());
    };
    run_native_backend_case("rms_norm_wgpu", &device, rms_norm_backend)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn sigmoid_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("sigmoid_wgpu")? else {
        return Ok(());
    };
    sigmoid_backend(&device)
}

#[test]
#[cfg(feature = "vulkan")]
fn softmax_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("softmax_vulkan")? else {
        return Ok(());
    };
    run_native_backend_case("softmax_vulkan", &device, softmax_last_dim_backend)
}

#[test]
#[cfg(feature = "vulkan")]
fn rms_norm_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("rms_norm_vulkan")? else {
        return Ok(());
    };
    run_native_backend_case("rms_norm_vulkan", &device, rms_norm_backend)
}

#[test]
#[cfg(feature = "vulkan")]
fn sigmoid_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("sigmoid_vulkan")? else {
        return Ok(());
    };
    sigmoid_backend(&device)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn layer_norm_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("layer_norm_wgpu")? else {
        return Ok(());
    };
    run_native_backend_case("layer_norm_wgpu", &device, layer_norm_backend)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn layer_norm_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("layer_norm_vulkan")? else {
        return Ok(());
    };
    run_native_backend_case("layer_norm_vulkan", &device, layer_norm_backend)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn rope_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("rope_wgpu")? else {
        return Ok(());
    };
    run_native_backend_case("rope_wgpu", &device, rope_backend)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn rope_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("rope_vulkan")? else {
        return Ok(());
    };
    run_native_backend_case("rope_vulkan", &device, rope_backend)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn sdpa_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("sdpa_wgpu")? else {
        return Ok(());
    };
    run_native_backend_case("sdpa_wgpu", &device, sdpa_backend)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn sdpa_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("sdpa_vulkan")? else {
        return Ok(());
    };
    run_native_backend_case("sdpa_vulkan", &device, sdpa_backend)
}

#[test]
#[ignore = "heavy synthetic mini-graph; requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn mini_graph_wgpu() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = wgpu_device_or_skip("mini_graph_wgpu")? else {
        return Ok(());
    };
    mini_graph_backend(&device)
}

#[test]
#[ignore = "heavy synthetic mini-graph; requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn mini_graph_vulkan() -> Result<()> {
    let _guard = gpu_test_guard();
    let Some(device) = vulkan_device_or_skip("mini_graph_vulkan")? else {
        return Ok(());
    };
    mini_graph_backend(&device)
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

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn mini_graph_backend(device: &Device) -> Result<()> {
    run_graph_case(
        device,
        "mini_graph_attention_block",
        run_synthetic_attention_graph,
        5e-3,
        2e-3,
    )?;
    run_graph_case(
        device,
        "mini_graph_gated_mlp_block",
        run_synthetic_gated_mlp_graph,
        5e-3,
        2e-3,
    )?;
    run_graph_case(
        device,
        "mini_graph_mixer_block",
        run_synthetic_mixer_graph,
        5e-3,
        2e-3,
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_graph_case(
    device: &Device,
    label: &str,
    graph: fn(&Device) -> Result<Tensor>,
    atol: f32,
    rtol: f32,
) -> Result<()> {
    let cpu = Device::Cpu;
    reset_backend_fallback_count(device);
    let expected = graph(&cpu)?;
    let actual = graph(device)?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_close_vec(&actual, &expected, atol, rtol, label);

    let fallback_count = backend_fallback_count(device);
    assert_eq!(
        fallback_count, 0,
        "{label} triggered {fallback_count} CPU fallbacks"
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_native_backend_case(
    label: &str,
    device: &Device,
    body: fn(&Device) -> Result<()>,
) -> Result<()> {
    reset_backend_fallback_count(device);
    body(device)?;
    device.synchronize()?;
    let fallback_count = backend_fallback_count(device);
    assert_eq!(
        fallback_count, 0,
        "{label} triggered {fallback_count} CPU fallbacks"
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_synthetic_attention_graph(device: &Device) -> Result<Tensor> {
    let (batch, seq, hidden, heads, vocab) = (2usize, 4usize, 8usize, 2usize, 16usize);
    let head_dim = hidden / heads;

    let ids_vals = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
    let ids = Tensor::from_vec(ids_vals, (batch, seq), device)?;
    let emb = fake_tensor((vocab, hidden), vocab * hidden, 0, device)?;
    let alpha = positive_fake_tensor(hidden, 10_000, device)?;
    let qkv_w = fake_tensor((hidden, hidden * 3), hidden * hidden * 3, 20_000, device)?;
    let out_w = fake_tensor((hidden, hidden), hidden * hidden, 30_000, device)?;
    let mlp_in = fake_tensor((hidden, hidden * 2), hidden * hidden * 2, 40_000, device)?;
    let mlp_out = fake_tensor((hidden * 2, hidden), hidden * hidden * 2, 50_000, device)?;
    let logits_w = fake_tensor((hidden, vocab), hidden * vocab, 60_000, device)?;
    let cos = fake_tensor((seq, head_dim / 2), seq * (head_dim / 2), 70_000, device)?;
    let sin = fake_tensor((seq, head_dim / 2), seq * (head_dim / 2), 80_000, device)?;

    let xs = emb
        .embedding(&ids.flatten_all()?)?
        .reshape((batch, seq, hidden))?;
    let normed = candle_nn::ops::rms_norm(&xs, &alpha, 1e-5)?;
    let qkv = normed
        .reshape((batch * seq, hidden))?
        .matmul(&qkv_w)?
        .reshape((batch, seq, hidden * 3))?;

    let q = qkv
        .narrow(2, 0, hidden)?
        .reshape((batch, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = qkv
        .narrow(2, hidden, hidden)?
        .reshape((batch, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = qkv
        .narrow(2, hidden * 2, hidden)?
        .reshape((batch, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;

    let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
    let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
    let att = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let att = (att * (head_dim as f64).sqrt().recip())?;
    let probs = candle_nn::ops::softmax_last_dim(&att)?;
    let ctx = probs
        .matmul(&v)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * seq, hidden))?;

    let projected = ctx.matmul(&out_w)?.reshape((batch, seq, hidden))?;
    let residual = normed.add(&projected)?;
    let ff = residual
        .reshape((batch * seq, hidden))?
        .matmul(&mlp_in)?
        .silu()?
        .matmul(&mlp_out)?
        .reshape((batch, seq, hidden))?;
    let hidden_out = residual.add(&ff)?;
    hidden_out
        .reshape((batch * seq, hidden))?
        .matmul(&logits_w)?
        .reshape((batch, seq, vocab))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_synthetic_gated_mlp_graph(device: &Device) -> Result<Tensor> {
    let (batch, seq, hidden, ff, vocab) = (2usize, 5usize, 8usize, 16usize, 12usize);

    let xs = fake_tensor((batch, seq, hidden), batch * seq * hidden, 90_000, device)?;
    let alpha = positive_fake_tensor(hidden, 91_000, device)?;
    let beta = fake_tensor(hidden, hidden, 92_000, device)?;
    let up_w = fake_tensor((hidden, ff), hidden * ff, 93_000, device)?;
    let gate_w = fake_tensor((hidden, ff), hidden * ff, 94_000, device)?;
    let down_w = fake_tensor((ff, hidden), ff * hidden, 95_000, device)?;
    let logits_w = fake_tensor((hidden, vocab), hidden * vocab, 96_000, device)?;

    let normed = candle_nn::ops::layer_norm(&xs, &alpha, &beta, 1e-5)?;
    let flat = normed.reshape((batch * seq, hidden))?;
    let up = flat.matmul(&up_w)?.gelu()?;
    let gate = candle_nn::ops::sigmoid(&flat.matmul(&gate_w)?)?;
    let mixed = up.broadcast_mul(&gate)?;
    let ff_out = mixed.matmul(&down_w)?.reshape((batch, seq, hidden))?;
    let residual = xs.add(&ff_out)?;
    residual
        .reshape((batch * seq, hidden))?
        .matmul(&logits_w)?
        .reshape((batch, seq, vocab))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_synthetic_mixer_graph(device: &Device) -> Result<Tensor> {
    let (batch, seq, hidden, token_ff, channel_ff, vocab) =
        (2usize, 6usize, 8usize, 10usize, 16usize, 11usize);

    let xs = fake_tensor((batch, seq, hidden), batch * seq * hidden, 100_000, device)?;
    let alpha1 = positive_fake_tensor(hidden, 101_000, device)?;
    let beta1 = scaled_fake_tensor(hidden, hidden, 102_000, 0.1, device)?;
    let token_w1 = scaled_fake_tensor((seq, token_ff), seq * token_ff, 103_000, 0.15, device)?;
    let token_w2 = scaled_fake_tensor((token_ff, seq), token_ff * seq, 104_000, 0.15, device)?;
    let alpha2 = positive_fake_tensor(hidden, 105_000, device)?;
    let beta2 = scaled_fake_tensor(hidden, hidden, 106_000, 0.1, device)?;
    let channel_w1 = scaled_fake_tensor(
        (hidden, channel_ff),
        hidden * channel_ff,
        107_000,
        0.15,
        device,
    )?;
    let channel_w2 = scaled_fake_tensor(
        (channel_ff, hidden),
        channel_ff * hidden,
        108_000,
        0.15,
        device,
    )?;
    let logits_w = scaled_fake_tensor((hidden, vocab), hidden * vocab, 109_000, 0.15, device)?;

    let normed = candle_nn::ops::layer_norm(&xs, &alpha1, &beta1, 1e-5)?;
    let token_mixed = normed
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * hidden, seq))?
        .matmul(&token_w1)?
        .gelu()?
        .matmul(&token_w2)?
        .reshape((batch, hidden, seq))?
        .transpose(1, 2)?
        .contiguous()?;
    let residual = xs.add(&token_mixed)?;

    let channel_mixed = candle_nn::ops::layer_norm(&residual, &alpha2, &beta2, 1e-5)?
        .reshape((batch * seq, hidden))?
        .matmul(&channel_w1)?
        .gelu()?
        .matmul(&channel_w2)?
        .reshape((batch, seq, hidden))?;
    let hidden_out = residual.add(&channel_mixed)?;
    hidden_out
        .reshape((batch * seq, hidden))?
        .matmul(&logits_w)?
        .reshape((batch, seq, vocab))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// GPU backend tests share process-global fallback counters and create/destroy
/// real driver devices; serialize them so the parallel test runner cannot race
/// concurrent wgpu/Vulkan device lifecycles or cross-pollute counter resets.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn reset_backend_fallback_count(device: &Device) {
    if device.is_wgpu() {
        candle::reset_wgpu_cpu_fallback_count();
    } else if device.is_vulkan() {
        candle::reset_vulkan_cpu_fallback_count();
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn backend_fallback_count(device: &Device) -> usize {
    if device.is_wgpu() {
        candle::wgpu_cpu_fallback_count()
    } else if device.is_vulkan() {
        candle::vulkan_cpu_fallback_count()
    } else {
        0
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn fake_weight(idx: usize) -> f32 {
    ((idx * 37 % 101) as f32 - 50.0) / 50.0
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn fake_tensor<S: candle::shape::ShapeWithOneHole>(
    shape: S,
    len: usize,
    start: usize,
    device: &Device,
) -> Result<Tensor> {
    scaled_fake_tensor(shape, len, start, 1.0, device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn scaled_fake_tensor<S: candle::shape::ShapeWithOneHole>(
    shape: S,
    len: usize,
    start: usize,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let data = (0..len)
        .map(|idx| fake_weight(start + idx) * scale)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, shape, device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn positive_fake_tensor(len: usize, start: usize, device: &Device) -> Result<Tensor> {
    let data = (0..len)
        .map(|idx| fake_weight(start + idx).abs() + 0.5)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, len, device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_close_vec(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tol = atol + rtol * actual.abs().max(expected.abs());
        assert!(
            diff <= tol,
            "{label}: mismatch at idx {idx}: got {actual}, expected {expected}, diff {diff}, tol {tol}"
        );
    }
}
