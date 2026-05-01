use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};

fn assert_close_2d(actual: &[Vec<f32>], expected: &[Vec<f32>], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (ar, er)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(ar.len(), er.len());
        for (j, (a, e)) in ar.iter().zip(er.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "mismatch at ({i}, {j}): got {a}, expected {e}, tol {tol}"
            );
        }
    }
}

fn run_moe_gemm_backend(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], device)?;
    let weights = Tensor::new(
        &[
            [[1f32, 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            [[0f32, 0., 0., 1.], [1., 1., 0., 0.], [0., 1., 1., 0.]],
            [[1f32, 1., 1., 1.], [2., 0., 0., 0.], [0., 0., 0., 2.]],
        ],
        device,
    )?;
    let topk_ids = Tensor::new(&[[2u32, 0u32], [1u32, 2u32]], device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    let out = candle_nn::moe::moe_gemm(
        &input,
        &weights,
        &None,
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
    )?;
    assert_eq!(out.dims2()?, (4, 3));
    assert_eq!(
        out.to_vec2::<f32>()?,
        vec![
            vec![10.0, 2.0, 8.0],
            vec![1.0, 2.0, 3.0],
            vec![8.0, 11.0, 13.0],
            vec![26.0, 10.0, 16.0],
        ]
    );

    let topk_weights = Tensor::new(&[[0.1f32, 0.9f32], [0.3f32, 0.7f32]], device)?;
    let weights_2 = Tensor::new(
        &[
            [[1f32, 0., 0.], [0., 1., 0.]],
            [[0f32, 0., 1.], [1., 1., 1.]],
            [[1f32, 1., 1.], [2., 0., 1.]],
        ],
        device,
    )?;
    let out_2 = candle_nn::moe::moe_gemm(
        &out,
        &weights_2,
        &Some(topk_weights.clone()),
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
    )?;
    assert_eq!(out_2.dims2()?, (4, 2));

    let out_rows = out.to_vec2::<f32>()?;
    let ids = topk_ids.to_vec2::<u32>()?;
    let gates = topk_weights.to_vec2::<f32>()?;
    let w2 = weights_2.to_vec3::<f32>()?;
    let mut expected = vec![vec![0f32; 2]; 4];
    for tok in 0..2 {
        for slot in 0..2 {
            let row = tok * 2 + slot;
            let expert = ids[tok][slot] as usize;
            for n in 0..2 {
                let mut acc = 0f32;
                for k in 0..3 {
                    acc += out_rows[row][k] * w2[expert][n][k];
                }
                expected[row][n] = acc * gates[tok][slot];
            }
        }
    }
    assert_close_2d(&out_2.to_vec2::<f32>()?, &expected, 1e-5);
    Ok(())
}

#[test]
fn moe_gemm_cpu() -> Result<()> {
    run_moe_gemm_backend(&Device::Cpu)
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn moe_gemm_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    run_moe_gemm_backend(&device)
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn moe_gemm_vulkan() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    run_moe_gemm_backend(&device)
}

#[test]
fn moe_gemm_gguf_cpu_fallback() -> Result<()> {
    let device = Device::Cpu;
    let input_data = (1..=32).map(|v| v as f32 / 8.0).collect::<Vec<_>>();
    let input = Tensor::from_vec(input_data, (1, 32), &device)?;
    let weight_data = (0..(2 * 2 * 32))
        .map(|v| (v as f32 - 64.0) / 32.0)
        .collect::<Vec<_>>();
    let dense_weights = Tensor::from_vec(weight_data, (2, 2, 32), &device)?;
    let qweights = QTensor::quantize(&dense_weights, GgmlDType::Q8_0)?;

    let topk_ids = Tensor::new(&[[1u32, 0u32]], &device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    let dense_ref = candle_nn::moe::moe_gemm(
        &input,
        &qweights.dequantize(&device)?,
        &None,
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
    )?;
    let gguf = candle_nn::moe::moe_gemm_gguf(
        &input,
        &qweights,
        &None,
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
        DType::F32,
    )?;
    assert_close_2d(&gguf.to_vec2::<f32>()?, &dense_ref.to_vec2::<f32>()?, 1e-3);
    Ok(())
}
