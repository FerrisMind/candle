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

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn moe_gemm_gguf_vulkan() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    // Same tiny synthetic MoE as the CPU fallback test, but on the Vulkan
    // device: exercises the dequantize -> index_select -> matmul dequant
    // fallback path that previously accumulated per-layer buffers (vulkan
    // prefill OOM). Runs through the same heavy `moe_gemm_gguf_*` API used by
    // quantized_qwen3_moe so the allocator reuse pool is hit on repeated calls.
    let input_data = (1..=32).map(|v| v as f32 / 8.0).collect::<Vec<_>>();
    let input = Tensor::from_vec(input_data, (1, 32), &device)?;
    let weight_data = (0..(2 * 2 * 32))
        .map(|v| (v as f32 - 64.0) / 32.0)
        .collect::<Vec<_>>();
    let dense_weights = Tensor::from_vec(weight_data, (2, 2, 32), &device)?;
    let qweights = QTensor::quantize(&dense_weights, GgmlDType::Q8_0)?;

    let topk_ids = Tensor::new(&[[1u32, 0u32]], &device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    // Reference: dense matmul against the dequantized weights.
    let dense_ref = candle_nn::moe::moe_gemm(
        &input,
        &qweights.dequantize(&device)?,
        &None,
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
    )?;

    // Prefill path (is_prefill=true) is the one that OOM'd; exercise it a few
    // times across "layers" so the reuse pool must recycle the same buffers.
    // The vulkan path returns [batch, topk, n] and the caller reshapes+sums, so
    // compare flattened values against the dense reference (shape-independent).
    let dense_flat = dense_ref.flatten_all()?.to_vec1::<f32>()?;
    let mut prev: Option<Vec<f32>> = None;
    for _ in 0..4 {
        let gguf = candle_nn::moe::moe_gemm_gguf(
            &input,
            &qweights,
            &None,
            &sorted_token_ids,
            &expert_ids,
            2,
            true,
            DType::F32,
        )?;
        let vals = gguf.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals.len(), dense_flat.len(), "flattened length mismatch");
        for (i, (a, e)) in vals.iter().zip(dense_flat.iter()).enumerate() {
            // Fused quantized indexed-MoE accumulates in a different order than
            // the dense dequant+matmul reference, so use a relative tolerance.
            let tol = 1e-3 * e.abs().max(1.0) + 1e-3;
            assert!(
                (a - e).abs() <= tol,
                "mismatch at {i}: got {a}, expected {e}, tol {tol}"
            );
        }
        prev = Some(vals);
    }
    assert!(prev.is_some());
    Ok(())
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn moe_gemm_gguf_wgpu_deterministic() -> Result<()> {
    // Realistic dimensions: 8 experts, topk=4, k=1024, n=768, batch=4, Q4_K.
    // This exercises multiple k-tiles and workgroup tiling (in contrast to the
    // tiny 2-expert Q8_0 tests), which is where tile-reduction or atomic
    // accumulation order would surface as run-to-run divergence. Repeats the
    // fused op 5x and demands BIT-IDENTICAL output — a MoE inference kernel
    // must be deterministic at temperature 0 (greedy) or decode collapses.
    let device = Device::new_wgpu(0)?;
    let (num_exp, topk, k, n, batch) = (8usize, 4usize, 1024usize, 768usize, 4usize);
    let input_data = (0..(batch * k))
        .map(|v| (v as f32 - (batch * k) as f32 / 2.0) / (batch * k) as f32)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input_data, (batch, k), &device)?;
    let weight_data = (0..(num_exp * n * k))
        .map(|v| (v as f32 - (num_exp * n * k) as f32 / 2.0) / (num_exp * n * k) as f32)
        .collect::<Vec<_>>();
    let dense_weights = Tensor::from_vec(weight_data, (num_exp, n, k), &device)?;
    let qweights = QTensor::quantize(&dense_weights, GgmlDType::Q4K)?;

    // Deterministic ids: token i routes to experts (i, i+1, i+2, i+3) mod num_exp.
    let mut ids = vec![0u32; batch * topk];
    for t in 0..batch {
        for slot in 0..topk {
            ids[t * topk + slot] = ((t + slot) % num_exp) as u32;
        }
    }
    let topk_ids = Tensor::from_vec(ids, (batch, topk), &device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    // Also assert numeric correctness against the dense reference (2% rel tol).
    let dense_ref = candle_nn::moe::moe_gemm(
        &input,
        &qweights.dequantize(&device)?,
        &None,
        &sorted_token_ids,
        &expert_ids,
        topk,
        false,
    )?;
    let dense_flat = dense_ref.flatten_all()?.to_vec1::<f32>()?;

    let mut first: Option<Vec<f32>> = None;
    for iter in 0..5 {
        let gguf = candle_nn::moe::moe_gemm_gguf(
            &input,
            &qweights,
            &None,
            &sorted_token_ids,
            &expert_ids,
            topk,
            true,
            DType::F32,
        )?;
        let vals = gguf.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals.len(), dense_flat.len(), "flattened length mismatch");
        for (i, (a, e)) in vals.iter().zip(dense_flat.iter()).enumerate() {
            // After enabling FLOAT_ACC_SHMEM (f32 shmem + f32 accumulator) the
            // fused kernel should match the f32 dense reference to within the
            // standard Q4_K tolerance (2e-2). The old f16 accumulator was ~7%
            // off at k=1024, which is why it's asserted here.
            let tol = 2e-2 * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol,
                "iter {iter} mismatch at {i}: got {a}, expected {e}, tol {tol}"
            );
        }
        if let Some(prev) = &first {
            assert_eq!(
                &vals, prev,
                "iter {iter} output differs from iter 0: fused wgpu MoE is NONDETERMINISTIC"
            );
        } else {
            first = Some(vals);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn moe_gemm_gguf_wgpu() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    // Twin of `moe_gemm_gguf_vulkan`: routes the Wgpu device through the fused
    // selected-expert path (`moe_gemm_gguf_fused` -> `indexed_moe_forward` ->
    // wgpu `quantized_indexed_moe_f32`), keeping weights quantized on-device
    // instead of the dense 64-expert dequant that OOM'd wgpu prefill.
    let input_data = (1..=32).map(|v| v as f32 / 8.0).collect::<Vec<_>>();
    let input = Tensor::from_vec(input_data, (1, 32), &device)?;
    let weight_data = (0..(2 * 2 * 32))
        .map(|v| (v as f32 - 64.0) / 32.0)
        .collect::<Vec<_>>();
    let dense_weights = Tensor::from_vec(weight_data, (2, 2, 32), &device)?;
    let qweights = QTensor::quantize(&dense_weights, GgmlDType::Q8_0)?;

    let topk_ids = Tensor::new(&[[1u32, 0u32]], &device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    // Reference: dense matmul against the dequantized weights.
    let dense_ref = candle_nn::moe::moe_gemm(
        &input,
        &qweights.dequantize(&device)?,
        &None,
        &sorted_token_ids,
        &expert_ids,
        2,
        false,
    )?;

    // Exercise the prefill path (is_prefill=true) a few times across "layers".
    // The wgpu fused path returns [batch, topk, n], reshaped+summed by the
    // caller, so compare flattened values against the dense reference.
    let dense_flat = dense_ref.flatten_all()?.to_vec1::<f32>()?;
    let mut prev: Option<Vec<f32>> = None;
    for _ in 0..4 {
        let gguf = candle_nn::moe::moe_gemm_gguf(
            &input,
            &qweights,
            &None,
            &sorted_token_ids,
            &expert_ids,
            2,
            true,
            DType::F32,
        )?;
        let vals = gguf.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals.len(), dense_flat.len(), "flattened length mismatch");
        for (i, (a, e)) in vals.iter().zip(dense_flat.iter()).enumerate() {
            // Fused quantized indexed-MoE accumulates in a different order than
            // the dense dequant+matmul reference, so use a relative tolerance.
            let tol = 1e-3 * e.abs().max(1.0) + 1e-3;
            assert!(
                (a - e).abs() <= tol,
                "mismatch at {i}: got {a}, expected {e}, tol {tol}"
            );
        }
        prev = Some(vals);
    }
    assert!(prev.is_some());
    Ok(())
}
