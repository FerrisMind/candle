// Worker M3 regression guard: verifies the Vulkan fused selected-expert MoE path
// (`moe_gemm_gguf` -> `moe_gemm_gguf_vulkan` -> `indexed_moe_forward` ->
// `quantized_indexed_moe_f32`) uses the in-kernel dequant (mul_mat_vec_id) shader
// and stays both CORRECT (vs denso dequant reference) and DETERMINISTIC for Q4_K.
//
// This guards the M3 fix: the non-q8_1 branch used to request the non-existent
// `mul_mat_vec_id_<stem>_f32` shader name, which fell back to the dense
// `index_select_rows0_f32` path that materializes `[batch*topk, n, k]` f32
// (the +1.65 GB dedicated on Qwen3-16B-A3B). The shader is now named
// `mul_mat_vec_id_<stem>_f32_f32` to match the generated SPIR-V.
use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn moe_gemm_gguf_vulkan_q4k_deterministic() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    // Realistic dimensions mirroring the 16B-A3B MoE: 8 experts, topk=4,
    // k=1024, n=768, batch=4, Q4_K. Multiple k-tiles + workgroup tiling so
    // accumulation-order divergence would surface.
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

    // Deterministic ids: token t routes to experts (t, t+1, t+2, t+3) mod num_exp.
    let mut ids = vec![0u32; batch * topk];
    for t in 0..batch {
        for slot in 0..topk {
            ids[t * topk + slot] = ((t + slot) % num_exp) as u32;
        }
    }
    let topk_ids = Tensor::from_vec(ids, (batch, topk), &device)?;
    let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;

    // Numeric correctness against the dense dequant-matmul reference (2% rel tol,
    // the standard Q4_K in-kernel tolerance).
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
    for iter in 0..3 {
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
            let tol = 2e-2 * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol,
                "iter {iter} mismatch at {i}: got {a}, expected {e}, tol {tol}"
            );
        }
        if let Some(prev) = &first {
            assert_eq!(
                &vals, prev,
                "iter {iter} output differs from iter 0: vulkan Q4_K MoE is NONDETERMINISTIC"
            );
        } else {
            first = Some(vals);
        }
    }
    Ok(())
}
