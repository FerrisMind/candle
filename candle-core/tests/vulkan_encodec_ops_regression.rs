//! Vulkan op regressions exposed by the encodec encoder (P0) and RWKV bf16
//! inference (P1). Both are fixed in candle-core/src/vulkan_backend.rs.
//!
//! 1. `vulkan_conv1d_long_l_matches_cpu`: conv1d im2col dispatch overflows
//!    Vulkan `maxComputeWorkGroupCount` when the output length (l_out) exceeds
//!    65535, so the shader grid-stride only covers a prefix and the rest of the
//!    output is garbage. The fix clamps the dispatch to the device limit; the
//!    im2col kernel grid-strides over the remainder.
//! 2. `vulkan_sigmoid_bf16_matches_cpu`: sigmoid rejected BF16
//!    ("unsupported dtype BF16 for op vulkan sigmoid"), blocking RWKV7 bf16
//!    inference. The fix emulates BF16 through an F32 hub and casts back.
//!
//! Run:
//!   cargo test -p candle-core --features vulkan --test vulkan_encodec_ops_regression

#![cfg(feature = "vulkan")]

use candle_core::{
    CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape, Tensor, VulkanStorage,
};

/// Minimal unary op that routes to the vulkan storage `sigmoid` (the method the
/// RWKV/encodec path uses via candle-nn's `ops::sigmoid`).
struct TestSigmoid;

impl CustomOp1 for TestSigmoid {
    fn name(&self) -> &'static str {
        "test_sigmoid"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // Not used in this test (cpu reference is computed with built-in ops).
        Err(candle_core::Error::Msg("test_sigmoid cpu not used".into()))
    }

    fn vulkan_fwd(
        &self,
        storage: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        let storage = storage.sigmoid(layout)?;
        Ok((storage, layout.shape().clone()))
    }
}

fn vulkan_device() -> Option<Device> {
    Device::new_vulkan(0).ok()
}

/// conv1d where the output length (l_out) exceeds the common Vulkan workgroup
/// limit (maxComputeWorkGroupCount[1] = 65535), exercising the im2col grid-stride
/// clamp. Before the fix the dispatch count was passed unclamped, so the driver
/// only executed a prefix of workgroups and produced garbage for the tail.
#[test]
fn vulkan_conv1d_long_l_matches_cpu() -> Result<()> {
    let vk = match vulkan_device() {
        Some(d) => d,
        None => {
            eprintln!("skipping: vulkan device unavailable");
            return Ok(());
        }
    };
    let cpu = Device::Cpu;

    let l_in = 150_000usize;
    let k = 3usize;
    let cin = 1usize;
    let cout = 4usize;
    let l_out = l_in - k + 1;

    // Deterministic, non-trivial input and a small weight.
    let input: Vec<f32> = (0..l_in).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
    let weight: Vec<f32> = (0..cout * cin * k).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let it_cpu = Tensor::from_vec(input.clone(), (1, cin, l_in), &cpu)?;
    let it_vk = Tensor::from_vec(input, (1, cin, l_in), &vk)?;
    let wt_cpu = Tensor::from_vec(weight.clone(), (cout, cin, k), &cpu)?;
    let wt_vk = Tensor::from_vec(weight, (cout, cin, k), &vk)?;

    let out_cpu = it_cpu.conv1d(&wt_cpu, 0, 1, 1, 1)?.to_vec3::<f32>()?;
    let out_vk = it_vk.conv1d(&wt_vk, 0, 1, 1, 1)?.to_vec3::<f32>()?;

    let flat_cpu: Vec<f32> = out_cpu.into_iter().flatten().flatten().collect();
    let flat_vk: Vec<f32> = out_vk.into_iter().flatten().flatten().collect();

    assert_eq!(flat_cpu.len(), flat_vk.len());
    assert_eq!(flat_cpu.len(), cout * l_out);
    assert!(
        l_out > 65_535,
        "test must exceed the common Vulkan maxComputeWorkGroupCount[1]"
    );

    let (max_abs, max_rel, max_ulp, first_bad) =
        candle_core::test_utils::compare_f32_slices(&flat_vk, &flat_cpu);
    assert!(
        max_abs <= 1e-3,
        "vulkan conv1d (long L) diverged from cpu: max_abs={max_abs} max_rel={max_rel} \
         max_ulp={max_ulp} first_bad={first_bad:?}"
    );
    Ok(())
}

/// sigmoid on a BF16 tensor must succeed on the vulkan device (previously
/// errored) and match an f32-computed CPU reference.
#[test]
fn vulkan_sigmoid_bf16_matches_cpu() -> Result<()> {
    let vk = match vulkan_device() {
        Some(d) => d,
        None => {
            eprintln!("skipping: vulkan device unavailable");
            return Ok(());
        }
    };
    let cpu = Device::Cpu;

    let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.3) - 10.0).collect();

    // Reference: sigmoid of the BF16-quantized input computed in f32, stored in
    // BF16 (so both sides compare the same bf16 value domain).
    let in_f32 = Tensor::from_vec(data.clone(), 64, &cpu)?;
    let in_bf16 = in_f32.to_dtype(DType::BF16)?;
    let in_bf16_as_f32 = in_bf16.to_dtype(DType::F32)?;
    let one = Tensor::new(1.0f32, &cpu)?;
    let ref_f32 = in_bf16_as_f32.neg()?.exp()?.broadcast_add(&one)?.recip()?;
    let ref_out = ref_f32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // Vulkan: bf16 input -> sigmoid (f32-emulated in the backend) -> f32.
    let vk_t = Tensor::from_vec(data, 64, &vk)?.to_dtype(DType::BF16)?;
    let vk_sigmoid = vk_t.apply_op1_no_bwd(&TestSigmoid)?;
    let vk_out = vk_sigmoid.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let (max_abs, max_rel, max_ulp, first_bad) =
        candle_core::test_utils::compare_f32_slices(&vk_out, &ref_out);
    assert!(
        max_abs <= 1e-2,
        "vulkan bf16 sigmoid diverged from cpu f32 reference: max_abs={max_abs} \
         max_rel={max_rel} max_ulp={max_ulp} first_bad={first_bad:?}"
    );
    Ok(())
}
