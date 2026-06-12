mod support;

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use candle_core::Module;
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use half::{bf16, f16};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use support::{backend_device_or_skip, backend_fallback_count};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use support::{fallback_allowed, native_required, TestBackend};

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_expected_gpu_name(actual: &str, backend: &str) {
    if let Some(expected) = std::env::var_os("CANDLE_EXPECTED_GPU_NAME") {
        let expected = expected.to_string_lossy().to_ascii_lowercase();
        assert!(
            actual.to_ascii_lowercase().contains(&expected),
            "{backend} selected {actual:?}, expected a device name containing {expected:?}"
        );
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_expected_backend_name(actual: &str, backend: &str, env_var: &str) {
    if let Some(expected) = std::env::var_os(env_var) {
        let expected = expected.to_string_lossy().to_ascii_lowercase();
        assert!(
            actual.to_ascii_lowercase().contains(&expected),
            "{backend} reported backend {actual:?}, expected a value containing {expected:?}"
        );
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
macro_rules! backend_family_test {
    ($(#[$meta:meta])* $name:ident, $backend:expr, $runner:ident, $family:ident) => {
        $(#[$meta])*
        #[test]
        fn $name() -> Result<()> {
            $runner(stringify!($name), $backend, |device| $family(device))
        }
    };
}

#[test]
#[cfg(feature = "wgpu")]
fn backend_smoke_wgpu_feature_flag_is_visible() {
    assert!(candle_core::utils::wgpu_is_available());
}

#[test]
#[cfg(not(feature = "wgpu"))]
fn backend_smoke_wgpu_feature_flag_is_not_visible() {
    assert!(!candle_core::utils::wgpu_is_available());
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_feature_flag_is_visible() {
    assert!(candle_core::utils::vulkan_is_available());
}

#[test]
#[cfg(not(feature = "vulkan"))]
fn backend_smoke_vulkan_feature_flag_is_not_visible() {
    assert!(!candle_core::utils::vulkan_is_available());
}

#[test]
#[cfg(not(feature = "wgpu"))]
fn backend_smoke_dummy_wgpu_does_not_claim_bf16() {
    let device = candle_core::Device::Wgpu(candle_core::WgpuDevice);
    assert!(!device.supports_bf16());
    assert_eq!(device.bf16_default_to_f32(), candle_core::DType::F32);
}

#[test]
#[cfg(not(feature = "vulkan"))]
fn backend_smoke_dummy_vulkan_does_not_claim_bf16() {
    let device = candle_core::Device::Vulkan(candle_core::VulkanDevice);
    assert!(!device.supports_bf16());
    assert_eq!(device.bf16_default_to_f32(), candle_core::DType::F32);
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_reports_adapter_identity() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let Some(device) = backend_device_or_skip(
        "backend_smoke_wgpu_reports_adapter_identity",
        TestBackend::Wgpu,
    )?
    else {
        return Ok(());
    };
    let device = device.as_wgpu_device()?;
    assert!(!device.adapter_name().is_empty());
    assert!(!device.adapter_backend().is_empty());
    assert_expected_gpu_name(device.adapter_name(), "wgpu");
    assert_expected_backend_name(
        device.adapter_backend(),
        "wgpu",
        "CANDLE_EXPECTED_WGPU_BACKEND",
    );
    Ok(())
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_reports_physical_device_identity() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let Some(device) = backend_device_or_skip(
        "backend_smoke_vulkan_reports_physical_device_identity",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };
    let device = device.as_vulkan_device()?;
    assert!(!device.physical_device_name().is_empty());
    assert_expected_gpu_name(device.physical_device_name(), "vulkan");
    Ok(())
}

backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_upload_and_dtype,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_upload_and_dtype_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_upload_and_dtype,
    TestBackend::Vulkan,
    fallback_allowed,
    smoke_upload_and_dtype_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_unary_binary,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_unary_binary_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_unary_binary,
    TestBackend::Vulkan,
    fallback_allowed,
    smoke_unary_binary_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_reductions,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_reductions_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_reductions,
    TestBackend::Vulkan,
    fallback_allowed,
    smoke_reductions_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_shape_layout,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_shape_layout_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_shape_layout,
    TestBackend::Vulkan,
    fallback_allowed,
    smoke_shape_layout_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_matmul_conv_pool,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_matmul_conv_pool_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_matmul_conv_pool,
    TestBackend::Vulkan,
    fallback_allowed,
    smoke_matmul_conv_pool_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_to_device_transfers_native_only,
    TestBackend::Wgpu,
    native_required,
    smoke_to_device_transfers
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_to_device_transfers_native_only,
    TestBackend::Vulkan,
    native_required,
    smoke_to_device_transfers
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_dtype_conversion_matrix_native_only,
    TestBackend::Wgpu,
    native_required,
    smoke_dtype_conversion_matrix_native_only
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_dtype_conversion_matrix_native_only,
    TestBackend::Vulkan,
    native_required,
    smoke_dtype_conversion_matrix_native_only
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_matmul_shape_sweep_native_only,
    TestBackend::Wgpu,
    native_required,
    smoke_f32_matmul_shape_sweep
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_matmul_shape_sweep_native_only,
    TestBackend::Vulkan,
    native_required,
    smoke_f32_matmul_shape_sweep
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_quantized_paths,
    TestBackend::Wgpu,
    native_required,
    smoke_quantized_family
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_quantized_family,
    TestBackend::Vulkan,
    native_required,
    smoke_quantized_family
);
backend_family_test!(
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a usable wgpu adapter and driver"]
    backend_smoke_wgpu_rank5_fallback_policy,
    TestBackend::Wgpu,
    fallback_allowed,
    smoke_rank5_unary_binary_fallback
);
backend_family_test!(
    #[cfg(feature = "vulkan")]
    backend_smoke_vulkan_rank5_native_policy,
    TestBackend::Vulkan,
    native_required,
    smoke_rank5_unary_binary_native_only
);

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_quantized_paths_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_quantized_paths_only",
        TestBackend::Vulkan,
        smoke_quantized_paths,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_q8_1_qmatmul_regression() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let cpu = Device::Cpu;
    let Some(vk) = backend_device_or_skip(
        "backend_smoke_vulkan_q8_1_qmatmul_regression",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };
    let k = 256;
    let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
    let rhs_vals = (0..(k * 4))
        .map(|v| (v as f32 - 384.0) / 64.0)
        .collect::<Vec<_>>();

    let lhs_cpu = Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
    let rhs_cpu = Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
    let lhs_vk = Tensor::from_slice(&lhs_vals, (1, k), &vk)?;
    let rhs_vk = Tensor::from_slice(&rhs_vals, (k, 4), &vk)?;

    let q_cpu = QTensor::quantize(&rhs_cpu.t()?, GgmlDType::Q8_1)?;
    let q_vk = QTensor::quantize(&rhs_vk.t()?, GgmlDType::Q8_1)?;

    assert_eq!(q_cpu.data()?.as_ref(), q_vk.data()?.as_ref());
    assert_eq!(q_vk.dequantize(&vk)?.shape().dims(), &[4, k]);

    let expected_qmm = QMatMul::from_qtensor(q_cpu)?;
    let actual_qmm = QMatMul::from_qtensor(q_vk)?;
    assert!(matches!(expected_qmm, QMatMul::QTensor(_)));
    assert!(matches!(actual_qmm, QMatMul::QTensor(_)));

    let expected = expected_qmm.forward(&lhs_cpu)?;
    let actual = actual_qmm.forward(&lhs_vk)?;

    let actual_vals = actual.to_vec2::<f32>()?;
    let expected_vals = expected.to_vec2::<f32>()?;
    for (row_idx, (actual_row, expected_row)) in
        actual_vals.iter().zip(expected_vals.iter()).enumerate()
    {
        for (col_idx, (got, want)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
            // Q8_1 quantization introduces lossy rounding. The CPU reference and
            // the Vulkan path use the same stored quantized blocks, but
            // dequantization arithmetic may differ slightly at f32. A relative
            // tolerance of 0.5% (with a minimum absolute floor of 1.0) is
            // consistent with the per-block quantization error expected for Q8_1.
            let abs_err = (got - want).abs();
            let rel_tol = want.abs() * 5e-3;
            let tol = rel_tol.max(1.0);
            assert!(
                abs_err <= tol,
                "integration q8_1 qmatmul mismatch at ({row_idx}, {col_idx}): got {got}, expected {want}, abs_err={abs_err:.4}, tol={tol:.4}"
            );
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_q8_1_quantized_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_q8_1_quantized_native_only",
        TestBackend::Wgpu,
        smoke_q8_1_quantized_native_only,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_q8k_quantized_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_q8k_quantized_native_only",
        TestBackend::Wgpu,
        smoke_q8k_quantized_native_only,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_quantized_dequantize_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_quantized_dequantize_native_only",
        TestBackend::Wgpu,
        smoke_wgpu_quantized_dequantize_native_only,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_q8_1_quantized_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_q8_1_quantized_native_only",
        TestBackend::Vulkan,
        smoke_q8_1_quantized_native_only,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_q8k_quantized_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_q8k_quantized_native_only",
        TestBackend::Vulkan,
        smoke_q8k_quantized_native_only,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_quantized_dequantize_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_quantized_dequantize_native_only",
        TestBackend::Vulkan,
        smoke_vulkan_quantized_dequantize_native_only,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_powf_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_powf_native_only",
        TestBackend::Vulkan,
        |device| {
            let xs = Tensor::from_slice(&[0.25f32, 1.0, 4.0, 9.0], (2, 2), device)?;
            assert_close(
                &xs.powf(1.5)?.to_vec2::<f32>()?,
                &[[0.125, 1.0], [8.0, 27.0]],
                1e-5,
            );
            Ok(())
        },
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_cmp_where_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_cmp_where_native_only",
        TestBackend::Wgpu,
        smoke_f32_cmp_where,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_cmp_where_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_cmp_where_native_only",
        TestBackend::Vulkan,
        smoke_f32_cmp_where,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_int_cmp_where_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_int_cmp_where_native_only",
        TestBackend::Wgpu,
        smoke_int_cmp_where,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_int_cmp_where_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_int_cmp_where_native_only",
        TestBackend::Vulkan,
        smoke_int_cmp_where,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_scatter_add_index_add_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_scatter_add_index_add_native_only",
        TestBackend::Wgpu,
        smoke_f32_scatter_add_and_index_add,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_scatter_add_index_add_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_scatter_add_index_add_native_only",
        TestBackend::Vulkan,
        smoke_f32_scatter_add_and_index_add,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_shape_indexing_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_shape_indexing_native_only",
        TestBackend::Wgpu,
        smoke_f32_gather_scatter_index_non_last_dim_native_only,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_argsort_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_argsort_native_only",
        TestBackend::Wgpu,
        smoke_f32_argsort_last_dim,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_shape_indexing_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_shape_indexing_native_only",
        TestBackend::Vulkan,
        smoke_f32_gather_scatter_index_non_last_dim_native_only,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_argsort_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_argsort_native_only",
        TestBackend::Vulkan,
        smoke_f32_argsort_last_dim,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_upsample_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_upsample_native_only",
        TestBackend::Wgpu,
        smoke_f32_upsample,
    )
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a usable wgpu adapter and driver"]
fn backend_smoke_wgpu_conv_transpose_native_only() -> Result<()> {
    native_required(
        "backend_smoke_wgpu_conv_transpose_native_only",
        TestBackend::Wgpu,
        smoke_f32_conv_transpose,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_upsample_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_upsample_native_only",
        TestBackend::Vulkan,
        smoke_f32_upsample,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_conv_transpose_native_only() -> Result<()> {
    native_required(
        "backend_smoke_vulkan_conv_transpose_native_only",
        TestBackend::Vulkan,
        smoke_f32_conv_transpose,
    )
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_non_zero_start_offset_roundtrip() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let Some(device) = backend_device_or_skip(
        "backend_smoke_vulkan_non_zero_start_offset_roundtrip",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };

    let values = (0..64).map(|v| v as f32 - 32.0).collect::<Vec<_>>();
    let xs = Tensor::from_vec(values, (4, 16), &device)?;
    let view = xs.narrow(1, 3, 10)?;
    let got = view.to_vec2::<f32>()?;

    let expected = (0..4)
        .map(|row| {
            (0..10)
                .map(|col| (row * 16 + col + 3) as f32 - 32.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(got, expected);
    Ok(())
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_boundary_sync_stale_readback_guard() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let Some(device) = backend_device_or_skip(
        "backend_smoke_vulkan_boundary_sync_stale_readback_guard",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };

    let ones = Tensor::ones((64, 64), DType::F32, &device)?;
    let mut xs = Tensor::zeros((64, 64), DType::F32, &device)?;
    for _ in 0..24 {
        xs = xs.broadcast_add(&ones)?;
        // Exercise copy/contiguous paths between compute dispatches.
        xs = xs.transpose(0, 1)?.contiguous()?;
        xs = xs.transpose(0, 1)?.contiguous()?;
    }

    let sample = xs.i((0, 0))?.to_scalar::<f32>()?;
    assert!(
        (sample - 24.0).abs() <= 1e-4,
        "stale readback detected: got {sample}, expected 24.0"
    );
    Ok(())
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_conv1d_multi_channel_regression() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let cpu = Device::Cpu;
    let Some(device) = backend_device_or_skip(
        "backend_smoke_vulkan_conv1d_multi_channel_regression",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };

    let input_vals = (0..(2 * 9))
        .map(|idx| (idx as f32 - 7.0) / 3.0)
        .collect::<Vec<_>>();
    let kernel_vals = (0..(3 * 2 * 3))
        .map(|idx| (idx as f32 - 9.0) / 5.0)
        .collect::<Vec<_>>();

    let input_cpu = Tensor::from_slice(&input_vals, (1, 2, 9), &cpu)?;
    let kernel_cpu = Tensor::from_slice(&kernel_vals, (3, 2, 3), &cpu)?;
    let input_vk = Tensor::from_slice(&input_vals, (1, 2, 9), &device)?;
    let kernel_vk = Tensor::from_slice(&kernel_vals, (3, 2, 3), &device)?;

    let expected = input_cpu.conv1d(&kernel_cpu, 1, 1, 1, 1)?;
    let actual = input_vk.conv1d(&kernel_vk, 1, 1, 1, 1)?;

    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(expected.len(), actual.len());
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-4,
            "multi-channel conv1d mismatch at idx {idx}: got {got}, expected {want}"
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_repeated_shape_reuse_regression_guard() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    let Some(device) = backend_device_or_skip(
        "backend_smoke_vulkan_repeated_shape_reuse_regression_guard",
        TestBackend::Vulkan,
    )?
    else {
        return Ok(());
    };

    candle_core::reset_vulkan_cpu_fallback_count();
    let ones = Tensor::ones((128, 128), DType::F32, &device)?;
    let mut xs = Tensor::zeros((128, 128), DType::F32, &device)?;
    for _ in 0..64 {
        xs = xs.broadcast_add(&ones)?;
        xs = xs.relu()?;
    }

    let sample = xs.i((0, 0))?.to_scalar::<f32>()?;
    assert!(
        (sample - 64.0).abs() <= 1e-4,
        "unexpected repeated-shape accumulation result: got {sample}, expected 64.0"
    );
    assert_eq!(
        candle_core::vulkan_cpu_fallback_count(),
        0,
        "repeated-shape core path triggered vulkan cpu fallback"
    );
    Ok(())
}

#[test]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_device_recreate_teardown_probe() -> Result<()> {
    let _guard = support::fallback_counter_guard();
    for probe_idx in 0..3 {
        let Some(device) = backend_device_or_skip(
            "backend_smoke_vulkan_device_recreate_teardown_probe",
            TestBackend::Vulkan,
        )?
        else {
            return Ok(());
        };
        let xs = Tensor::zeros((32,), DType::F32, &device)?;
        let values = xs.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|&v| v == 0.0),
            "probe {probe_idx}: expected zero tensor after device recreate"
        );
        device.synchronize()?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
#[allow(dead_code)]
fn smoke_f32_upload_unary_binary_roundtrip(device: &Device) -> Result<()> {
    smoke_upload_and_dtype_family(device)?;
    smoke_unary_binary_family(device)?;
    smoke_reductions_family(device)?;
    smoke_shape_layout_family(device)?;
    smoke_matmul_conv_pool_family(device)?;
    if device.is_wgpu() {
        smoke_rank5_unary_binary_fallback(device)?;
    }
    #[cfg(feature = "vulkan")]
    if device.is_vulkan() {
        smoke_rank5_unary_binary_native_only(device)?;
    }
    smoke_quantized_family(device)?;
    device.synchronize()?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_upload_and_dtype_family(device: &Device) -> Result<()> {
    assert!(!device.supports_bf16());
    assert_eq!(device.bf16_default_to_f32(), DType::F32);

    let xs = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 3.0], (2, 2), device)?;
    assert_eq!(xs.to_vec2::<f32>()?, [[-2.0, -1.0], [0.0, 3.0]]);

    let zeros = Tensor::zeros((2, 2), DType::F32, device)?;
    assert_eq!(zeros.to_vec2::<f32>()?, [[0.0, 0.0], [0.0, 0.0]]);

    assert_eq!(xs.relu()?.to_vec2::<f32>()?, [[0.0, 0.0], [0.0, 3.0]]);

    let ys = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), device)?;
    assert_eq!((&xs + &ys)?.to_vec2::<f32>()?, [[8.0, 19.0], [30.0, 43.0]]);

    smoke_non_f32_upload_download(device)?;
    smoke_f32_to_i32_dtype_conversion(device)?;
    smoke_f32_f16_dtype_conversion(device)?;
    #[cfg(feature = "vulkan")]
    if device.is_vulkan() {
        smoke_i32_to_f32_dtype_conversion(device)?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_unary_binary_family(device: &Device) -> Result<()> {
    smoke_f32_large_linear_elementwise(device)?;
    smoke_f16_elementwise_ops(device)?;
    smoke_f32_binary_broadcast_and_strided_layout(device)?;
    smoke_f32_extended_unary_ops(device)?;
    if device.is_wgpu() {
        smoke_strided_const_set(device)?;
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_reductions_family(device: &Device) -> Result<()> {
    smoke_f32_sum_last_dim(device)?;
    smoke_f32_cumsum(device)?;
    smoke_f32_argmax_last_dim(device)?;
    smoke_f32_extrema_last_dim(device)?;
    smoke_f32_argsort_last_dim(device)?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_shape_layout_family(device: &Device) -> Result<()> {
    smoke_strided_contiguous_copy(device)?;
    smoke_f32_cat_repeat_pad(device)?;
    smoke_f32_index_select(device)?;
    smoke_f32_gather_last_dim(device)?;
    smoke_f32_scatter_set_last_dim(device)?;
    smoke_f32_gather_scatter_non_last_dim(device)?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_matmul_conv_pool_family(device: &Device) -> Result<()> {
    smoke_f32_matmul(device)?;
    smoke_f32_conv1d(device)?;
    smoke_f32_conv2d(device)?;
    smoke_f32_conv_transpose(device)?;
    smoke_f32_upsample(device)?;
    smoke_f32_pool2d(device)?;
    smoke_f32_cmp_where(device)?;
    smoke_f32_scatter_add_and_index_add(device)?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_quantized_family(device: &Device) -> Result<()> {
    smoke_quantized_paths(device)
}

/// Shape sweep over the dense f32 GEMM routes. The `m` axis crosses the
/// matvec-vs-tiled routing threshold (8) and the tile sizes (32/64), the `n`/`k`
/// axes cross workgroup-tile and K-loop boundaries, and odd sizes exercise the
/// out-of-bounds guards in the tiled shader. A routing or specialization bug in
/// any one route shows up as a large numeric divergence from the CPU result.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_matmul_shape_sweep(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let cases: &[(usize, usize, usize)] = &[
        (1, 4, 7),
        (8, 16, 16),
        (9, 4, 7),
        (16, 16, 16),
        (33, 65, 129),
        (64, 64, 64),
        (100, 50, 30),
        (96, 100, 1024),
        (128, 196, 256),
        (196, 256, 9),
        (256, 196, 1152),
        (1000, 64, 1),
    ];
    for &(m, n, k) in cases {
        let a_vals: Vec<f32> = (0..m * k)
            .map(|i| ((i % 71) as f32 - 35.0) / 17.0)
            .collect();
        let b_vals: Vec<f32> = (0..k * n)
            .map(|i| ((i % 53) as f32 - 26.0) / 13.0)
            .collect();
        let a_cpu = Tensor::from_vec(a_vals.clone(), (m, k), &cpu)?;
        let b_cpu = Tensor::from_vec(b_vals.clone(), (k, n), &cpu)?;
        let expected = a_cpu.matmul(&b_cpu)?;
        let a_dev = Tensor::from_vec(a_vals, (m, k), device)?;
        let b_dev = Tensor::from_vec(b_vals, (k, n), device)?;
        let actual = a_dev.matmul(&b_dev)?;
        assert_matmul_close(&actual, &expected, k, &format!("matmul {m}x{n}x{k}"))?;

        // Transposed rhs exercises the strided-input materialization path.
        let bt_cpu = b_cpu.t()?.contiguous()?;
        let bt_dev = b_dev.t()?.contiguous()?;
        let expected_t = a_cpu.matmul(&bt_cpu.t()?)?;
        let actual_t = a_dev.matmul(&bt_dev.t()?)?;
        assert_matmul_close(&actual_t, &expected_t, k, &format!("matmul-tr {m}x{n}x{k}"))?;
    }
    // Batched matmul crossing the same routing threshold per batch.
    for &(b, m, n, k) in &[(2usize, 4usize, 8usize, 16usize), (3, 33, 17, 64)] {
        let a_vals: Vec<f32> = (0..b * m * k)
            .map(|i| ((i % 61) as f32 - 30.0) / 11.0)
            .collect();
        let b_vals: Vec<f32> = (0..b * k * n)
            .map(|i| ((i % 47) as f32 - 23.0) / 7.0)
            .collect();
        let a_cpu = Tensor::from_vec(a_vals.clone(), (b, m, k), &cpu)?;
        let b_cpu = Tensor::from_vec(b_vals.clone(), (b, k, n), &cpu)?;
        let expected = a_cpu.matmul(&b_cpu)?;
        let a_dev = Tensor::from_vec(a_vals, (b, m, k), device)?;
        let b_dev = Tensor::from_vec(b_vals, (b, k, n), device)?;
        let actual = a_dev.matmul(&b_dev)?;
        assert_matmul_close(&actual, &expected, k, &format!("bmm {b}x{m}x{n}x{k}"))?;
    }
    Ok(())
}

/// Full dtype-conversion matrix over the CUDA-parity dtype set. Each pair must
/// run natively (no recorded CPU fallback) and match the CPU `as`-cast
/// semantics. BF16 destinations are remapped to F16 by the storage layer on
/// these backends, which the CPU reference mirrors here.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_dtype_conversion_matrix_native_only(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::U8,
        DType::U32,
        DType::I64,
    ];
    let vals: Vec<f32> = (0..64).map(|v| v as f32 / 8.0).collect();
    for src in dtypes {
        for dst in dtypes {
            let dev_t = Tensor::from_vec(vals.clone(), 64, device)?.to_dtype(src)?;
            let cpu_t = Tensor::from_vec(vals.clone(), 64, &cpu)?.to_dtype(src)?;
            let dst_cpu = if dst == DType::BF16 { DType::F16 } else { dst };
            let got = dev_t
                .to_dtype(dst)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = cpu_t
                .to_dtype(dst_cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 0.011,
                    "dtype matrix {src:?}->{dst:?}: mismatch at idx {idx}: got {g}, expected {w}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_matmul_close(actual: &Tensor, expected: &Tensor, k: usize, case: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{case}: shape mismatch");
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let scale = expected.iter().fold(1f32, |acc, v| acc.max(v.abs()));
    // f32 accumulation error grows with K; scale the relative tolerance.
    let tol = scale * 1e-6 * (k as f32).sqrt().max(1.0) + 1e-5;
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "{case}: mismatch at idx {idx}: got {got}, expected {want}, diff {diff}, tol {tol}"
        );
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_q8_1_quantized_native_only(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let dtype = GgmlDType::Q8_1;
    let k = 256;

    let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
    let rhs_vals = (0..(k * 4))
        .map(|v| (v as f32 - 384.0) / 64.0)
        .collect::<Vec<_>>();
    let lhs_cpu = Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
    let rhs_cpu = Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
    let lhs = Tensor::from_slice(&lhs_vals, (1, k), device)?;
    let rhs = Tensor::from_slice(&rhs_vals, (k, 4), device)?;

    let q_rhs = QTensor::quantize(&rhs.t()?, dtype)?;
    let qmm = QMatMul::from_qtensor(q_rhs)?;
    let expected_mm = expected_quantized_matmul(device, &rhs_cpu, &lhs_cpu, dtype)?;
    let actual_mm = qmm.forward(&lhs)?;
    assert_quantized_close(&actual_mm, &expected_mm, dtype, "vulkan-q8_1-matvec")?;

    let ids = Tensor::from_slice(&[3u32, 1, 0], 3, device)?;
    let ids_cpu = Tensor::from_slice(&[3u32, 1, 0], 3, &cpu)?;
    let q_rows = QTensor::quantize(&rhs.t()?, dtype)?;
    let q_rows_cpu = QTensor::quantize(&rhs_cpu.t()?, dtype)?;
    let actual_rows = q_rows.embedding(&ids)?;
    let expected_rows = q_rows_cpu.dequantize(&cpu)?.index_select(&ids_cpu, 0)?;
    assert_quantized_close(&actual_rows, &expected_rows, dtype, "vulkan-q8_1-get-rows")?;

    let moe_w_vals = (0..(2 * 3 * k))
        .map(|v| (v as f32 - 3.0 * k as f32) / 128.0)
        .collect::<Vec<_>>();
    let moe_x_vals = (0..(2 * k))
        .map(|v| (v as f32 - k as f32 / 2.0) / 16.0)
        .collect::<Vec<_>>();
    let moe_w = Tensor::from_slice(&moe_w_vals, (2, 3, k), device)?;
    let moe_x = Tensor::from_slice(&moe_x_vals, (2, k), device)?;
    let moe_ids = Tensor::from_slice(&[0u32, 1, 1, 0], (2, 2), device)?;
    let moe_w_cpu = Tensor::from_slice(&moe_w_vals, (2, 3, k), &cpu)?;
    let moe_x_cpu = Tensor::from_slice(&moe_x_vals, (2, k), &cpu)?;
    let moe_ids_cpu = Tensor::from_slice(&[0u32, 1, 1, 0], (2, 2), &cpu)?;
    let q_moe = QTensor::quantize(&moe_w, dtype)?;
    let q_moe_cpu = QTensor::quantize(&moe_w_cpu, dtype)?;
    let expected_moe = if device.is_vulkan()
        && vulkan_uses_q8_1_rhs_for_indexed_moe(device, dtype, moe_ids_cpu.dims()[0], k)
    {
        q8_1_activation_indexed_moe_reference(&q_moe_cpu, &moe_x_cpu, &moe_ids_cpu)?
    } else {
        q_moe_cpu.indexed_moe_forward(&moe_x_cpu, &moe_ids_cpu)?
    };
    let actual_moe = q_moe.indexed_moe_forward(&moe_x, &moe_ids)?;
    let label = if device.is_wgpu() {
        "wgpu-q8_1-indexed-moe"
    } else {
        "vulkan-q8_1-indexed-moe"
    };
    assert_quantized_close(&actual_moe, &expected_moe, dtype, label)?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_q8k_quantized_native_only(device: &Device) -> Result<()> {
    let k = 256;
    let cpu = Device::Cpu;
    let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
    let rhs_vals = (0..(k * 4))
        .map(|v| (v as f32 - 384.0) / 64.0)
        .collect::<Vec<_>>();
    let lhs_cpu = Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
    let rhs_cpu = Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
    let lhs = Tensor::from_slice(&lhs_vals, (1, k), device)?;
    let rhs = Tensor::from_slice(&rhs_vals, (k, 4), device)?;

    let q_rhs = QTensor::quantize(&rhs.t()?, GgmlDType::Q8K)?;
    let qmm = QMatMul::from_qtensor(q_rhs)?;
    let expected_mm = expected_quantized_matmul(device, &rhs_cpu, &lhs_cpu, GgmlDType::Q8K)?;
    let actual_mm = qmm.forward(&lhs)?;
    assert_quantized_close(&actual_mm, &expected_mm, GgmlDType::Q8K, "q8k-matvec")?;

    let ids = Tensor::from_slice(&[3u32, 1, 0], 3, device)?;
    let ids_cpu = Tensor::from_slice(&[3u32, 1, 0], 3, &cpu)?;
    let q_rows = QTensor::quantize(&rhs.t()?, GgmlDType::Q8K)?;
    let q_rows_cpu = QTensor::quantize(&rhs_cpu.t()?, GgmlDType::Q8K)?;
    let actual_rows = q_rows.embedding(&ids)?;
    let expected_rows = q_rows_cpu.dequantize(&cpu)?.index_select(&ids_cpu, 0)?;
    assert_quantized_close(&actual_rows, &expected_rows, GgmlDType::Q8K, "q8k-get-rows")?;
    Ok(())
}

#[cfg(feature = "wgpu")]
fn smoke_wgpu_quantized_dequantize_native_only(device: &Device) -> Result<()> {
    let dtypes = [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];
    smoke_quantized_dequantize_native_only(device, &dtypes, "wgpu-dequantize")
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_quantized_dequantize_native_only(
    device: &Device,
    dtypes: &[GgmlDType],
    label: &str,
) -> Result<()> {
    let cpu = Device::Cpu;
    for &dtype in dtypes {
        let cols = dtype.block_size() * 2;
        let rows = 3usize;
        let vals = (0..rows * cols)
            .map(|idx| (idx as f32 - (rows * cols) as f32 / 2.0) / 37.0)
            .collect::<Vec<_>>();
        let src_cpu = Tensor::from_slice(&vals, (rows, cols), &cpu)?;
        let src_gpu = Tensor::from_slice(&vals, (rows, cols), device)?;
        let q_cpu = QTensor::quantize(&src_cpu, dtype)?;
        let q_gpu = QTensor::quantize(&src_gpu, dtype)?;
        let expected = q_cpu.dequantize(&cpu)?;
        let actual = q_gpu.dequantize(device)?;
        assert_quantized_close(&actual, &expected, dtype, label)?;
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn smoke_vulkan_quantized_dequantize_native_only(device: &Device) -> Result<()> {
    let dtypes = [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];
    smoke_quantized_dequantize_native_only(device, &dtypes, "vulkan-dequantize")
}

#[cfg(feature = "vulkan")]
fn smoke_rank5_unary_binary_native_only(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(
        &[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0],
        (1, 1, 1, 2, 3),
        device,
    )?;
    let ys = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (1, 1, 1, 1, 3), device)?;

    // Core Vulkan paths are strict: unsupported rank/layout should return errors
    // instead of silently falling back to CPU.
    let _ = xs.relu();
    let _ = xs.broadcast_add(&ys);
    assert_eq!(
        backend_fallback_count(TestBackend::Vulkan),
        0,
        "vulkan core paths must not trigger silent CPU fallback"
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_large_linear_elementwise(device: &Device) -> Result<()> {
    let values = (0..4096)
        .map(|idx| idx as f32 / 128.0 - 8.0)
        .collect::<Vec<_>>();
    let xs = Tensor::from_slice(&values, (1, 4, 1024), device)?;
    let zeros = Tensor::zeros((1, 4, 1024), DType::F32, device)?;
    assert_close_vec(
        &xs.add(&zeros)?.flatten_all()?.to_vec1::<f32>()?,
        &values,
        1e-4,
        "large linear add",
    );
    let expected_affine = values
        .iter()
        .map(|value| value * 0.5 + 1.25)
        .collect::<Vec<_>>();
    assert_close_vec(
        &xs.affine(0.5, 1.25)?.flatten_all()?.to_vec1::<f32>()?,
        &expected_affine,
        1e-4,
        "large linear affine",
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_to_device_transfers(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let values = support::deterministic_f32_data(24, 7);
    let cpu_tensor = Tensor::from_vec(values.clone(), (4, 6), &cpu)?;

    // CPU -> GPU via Tensor::to_device (not just from_vec upload).
    let gpu_tensor = cpu_tensor.to_device(device)?;
    assert!(gpu_tensor.device().same_device(device));
    assert_eq!(gpu_tensor.flatten_all()?.to_vec1::<f32>()?, values);

    // GPU -> CPU via Tensor::to_device (not just to_vec download).
    let back = gpu_tensor.to_device(&cpu)?;
    assert!(back.device().is_cpu());
    assert_eq!(back.flatten_all()?.to_vec1::<f32>()?, values);

    // Same-device transfer takes the shallow-clone early return.
    let same = gpu_tensor.to_device(device)?;
    assert!(same.device().same_device(device));
    assert_eq!(same.flatten_all()?.to_vec1::<f32>()?, values);

    // Roundtrip on a strided view keeps layout semantics.
    let view = gpu_tensor.t()?;
    let view_back = view.to_device(&cpu)?;
    assert_eq!(
        view_back.contiguous()?.flatten_all()?.to_vec1::<f32>()?,
        cpu_tensor
            .t()?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?
    );

    // Integer and half dtypes must transfer unchanged.
    let u32_cpu = Tensor::from_slice(&[1u32, 2, 3, 4000], (2, 2), &cpu)?;
    let u32_back = u32_cpu.to_device(device)?.to_device(&cpu)?;
    assert_eq!(u32_back.to_vec2::<u32>()?, [[1, 2], [3, 4000]]);

    let f16_cpu = Tensor::from_slice(
        &[
            f16::from_f32(0.5),
            f16::from_f32(-1.5),
            f16::from_f32(2.0),
            f16::from_f32(-4.0),
        ],
        (2, 2),
        &cpu,
    )?;
    let f16_back = f16_cpu.to_device(device)?.to_device(&cpu)?;
    assert_eq!(
        f16_back.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.5), f16::from_f32(-1.5)],
            [f16::from_f32(2.0), f16::from_f32(-4.0)]
        ]
    );

    // BF16 normalizes to F16 on wgpu/Vulkan; dtype must follow the storage.
    let bf16_cpu = Tensor::from_slice(
        &[
            bf16::from_f32(0.25),
            bf16::from_f32(-1.0),
            bf16::from_f32(3.0),
            bf16::from_f32(-8.0),
        ],
        (2, 2),
        &cpu,
    )?;
    let bf16_gpu = bf16_cpu.to_device(device)?;
    assert_eq!(bf16_gpu.dtype(), DType::F16);
    let bf16_back = bf16_gpu.to_device(&cpu)?.to_dtype(DType::F32)?;
    assert_eq!(bf16_back.to_vec2::<f32>()?, [[0.25, -1.0], [3.0, -8.0]]);
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_non_f32_upload_download(device: &Device) -> Result<()> {
    let u8s = Tensor::from_slice(&[1u8, 2, 3, 4], (2, 2), device)?;
    assert_eq!(u8s.to_vec2::<u8>()?, [[1, 2], [3, 4]]);
    assert_eq!(
        Tensor::zeros((2, 2), DType::U8, device)?.to_vec2::<u8>()?,
        [[0, 0], [0, 0]]
    );

    let u32s = Tensor::from_slice(&[1u32, 20, 300, 4000], (2, 2), device)?;
    assert_eq!(u32s.to_vec2::<u32>()?, [[1, 20], [300, 4000]]);
    assert_eq!(
        Tensor::zeros((2, 2), DType::U32, device)?.to_vec2::<u32>()?,
        [[0, 0], [0, 0]]
    );

    let i64s = Tensor::from_slice(&[-1i64, 2, -3, 4], (2, 2), device)?;
    assert_eq!(i64s.to_vec2::<i64>()?, [[-1, 2], [-3, 4]]);
    assert_eq!(
        Tensor::zeros((2, 2), DType::I64, device)?.to_vec2::<i64>()?,
        [[0, 0], [0, 0]]
    );

    let f16s = Tensor::from_slice(
        &[
            f16::from_f32(0.5),
            f16::from_f32(1.5),
            f16::from_f32(-2.0),
            f16::from_f32(4.0),
        ],
        (2, 2),
        device,
    )?;
    assert_eq!(
        f16s.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.5), f16::from_f32(1.5)],
            [f16::from_f32(-2.0), f16::from_f32(4.0)]
        ]
    );
    assert_eq!(
        Tensor::zeros((2, 2), DType::F16, device)?.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.0), f16::from_f32(0.0)],
            [f16::from_f32(0.0), f16::from_f32(0.0)]
        ]
    );

    let bf16s = Tensor::from_slice(
        &[
            bf16::from_f32(0.5),
            bf16::from_f32(1.5),
            bf16::from_f32(-2.0),
            bf16::from_f32(4.0),
        ],
        (2, 2),
        device,
    )?;
    assert_eq!(bf16s.dtype(), DType::F16);
    assert_eq!(
        bf16s.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.5), f16::from_f32(1.5)],
            [f16::from_f32(-2.0), f16::from_f32(4.0)]
        ]
    );
    let bf16_zeros = Tensor::zeros((2, 2), DType::BF16, device)?;
    assert_eq!(bf16_zeros.dtype(), DType::F16);
    assert_eq!(
        bf16_zeros.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.0), f16::from_f32(0.0)],
            [f16::from_f32(0.0), f16::from_f32(0.0)]
        ]
    );
    let f32_to_bf16 =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?.to_dtype(DType::BF16)?;
    assert_eq!(f32_to_bf16.dtype(), DType::F16);
    assert_eq!(
        f32_to_bf16.to_vec2::<f16>()?,
        [
            [f16::from_f32(1.0), f16::from_f32(2.0)],
            [f16::from_f32(3.0), f16::from_f32(4.0)]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_to_i32_dtype_conversion(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, -2.0, 3.75, 4.25], (2, 2), device)?;
    assert_eq!(
        xs.to_dtype(DType::I32)?.to_vec2::<i32>()?,
        [[1, -2], [3, 4]]
    );

    let transposed = xs.t()?;
    assert_eq!(
        transposed.to_dtype(DType::I32)?.to_vec2::<i32>()?,
        [[1, 3], [-2, 4]]
    );

    Ok(())
}

#[cfg(feature = "vulkan")]
fn smoke_i32_to_f32_dtype_conversion(device: &Device) -> Result<()> {
    let ints = Tensor::from_slice(&[1i32, -2, 3, 4], (2, 2), device)?;
    assert_eq!(
        ints.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        [[1.0, -2.0], [3.0, 4.0]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_f16_dtype_conversion(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[0.5f32, -1.5, 3.0, 4.25], (2, 2), device)?;
    assert_eq!(
        xs.to_dtype(DType::F16)?.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.5), f16::from_f32(-1.5)],
            [f16::from_f32(3.0), f16::from_f32(4.25)]
        ]
    );
    assert_eq!(
        xs.to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
        [[0.5, -1.5], [3.0, 4.25]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f16_elementwise_ops(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(
        &[
            f16::from_f32(0.25),
            f16::from_f32(1.0),
            f16::from_f32(-2.0),
            f16::from_f32(4.0),
        ],
        (2, 2),
        device,
    )?;

    assert_close(
        &xs.relu()?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        &[[0.25, 1.0], [0.0, 4.0]],
        1e-3,
    );
    assert_close(
        &xs.neg()?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        &[[-0.25, -1.0], [2.0, -4.0]],
        1e-3,
    );

    let ys = Tensor::from_slice(
        &[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        (2, 2),
        device,
    )?;
    assert_close(
        &xs.add(&ys)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        &[[1.25, 3.0], [1.0, 8.0]],
        1e-3,
    );
    assert_close(
        &xs.mul(&ys)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        &[[0.25, 2.0], [-6.0, 16.0]],
        1e-3,
    );
    assert_close(
        &xs.elu(1.0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        &[[0.25, 1.0], [std::f32::consts::E.powf(-2.0) - 1.0, 4.0]],
        2e-3,
    );

    let positive = Tensor::from_slice(
        &[
            f16::from_f32(0.25),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
        ],
        (2, 2),
        device,
    )?;
    assert_close(
        &positive
            .log()?
            .exp()?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
        &[[0.25, 1.0], [2.0, 4.0]],
        3e-3,
    );

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_binary_broadcast_and_strided_layout(device: &Device) -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    let rhs = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (1, 3), device)?;
    assert_eq!(
        lhs.broadcast_add(&rhs)?.to_vec2::<f32>()?,
        [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
    );

    let transposed = lhs.t()?;
    let row = Tensor::from_slice(&[10.0f32, 20.0], (1, 2), device)?;
    assert_eq!(
        transposed.broadcast_add(&row)?.to_vec2::<f32>()?,
        [[11.0, 24.0], [12.0, 25.0], [13.0, 26.0]]
    );

    let lhs = Tensor::from_slice(&[1.0f32, 5.0, -2.0, 4.0], (2, 2), device)?;
    let rhs = Tensor::from_slice(&[0.0f32, 6.0], (1, 2), device)?;
    assert_eq!(
        lhs.broadcast_maximum(&rhs)?.to_vec2::<f32>()?,
        [[1.0, 6.0], [0.0, 6.0]]
    );
    assert_eq!(
        lhs.broadcast_minimum(&rhs)?.to_vec2::<f32>()?,
        [[0.0, 5.0], [-2.0, 4.0]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_strided_contiguous_copy(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    assert_eq!(
        xs.t()?.contiguous()?.to_vec2::<f32>()?,
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    );

    let halves = Tensor::from_slice(
        &[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        (2, 3),
        device,
    )?;
    assert_eq!(
        halves.t()?.contiguous()?.to_vec2::<f16>()?,
        [
            [f16::from_f32(1.0), f16::from_f32(4.0)],
            [f16::from_f32(2.0), f16::from_f32(5.0)],
            [f16::from_f32(3.0), f16::from_f32(6.0)]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_cat_repeat_pad(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    let ys = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), device)?;
    assert_eq!(
        Tensor::cat(&[&xs, &ys], 1)?.to_vec2::<f32>()?,
        [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]
    );
    assert_eq!(
        xs.repeat((1, 2))?.to_vec2::<f32>()?,
        [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]
    );
    assert_eq!(
        xs.pad_with_zeros(1, 1, 2)?.to_vec2::<f32>()?,
        [[0.0, 1.0, 2.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0, 0.0]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_strided_const_set(device: &Device) -> Result<()> {
    let xs = Tensor::zeros((2, 3), DType::F32, device)?;
    xs.i((.., 1))?.const_set(7.0f32.into())?;
    assert_eq!(xs.to_vec2::<f32>()?, [[0.0, 7.0, 0.0], [0.0, 7.0, 0.0]]);

    let halves = Tensor::zeros((2, 3), DType::F16, device)?;
    halves.i((1, ..))?.const_set(f16::from_f32(2.5).into())?;
    assert_eq!(
        halves.to_vec2::<f16>()?,
        [
            [f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0)],
            [f16::from_f32(2.5), f16::from_f32(2.5), f16::from_f32(2.5)],
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_sum_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    assert_eq!(xs.sum_keepdim(1)?.to_vec2::<f32>()?, [[6.0], [15.0]]);
    assert_eq!(xs.sum_keepdim(0)?.to_vec2::<f32>()?, [[5.0, 7.0, 9.0]]);
    assert_eq!(xs.sum_keepdim((0, 1))?.to_vec2::<f32>()?, [[21.0]]);
    assert_eq!(xs.sum_all()?.to_scalar::<f32>()?, 21.0);

    let ys = Tensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
        ],
        (3, 2, 2),
        device,
    )?;
    assert_eq!(
        ys.sum_keepdim(2)?.to_vec3::<f32>()?,
        [[[3.0], [7.0]], [[30.0], [70.0]], [[300.0], [700.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_cumsum(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3), device)?;
    assert_eq!(
        xs.cumsum(1)?.to_vec2::<f32>()?,
        [[1.0, 3.0, 6.0], [10.0, 30.0, 60.0]]
    );
    assert_eq!(
        xs.cumsum(0)?.to_vec2::<f32>()?,
        [[1.0, 2.0, 3.0], [11.0, 22.0, 33.0]]
    );

    let ys = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 2, 2),
        device,
    )?;
    assert_eq!(
        ys.cumsum(2)?.to_vec3::<f32>()?,
        [[[1.0, 3.0], [3.0, 7.0]], [[10.0, 30.0], [30.0, 70.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_argmax_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 4.0, 2.0, 9.0, 3.0, 5.0], (2, 3), device)?;
    assert_eq!(xs.argmax_keepdim(1)?.to_vec2::<u32>()?, [[1], [0]]);
    assert_eq!(xs.argmax_keepdim(0)?.to_vec2::<u32>()?, [[1, 0, 1]]);

    let ys = Tensor::from_slice(
        &[1.0f32, 7.0, 3.0, 4.0, 8.0, 5.0, 6.0, 2.0],
        (2, 2, 2),
        device,
    )?;
    assert_eq!(
        ys.argmax_keepdim(2)?.to_vec3::<u32>()?,
        [[[1], [1]], [[0], [0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_extrema_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, -3.0, 2.0, 4.0, 0.0, -5.0], (2, 3), device)?;
    assert_eq!(xs.max_keepdim(1)?.to_vec2::<f32>()?, [[2.0], [4.0]]);
    assert_eq!(xs.max(1)?.to_vec1::<f32>()?, [2.0, 4.0]);
    assert_eq!(xs.max_keepdim(0)?.to_vec2::<f32>()?, [[4.0, 0.0, 2.0]]);
    assert_eq!(xs.min_keepdim(1)?.to_vec2::<f32>()?, [[-3.0], [-5.0]]);
    assert_eq!(xs.min(1)?.to_vec1::<f32>()?, [-3.0, -5.0]);
    assert_eq!(xs.min_keepdim(0)?.to_vec2::<f32>()?, [[1.0, -3.0, -5.0]]);
    assert_eq!(xs.argmin_keepdim(1)?.to_vec2::<u32>()?, [[1], [2]]);
    assert_eq!(xs.argmin(1)?.to_vec1::<u32>()?, [1, 2]);
    assert_eq!(xs.argmin_keepdim(0)?.to_vec2::<u32>()?, [[0, 0, 1]]);
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_argsort_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[3.0f32, 1.0, 2.0, 10.0, 5.0, 7.0], (2, 3), device)?;
    assert_eq!(
        xs.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        [[1, 2, 0], [1, 2, 0]]
    );
    assert_eq!(
        xs.arg_sort_last_dim(false)?.to_vec2::<u32>()?,
        [[0, 2, 1], [0, 2, 1]]
    );

    let (sorted, idx) = xs.sort_last_dim(true)?;
    assert_eq!(idx.to_vec2::<u32>()?, [[1, 2, 0], [1, 2, 0]]);
    assert_eq!(
        sorted.to_vec2::<f32>()?,
        [[1.0, 2.0, 3.0], [5.0, 7.0, 10.0]]
    );

    let large_values = (0..300).rev().map(|v| v as f32).collect::<Vec<_>>();
    let large = Tensor::from_vec(large_values, (1, 300), device)?;
    let asc = large.arg_sort_last_dim(true)?.to_vec2::<u32>()?;
    assert_eq!(&asc[0][..5], &[299, 298, 297, 296, 295]);
    assert_eq!(&asc[0][295..], &[4, 3, 2, 1, 0]);
    let desc = large.arg_sort_last_dim(false)?.to_vec2::<u32>()?;
    assert_eq!(&desc[0][..5], &[0, 1, 2, 3, 4]);
    assert_eq!(&desc[0][295..], &[295, 296, 297, 298, 299]);

    if device.is_vulkan() {
        let vulkan_large_values = (0..1100).rev().map(|v| v as f32).collect::<Vec<_>>();
        let vulkan_large = Tensor::from_vec(vulkan_large_values, (1, 1100), device)?;
        let asc = vulkan_large.arg_sort_last_dim(true)?.to_vec2::<u32>()?;
        assert_eq!(&asc[0][..5], &[1099, 1098, 1097, 1096, 1095]);
        assert_eq!(&asc[0][1095..], &[4, 3, 2, 1, 0]);
        let desc = vulkan_large.arg_sort_last_dim(false)?.to_vec2::<u32>()?;
        assert_eq!(&desc[0][..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&desc[0][1095..], &[1095, 1096, 1097, 1098, 1099]);
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_index_select(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3), device)?;
    let ids = Tensor::from_slice(&[1u32, 0], (2,), device)?;
    assert_eq!(
        xs.index_select(&ids, 0)?.to_vec2::<f32>()?,
        [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]
    );
    assert_eq!(
        xs.index_select(&ids, 1)?.to_vec2::<f32>()?,
        [[2.0, 1.0], [20.0, 10.0]]
    );

    let ys = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 2, 2),
        device,
    )?;
    assert_eq!(
        ys.index_select(&ids, 1)?.to_vec3::<f32>()?,
        [[[3.0, 4.0], [1.0, 2.0]], [[30.0, 40.0], [10.0, 20.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_gather_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3), device)?;
    let ids = Tensor::from_slice(&[2u32, 0, 1, 1], (2, 2), device)?;
    assert_eq!(
        xs.gather(&ids, 1)?.to_vec2::<f32>()?,
        [[3.0, 1.0], [20.0, 20.0]]
    );

    let xs_f16 = Tensor::from_slice(
        &[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        (2, 3),
        device,
    )?;
    assert_eq!(
        xs_f16.gather(&ids, 1)?.to_vec2::<f16>()?,
        [
            [f16::from_f32(3.0), f16::from_f32(1.0)],
            [f16::from_f32(20.0), f16::from_f32(20.0)]
        ]
    );

    let ys = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 2, 2),
        device,
    )?;
    let ids = Tensor::from_slice(&[1u32, 0, 0, 1, 1, 1, 0, 0], (2, 2, 2), device)?;
    assert_eq!(
        ys.gather(&ids, 2)?.to_vec3::<f32>()?,
        [[[2.0, 1.0], [3.0, 4.0]], [[20.0, 20.0], [30.0, 30.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_scatter_set_last_dim(device: &Device) -> Result<()> {
    let base = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 4),
        device,
    )?;
    let ids = Tensor::from_slice(&[2u32, 0, 1, 3], (2, 2), device)?;
    let src = Tensor::from_slice(&[30.0f32, 10.0, 200.0, 400.0], (2, 2), device)?;
    assert_eq!(
        base.scatter(&ids, &src, 1)?.to_vec2::<f32>()?,
        [[10.0, 2.0, 30.0, 4.0], [10.0, 200.0, 30.0, 400.0]]
    );

    let dst = Tensor::zeros((2, 4), DType::F32, device)?;
    dst.scatter_set(&ids, &src, 1)?;
    assert_eq!(
        dst.to_vec2::<f32>()?,
        [[10.0, 0.0, 30.0, 0.0], [0.0, 200.0, 0.0, 400.0]]
    );

    let dst_f16 = Tensor::zeros((2, 4), DType::F16, device)?;
    let src_f16 = Tensor::from_slice(
        &[
            f16::from_f32(30.0),
            f16::from_f32(10.0),
            f16::from_f32(200.0),
            f16::from_f32(400.0),
        ],
        (2, 2),
        device,
    )?;
    dst_f16.scatter_set(&ids, &src_f16, 1)?;
    assert_eq!(
        dst_f16.to_vec2::<f16>()?,
        [
            [
                f16::from_f32(10.0),
                f16::from_f32(0.0),
                f16::from_f32(30.0),
                f16::from_f32(0.0)
            ],
            [
                f16::from_f32(0.0),
                f16::from_f32(200.0),
                f16::from_f32(0.0),
                f16::from_f32(400.0)
            ]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_gather_scatter_non_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], (3, 2), device)?;
    let ids = Tensor::from_slice(&[0u32, 1, 2, 0], (2, 2), device)?;

    let cpu = Device::Cpu;
    let xs_cpu = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], (3, 2), &cpu)?;
    let ids_cpu = Tensor::from_slice(&[0u32, 1, 2, 0], (2, 2), &cpu)?;
    let expected_gather = xs_cpu.gather(&ids_cpu, 0)?.to_vec2::<f32>()?;
    assert_eq!(xs.gather(&ids, 0)?.to_vec2::<f32>()?, expected_gather);

    let base = Tensor::zeros((3, 2), DType::F32, device)?;
    let src = Tensor::from_slice(&[7.0f32, 70.0, 8.0, 80.0], (2, 2), device)?;
    let base_cpu = Tensor::zeros((3, 2), DType::F32, &cpu)?;
    let src_cpu = Tensor::from_slice(&[7.0f32, 70.0, 8.0, 80.0], (2, 2), &cpu)?;
    let expected_scatter = base_cpu.scatter(&ids_cpu, &src_cpu, 0)?.to_vec2::<f32>()?;
    assert_eq!(
        base.scatter(&ids, &src, 0)?.to_vec2::<f32>()?,
        expected_scatter
    );
    let dst = Tensor::zeros((3, 2), DType::F32, device)?;
    dst.scatter_set(&ids, &src, 0)?;
    assert_eq!(dst.to_vec2::<f32>()?, expected_scatter);

    let row_ids = Tensor::from_slice(&[1u32, 0], (2,), device)?;
    let expected_index_select =
        xs_cpu.index_select(&Tensor::from_slice(&[1u32, 0], (2,), &cpu)?, 0)?;
    assert_eq!(
        xs.index_select(&row_ids, 0)?.to_vec2::<f32>()?,
        expected_index_select.to_vec2::<f32>()?
    );

    let xs_f16 = Tensor::from_slice(
        &[
            f16::from_f32(1.0),
            f16::from_f32(10.0),
            f16::from_f32(2.0),
            f16::from_f32(20.0),
            f16::from_f32(3.0),
            f16::from_f32(30.0),
        ],
        (3, 2),
        device,
    )?;
    let src_f16 = Tensor::from_slice(
        &[
            f16::from_f32(7.0),
            f16::from_f32(70.0),
            f16::from_f32(8.0),
            f16::from_f32(80.0),
        ],
        (2, 2),
        device,
    )?;
    assert_eq!(
        xs_f16.gather(&ids, 0)?.to_vec2::<f16>()?,
        [
            [f16::from_f32(1.0), f16::from_f32(20.0)],
            [f16::from_f32(3.0), f16::from_f32(10.0)]
        ]
    );
    let base_f16 = Tensor::zeros((3, 2), DType::F16, device)?;
    assert_eq!(
        base_f16.scatter(&ids, &src_f16, 0)?.to_vec2::<f16>()?,
        [
            [f16::from_f32(7.0), f16::from_f32(80.0)],
            [f16::from_f32(0.0), f16::from_f32(70.0)],
            [f16::from_f32(8.0), f16::from_f32(0.0)]
        ]
    );
    let dst_f16 = Tensor::zeros((3, 2), DType::F16, device)?;
    dst_f16.scatter_set(&ids, &src_f16, 0)?;
    assert_eq!(
        dst_f16.to_vec2::<f16>()?,
        [
            [f16::from_f32(7.0), f16::from_f32(80.0)],
            [f16::from_f32(0.0), f16::from_f32(70.0)],
            [f16::from_f32(8.0), f16::from_f32(0.0)]
        ]
    );

    // Ensure non-contiguous copy paths remain correct across core dtypes.
    let xs_t = xs.t()?;
    assert_eq!(
        xs_t.contiguous()?.to_vec2::<f32>()?,
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    );

    let xs_u32 = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (3, 2), device)?;
    let xs_u32_t = xs_u32.t()?;
    assert_eq!(
        xs_u32_t.contiguous()?.to_vec2::<u32>()?,
        [[1, 3, 5], [2, 4, 6]]
    );

    let xs_f16_t = xs_f16.t()?;
    assert_eq!(
        xs_f16_t.contiguous()?.to_vec2::<f16>()?,
        [
            [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            [
                f16::from_f32(10.0),
                f16::from_f32(20.0),
                f16::from_f32(30.0)
            ]
        ]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_gather_scatter_index_non_last_dim_native_only(device: &Device) -> Result<()> {
    smoke_f32_gather_scatter_non_last_dim(device)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    let rhs = Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), device)?;
    assert_eq!(
        lhs.matmul(&rhs)?.to_vec2::<f32>()?,
        [[58.0, 64.0], [139.0, 154.0]]
    );

    let lhs_b = Tensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.0,
        ],
        (2, 2, 3),
        device,
    )?;
    let rhs_b = Tensor::from_slice(
        &[
            7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ],
        (2, 3, 2),
        device,
    )?;
    assert_eq!(
        lhs_b.matmul(&rhs_b)?.to_vec3::<f32>()?,
        [[[58.0, 64.0], [139.0, 154.0]], [[7.0, 10.0], [40.0, 52.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_conv1d(device: &Device) -> Result<()> {
    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), device)?;
    let kernel = Tensor::from_slice(&[1.0f32, 0.0, 1.0], (1, 1, 3), device)?;
    assert_eq!(
        input.conv1d(&kernel, 0, 1, 1, 1)?.to_vec3::<f32>()?,
        [[[4.0, 6.0]]]
    );

    assert_eq!(
        input.conv1d(&kernel, 1, 1, 1, 1)?.to_vec3::<f32>()?,
        [[[2.0, 4.0, 6.0, 3.0]]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_conv2d(device: &Device) -> Result<()> {
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        (1, 1, 3, 3),
        device,
    )?;
    let kernel = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, 2), device)?;
    assert_eq!(
        input
            .conv2d(&kernel, 0, 1, 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [6.0, 8.0, 12.0, 14.0]
    );

    let padded = input.conv2d(&kernel, 1, 1, 1, 1)?;
    assert_eq!(padded.dims(), &[1, 1, 4, 4]);
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_conv_transpose(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;

    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), device)?;
    let kernel = Tensor::from_slice(&[1.0f32, 0.0, 1.0], (1, 1, 3), device)?;
    assert_eq!(
        input
            .conv_transpose1d(&kernel, 1, 0, 1, 1, 1)?
            .to_vec3::<f32>()?,
        [[[2.0, 4.0, 2.0]]]
    );

    let stride_kernel = Tensor::from_slice(&[1.0f32, 1.0], (1, 1, 2), device)?;
    assert_eq!(
        input
            .conv_transpose1d(&stride_kernel, 0, 0, 2, 1, 1)?
            .to_vec3::<f32>()?,
        [[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]]
    );

    let image = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), device)?;
    let kernel_2d = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (1, 1, 2, 2), device)?;
    assert_eq!(
        image
            .conv_transpose2d(&kernel_2d, 0, 0, 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]
    );
    assert_eq!(
        image
            .conv_transpose2d(&kernel_2d, 0, 0, 2, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
    );

    let grouped_input_vals = vec![0.5f32, -1.0, 0.25, 1.5, 0.0, -0.5];
    let grouped_kernel_vals = vec![1.0f32, 0.5, -1.0, 0.25];
    let grouped_input_gpu = Tensor::from_vec(grouped_input_vals.clone(), (1, 2, 3), device)?;
    let grouped_kernel_gpu = Tensor::from_vec(grouped_kernel_vals.clone(), (2, 1, 2), device)?;
    let grouped_input_cpu = Tensor::from_vec(grouped_input_vals, (1, 2, 3), &cpu)?;
    let grouped_kernel_cpu = Tensor::from_vec(grouped_kernel_vals, (2, 1, 2), &cpu)?;
    let grouped_gpu = grouped_input_gpu.conv_transpose1d(&grouped_kernel_gpu, 0, 0, 1, 1, 2)?;
    let grouped_cpu = grouped_input_cpu.conv_transpose1d(&grouped_kernel_cpu, 0, 0, 1, 1, 2)?;
    assert_close_vec(
        &grouped_gpu.flatten_all()?.to_vec1::<f32>()?,
        &grouped_cpu.flatten_all()?.to_vec1::<f32>()?,
        1e-5,
        "conv_transpose1d groups parity",
    );

    let dilated_1d_vals = vec![1.0f32, -2.0, 0.5];
    let dilated_1d_kernel_vals = vec![0.5f32, -1.0, 1.5];
    let dilated_1d_gpu = Tensor::from_vec(dilated_1d_vals.clone(), (1, 1, 3), device)?;
    let dilated_1d_kernel_gpu =
        Tensor::from_vec(dilated_1d_kernel_vals.clone(), (1, 1, 3), device)?;
    let dilated_1d_cpu = Tensor::from_vec(dilated_1d_vals, (1, 1, 3), &cpu)?;
    let dilated_1d_kernel_cpu = Tensor::from_vec(dilated_1d_kernel_vals, (1, 1, 3), &cpu)?;
    let dilated_1d_gpu = dilated_1d_gpu.conv_transpose1d(&dilated_1d_kernel_gpu, 1, 1, 2, 2, 1)?;
    let dilated_1d_cpu = dilated_1d_cpu.conv_transpose1d(&dilated_1d_kernel_cpu, 1, 1, 2, 2, 1)?;
    assert_close_vec(
        &dilated_1d_gpu.flatten_all()?.to_vec1::<f32>()?,
        &dilated_1d_cpu.flatten_all()?.to_vec1::<f32>()?,
        1e-5,
        "conv_transpose1d dilation/output_padding parity",
    );

    let dilated_2d_vals = vec![1.0f32, -2.0, 0.5, 0.25, 3.0, -1.5];
    let dilated_2d_kernel_vals = vec![0.5f32, -1.0, 1.5, 0.25];
    let dilated_2d_gpu = Tensor::from_vec(dilated_2d_vals.clone(), (1, 1, 2, 3), device)?;
    let dilated_2d_kernel_gpu =
        Tensor::from_vec(dilated_2d_kernel_vals.clone(), (1, 1, 2, 2), device)?;
    let dilated_2d_cpu = Tensor::from_vec(dilated_2d_vals, (1, 1, 2, 3), &cpu)?;
    let dilated_2d_kernel_cpu = Tensor::from_vec(dilated_2d_kernel_vals, (1, 1, 2, 2), &cpu)?;
    let dilated_2d_gpu = dilated_2d_gpu.conv_transpose2d(&dilated_2d_kernel_gpu, 1, 1, 2, 2)?;
    let dilated_2d_cpu = dilated_2d_cpu.conv_transpose2d(&dilated_2d_kernel_cpu, 1, 1, 2, 2)?;
    assert_close_vec(
        &dilated_2d_gpu.flatten_all()?.to_vec1::<f32>()?,
        &dilated_2d_cpu.flatten_all()?.to_vec1::<f32>()?,
        1e-5,
        "conv_transpose2d dilation/output_padding parity",
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_int_cmp_where(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;

    // u8 cmp: exact integer semantics, including values > 127.
    let lhs_u8 = Tensor::from_slice(&[1u8, 200, 3, 255, 0, 128], (2, 3), device)?;
    let rhs_u8 = Tensor::from_slice(&[1u8, 199, 4, 255, 1, 127], (2, 3), device)?;
    assert_eq!(lhs_u8.eq(&rhs_u8)?.to_vec2::<u8>()?, [[1, 0, 0], [1, 0, 0]]);
    assert_eq!(lhs_u8.gt(&rhs_u8)?.to_vec2::<u8>()?, [[0, 1, 0], [0, 0, 1]]);
    assert_eq!(lhs_u8.le(&rhs_u8)?.to_vec2::<u8>()?, [[1, 0, 1], [1, 1, 0]]);

    // u32 cmp: values past the f32 24-bit mantissa must stay exact.
    let lhs_u32 = Tensor::from_slice(&[16_777_217u32, 5, 4_000_000_000], (3,), device)?;
    let rhs_u32 = Tensor::from_slice(&[16_777_216u32, 5, 4_000_000_001], (3,), device)?;
    assert_eq!(lhs_u32.eq(&rhs_u32)?.to_vec1::<u8>()?, [0, 1, 0]);
    assert_eq!(lhs_u32.gt(&rhs_u32)?.to_vec1::<u8>()?, [1, 0, 0]);
    assert_eq!(lhs_u32.ge(&rhs_u32)?.to_vec1::<u8>()?, [1, 1, 0]);

    // i64 cmp: sign handling and magnitudes past 2^32.
    let lhs_i64 = Tensor::from_slice(&[-1i64, 1, 8_589_934_593, -8_589_934_593, 42], (5,), device)?;
    let rhs_i64 = Tensor::from_slice(&[1i64, -1, 8_589_934_592, -8_589_934_592, 42], (5,), device)?;
    assert_eq!(lhs_i64.lt(&rhs_i64)?.to_vec1::<u8>()?, [1, 0, 0, 1, 0]);
    assert_eq!(lhs_i64.gt(&rhs_i64)?.to_vec1::<u8>()?, [0, 1, 1, 0, 0]);
    assert_eq!(lhs_i64.eq(&rhs_i64)?.to_vec1::<u8>()?, [0, 0, 0, 0, 1]);
    assert_eq!(lhs_i64.ne(&rhs_i64)?.to_vec1::<u8>()?, [1, 1, 1, 1, 0]);

    // where_cond over integer value dtypes, compared against CPU reference.
    let cond_vals = [1u8, 0, 0, 1, 1, 0];
    let cond = Tensor::from_slice(&cond_vals, (2, 3), device)?;
    let cond_cpu = Tensor::from_slice(&cond_vals, (2, 3), &cpu)?;

    let t_u8 = [10u8, 20, 30, 40, 250, 60];
    let f_u8 = [1u8, 2, 3, 4, 5, 6];
    let got = Tensor::from_slice(&t_u8, (2, 3), device)?;
    let alt = Tensor::from_slice(&f_u8, (2, 3), device)?;
    let expected = cond_cpu.where_cond(
        &Tensor::from_slice(&t_u8, (2, 3), &cpu)?,
        &Tensor::from_slice(&f_u8, (2, 3), &cpu)?,
    )?;
    assert_eq!(
        cond.where_cond(&got, &alt)?.to_vec2::<u8>()?,
        expected.to_vec2::<u8>()?
    );

    let t_u32 = [16_777_217u32, 2, 3, 4_000_000_000, 5, 6];
    let f_u32 = [9u32, 8, 7, 6, 5, 4];
    let got = Tensor::from_slice(&t_u32, (2, 3), device)?;
    let alt = Tensor::from_slice(&f_u32, (2, 3), device)?;
    let expected = cond_cpu.where_cond(
        &Tensor::from_slice(&t_u32, (2, 3), &cpu)?,
        &Tensor::from_slice(&f_u32, (2, 3), &cpu)?,
    )?;
    assert_eq!(
        cond.where_cond(&got, &alt)?.to_vec2::<u32>()?,
        expected.to_vec2::<u32>()?
    );

    let t_i64 = [-8_589_934_593i64, 2, 3, 8_589_934_593, -5, 6];
    let f_i64 = [9i64, -8, 7, -6, 5, -4];
    let got = Tensor::from_slice(&t_i64, (2, 3), device)?;
    let alt = Tensor::from_slice(&f_i64, (2, 3), device)?;
    let expected = cond_cpu.where_cond(
        &Tensor::from_slice(&t_i64, (2, 3), &cpu)?,
        &Tensor::from_slice(&f_i64, (2, 3), &cpu)?,
    )?;
    assert_eq!(
        cond.where_cond(&got, &alt)?.to_vec2::<i64>()?,
        expected.to_vec2::<i64>()?
    );

    // Strided cond/value views must keep working for integer dtypes.
    let strided_expected = cond_cpu.t()?.where_cond(
        &Tensor::from_slice(&t_i64, (2, 3), &cpu)?.t()?,
        &Tensor::from_slice(&f_i64, (2, 3), &cpu)?.t()?,
    )?;
    let got_t = Tensor::from_slice(&t_i64, (2, 3), device)?.t()?;
    let alt_t = Tensor::from_slice(&f_i64, (2, 3), device)?.t()?;
    assert_eq!(
        cond.t()?.where_cond(&got_t, &alt_t)?.to_vec2::<i64>()?,
        strided_expected.to_vec2::<i64>()?
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_cmp_where(device: &Device) -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    let rhs = Tensor::from_slice(&[1.0f32, 0.0, 3.0, 5.0], (2, 2), device)?;
    assert_eq!(lhs.eq(&rhs)?.to_vec2::<u8>()?, [[1, 0], [1, 0]]);
    assert_eq!(lhs.gt(&rhs)?.to_vec2::<u8>()?, [[0, 1], [0, 0]]);
    assert_eq!(
        lhs.narrow(0, 1, 1)?
            .le(&rhs.narrow(0, 1, 1)?)?
            .to_vec2::<u8>()?,
        [[1, 1]]
    );

    let cond = Tensor::from_slice(&[1u8, 0, 1, 0], (2, 2), device)?;
    let on_true = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), device)?;
    let on_false = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    assert_eq!(
        cond.where_cond(&on_true, &on_false)?.to_vec2::<f32>()?,
        [[10.0, 2.0], [30.0, 4.0]]
    );
    assert_eq!(
        cond.transpose(0, 1)?
            .where_cond(&on_true.transpose(0, 1)?, &on_false.transpose(0, 1)?)?
            .to_vec2::<f32>()?,
        [[10.0, 30.0], [2.0, 4.0]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_scatter_add_and_index_add(device: &Device) -> Result<()> {
    let dst = Tensor::zeros((1, 5), DType::F32, device)?;
    let ids = Tensor::from_slice(&[1u32, 1, 3], (1, 3), device)?;
    let src = Tensor::from_slice(&[1.0f32, 2.5, 4.0], (1, 3), device)?;
    assert_eq!(
        dst.scatter_add(&ids, &src, 1)?.to_vec2::<f32>()?,
        [[0.0, 3.5, 0.0, 4.0, 0.0]]
    );

    let base = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    let row_ids = Tensor::from_slice(&[1u32, 0], (2,), device)?;
    let add_rows = Tensor::from_slice(&[10.0f32, 10.0, 10.0, 1.0, 1.0, 1.0], (2, 3), device)?;
    assert_eq!(
        base.index_add(&row_ids, &add_rows, 0)?.to_vec2::<f32>()?,
        [[2.0, 3.0, 4.0], [14.0, 15.0, 16.0]]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_upsample(device: &Device) -> Result<()> {
    let line = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), device)?;
    assert_eq!(
        line.upsample_nearest1d(5)?.to_vec3::<f32>()?,
        [[[1.0, 1.0, 2.0, 2.0, 3.0]]]
    );

    let image = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), device)?;
    assert_eq!(
        image
            .upsample_nearest2d(3, 4)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );
    assert_eq!(
        image
            .upsample_bilinear2d(3, 3, true)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_pool2d(device: &Device) -> Result<()> {
    let image = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        (1, 1, 3, 3),
        device,
    )?;
    assert_eq!(
        image.avg_pool2d((2, 2))?.flatten_all()?.to_vec1::<f32>()?,
        [3.0]
    );
    assert_eq!(
        image.max_pool2d((2, 2))?.flatten_all()?.to_vec1::<f32>()?,
        [5.0]
    );
    assert_eq!(
        image
            .avg_pool2d_with_stride((2, 2), (1, 1))?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [3.0, 4.0, 6.0, 7.0]
    );
    assert_eq!(
        image
            .max_pool2d_with_stride((2, 2), (1, 1))?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [5.0, 6.0, 8.0, 9.0]
    );
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_extended_unary_ops(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[0.25f32, 1.0, 4.0, 9.0], (2, 2), device)?;
    assert_close(
        &xs.sqrt()?.to_vec2::<f32>()?,
        &[[0.5, 1.0], [2.0, 3.0]],
        1e-6,
    );
    assert_close(
        &xs.sqr()?.to_vec2::<f32>()?,
        &[[0.0625, 1.0], [16.0, 81.0]],
        1e-6,
    );

    let logged = xs.log()?.exp()?;
    assert_close(&logged.to_vec2::<f32>()?, &[[0.25, 1.0], [4.0, 9.0]], 1e-5);

    let trig = Tensor::from_slice(
        &[
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            0.5,
        ],
        (2, 2),
        device,
    )?;
    assert_close(
        &trig.sin()?.to_vec2::<f32>()?,
        &[[0.0, 1.0], [0.0, 0.47942555]],
        1e-5,
    );
    assert_close(
        &trig.cos()?.to_vec2::<f32>()?,
        &[[1.0, 0.0], [-1.0, 0.87758255]],
        1e-5,
    );

    let transposed = xs.t()?;
    assert_close(
        &transposed.sqrt()?.to_vec2::<f32>()?,
        &[[0.5, 2.0], [1.0, 3.0]],
        1e-6,
    );

    assert_close(
        &xs.affine(2.0, -0.5)?.to_vec2::<f32>()?,
        &[[0.0, 1.5], [7.5, 17.5]],
        1e-6,
    );
    assert_close(
        &xs.clamp(0.5, 4.5)?.to_vec2::<f32>()?,
        &[[0.5, 1.0], [4.0, 4.5]],
        1e-6,
    );
    assert_close(
        &xs.powf(1.5)?.to_vec2::<f32>()?,
        &[[0.125, 1.0], [8.0, 27.0]],
        1e-5,
    );
    let elu_input = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], (2, 2), device)?;
    assert_close(
        &elu_input.elu(1.0)?.to_vec2::<f32>()?,
        &[[std::f32::consts::E.recip() - 1.0, 0.0], [1.0, 2.0]],
        1e-5,
    );

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_rank5_unary_binary_fallback(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(
        &[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0],
        (1, 1, 1, 2, 3),
        device,
    )?;
    let relu = xs.relu()?;
    assert_eq!(relu.dims(), &[1, 1, 1, 2, 3]);
    assert_eq!(
        relu.flatten_all()?.to_vec1::<f32>()?,
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    );

    let ys = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (1, 1, 1, 1, 3), device)?;
    let add = xs.broadcast_add(&ys)?;
    assert_eq!(add.dims(), &[1, 1, 1, 2, 3]);
    assert_eq!(
        add.flatten_all()?.to_vec1::<f32>()?,
        [8.0, 19.0, 30.0, 11.0, 22.0, 33.0]
    );

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn vulkan_uses_q8_1_rhs(device: &Device, dtype: GgmlDType, n: usize, k: usize) -> bool {
    const VULKAN_VENDOR_ID_NVIDIA: u32 = 0x10DE;
    const VULKAN_VENDOR_ID_AMD: u32 = 0x1002;
    const VULKAN_VENDOR_ID_INTEL: u32 = 0x8086;

    fn has_subgroup_min_16(device: &candle_core::VulkanDevice) -> bool {
        device.subgroup_size_control_supported()
            && device.subgroup_min_size() <= 16
            && device.subgroup_max_size() >= 16
    }

    fn uses_dmmv_subgroups(device: &candle_core::VulkanDevice, dtype: GgmlDType) -> bool {
        if matches!(
            dtype,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
        ) {
            device.subgroup_arithmetic_supported() && has_subgroup_min_16(device)
        } else {
            device.subgroup_arithmetic_supported()
        }
    }

    let supported = matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    );
    if !supported {
        return false;
    }
    if matches!(dtype, GgmlDType::Q3K | GgmlDType::Q6K) {
        return false;
    }
    let Ok(vk_device) = device.as_vulkan_device() else {
        return false;
    };
    let vendor_id = vk_device.vendor_id();
    let should_use = if n > 1 {
        true
    } else {
        match vendor_id {
            VULKAN_VENDOR_ID_NVIDIA => {
                if matches!(dtype, GgmlDType::Q2K) {
                    true
                } else if k <= 4096 {
                    false
                } else {
                    !matches!(dtype, GgmlDType::Q8_0)
                }
            }
            VULKAN_VENDOR_ID_AMD => {
                if k < 2048 {
                    false
                } else {
                    !matches!(dtype, GgmlDType::Q8_0)
                }
            }
            VULKAN_VENDOR_ID_INTEL => {
                if k < 2048 {
                    false
                } else {
                    !matches!(dtype, GgmlDType::Q4_0 | GgmlDType::Q5_1)
                }
            }
            _ => true,
        }
    };
    if !should_use || !vk_device.integer_dot_product_supported() {
        return false;
    }
    let stem = match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        _ => return false,
    };
    let base_name = format!("mul_mat_vec_{stem}_q8_1_f32");
    let shader_name = if !uses_dmmv_subgroups(vk_device, dtype) {
        base_name
    } else {
        let use_large = if matches!(vendor_id, VULKAN_VENDOR_ID_NVIDIA | VULKAN_VENDOR_ID_INTEL) {
            if dtype == GgmlDType::Q6K {
                n < 4096 && k >= 1024
            } else {
                n <= 8192 && k >= 1024
            }
        } else {
            false
        };
        let use_large = vendor_id != VULKAN_VENDOR_ID_INTEL && use_large;
        if use_large {
            format!("{base_name}_subgroup_no_shmem")
        } else {
            format!("{base_name}_subgroup")
        }
    };
    candle_vulkan_kernels::spirv(&shader_name).is_some()
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn vulkan_uses_q8_1_rhs_for_indexed_moe(
    device: &Device,
    dtype: GgmlDType,
    batch: usize,
    k: usize,
) -> bool {
    dtype != GgmlDType::Q8_0 && vulkan_uses_q8_1_rhs(device, dtype, batch, k)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
#[derive(Clone)]
struct ExactQ81Block {
    d: f32,
    s: f32,
    qs: [i8; 32],
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn read_f16_le(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn quantize_f32_to_exact_q8_1_blocks(xs: &[f32]) -> Vec<ExactQ81Block> {
    assert_eq!(xs.len() % 32, 0);
    let mut out = Vec::with_capacity(xs.len() / 32);
    for block in xs.chunks_exact(32) {
        let mut amax = 0f32;
        for &x in block {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        let mut sum = 0i32;
        for i in 0..16 {
            let q0 = f32::round(block[i] * id) as i8;
            let q1 = f32::round(block[i + 16] * id) as i8;
            qs[i] = q0;
            qs[i + 16] = q1;
            sum += q0 as i32 + q1 as i32;
        }
        out.push(ExactQ81Block {
            d: f16::from_f32(d).to_f32(),
            s: f16::from_f32(sum as f32 * d).to_f32(),
            qs,
        });
    }
    out
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn exact_q8_1_legacy_matmul_reference(qweights: &QTensor, lhs: &Tensor) -> Result<Tensor> {
    let cpu = Device::Cpu;
    let lhs_vals = lhs.flatten_all()?.to_vec1::<f32>()?;
    let q8 = quantize_f32_to_exact_q8_1_blocks(&lhs_vals);
    let (rows, k) = qweights.shape().dims2()?;
    let dtype = qweights.dtype();
    let data = qweights.data()?.into_owned();
    let bytes_per_row = k / dtype.block_size() * dtype.type_size();
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let row_bytes = &data[row * bytes_per_row..(row + 1) * bytes_per_row];
        let mut acc = 0f32;
        match dtype {
            GgmlDType::Q4_0 => {
                for (block_idx, block_bytes) in row_bytes.chunks_exact(18).enumerate() {
                    let d = read_f16_le(block_bytes, 0);
                    let qb = &q8[block_idx];
                    let mut qsum = 0i32;
                    for j in 0..16 {
                        let packed = block_bytes[2 + j];
                        qsum += (packed as i32 & 0x0F) * qb.qs[j] as i32;
                        qsum += (packed as i32 >> 4) * qb.qs[j + 16] as i32;
                    }
                    acc += d * (qsum as f32 * qb.d - 8.0 * qb.s);
                }
            }
            GgmlDType::Q4_1 => {
                for (block_idx, block_bytes) in row_bytes.chunks_exact(20).enumerate() {
                    let d = read_f16_le(block_bytes, 0);
                    let m = read_f16_le(block_bytes, 2);
                    let qb = &q8[block_idx];
                    let mut qsum = 0i32;
                    for j in 0..16 {
                        let packed = block_bytes[4 + j];
                        qsum += (packed as i32 & 0x0F) * qb.qs[j] as i32;
                        qsum += (packed as i32 >> 4) * qb.qs[j + 16] as i32;
                    }
                    acc += qsum as f32 * d * qb.d + m * qb.s;
                }
            }
            GgmlDType::Q5_0 => {
                for (block_idx, block_bytes) in row_bytes.chunks_exact(22).enumerate() {
                    let d = read_f16_le(block_bytes, 0);
                    let qh = u32::from_le_bytes([
                        block_bytes[2],
                        block_bytes[3],
                        block_bytes[4],
                        block_bytes[5],
                    ]);
                    let qb = &q8[block_idx];
                    let mut qsum = 0i32;
                    for j in 0..16 {
                        let packed = block_bytes[6 + j] as i32;
                        let x0 = (packed & 0x0F) | ((((qh >> j) << 4) & 0x10) as i32);
                        let x1 = (packed >> 4) | (((qh >> (j + 12)) & 0x10) as i32);
                        qsum += x0 * qb.qs[j] as i32;
                        qsum += x1 * qb.qs[j + 16] as i32;
                    }
                    acc += d * (qsum as f32 * qb.d - 16.0 * qb.s);
                }
            }
            GgmlDType::Q5_1 => {
                for (block_idx, block_bytes) in row_bytes.chunks_exact(24).enumerate() {
                    let d = read_f16_le(block_bytes, 0);
                    let m = read_f16_le(block_bytes, 2);
                    let qh = u32::from_le_bytes([
                        block_bytes[4],
                        block_bytes[5],
                        block_bytes[6],
                        block_bytes[7],
                    ]);
                    let qb = &q8[block_idx];
                    let mut qsum = 0i32;
                    for j in 0..16 {
                        let packed = block_bytes[8 + j] as i32;
                        let x0 = (packed & 0x0F) | ((((qh >> j) << 4) & 0x10) as i32);
                        let x1 = (packed >> 4) | (((qh >> (j + 12)) & 0x10) as i32);
                        qsum += x0 * qb.qs[j] as i32;
                        qsum += x1 * qb.qs[j + 16] as i32;
                    }
                    acc += qsum as f32 * d * qb.d + m * qb.s;
                }
            }
            other => panic!("exact q8_1 legacy matmul ref unsupported for {other:?}"),
        }
        out[row] = acc;
    }
    Tensor::from_vec(out, (1, rows), &cpu)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn q8_1_activation_matmul_reference(qweights: &QTensor, lhs: &Tensor) -> Result<Tensor> {
    if matches!(
        qweights.dtype(),
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1
    ) {
        return exact_q8_1_legacy_matmul_reference(qweights, lhs);
    }
    let cpu = Device::Cpu;
    let lhs_q8_1 = QTensor::quantize(lhs, GgmlDType::Q8_1)?.dequantize(&cpu)?;
    let weights = qweights.dequantize(&cpu)?;
    lhs_q8_1.matmul(&weights.t()?)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn q8_1_activation_matmul_reference_general(qweights: &QTensor, lhs: &Tensor) -> Result<Tensor> {
    let cpu = Device::Cpu;
    let lhs_q8_1 = QTensor::quantize(lhs, GgmlDType::Q8_1)?.dequantize(&cpu)?;
    let weights = qweights.dequantize(&cpu)?;
    let (n, k) = qweights.shape().dims2()?;
    let lhs_dims = lhs_q8_1.dims().to_vec();
    let lhs_rows = lhs_dims.iter().product::<usize>() / k;
    let lhs_2d = lhs_q8_1.reshape((lhs_rows, k))?;
    let out_2d = lhs_2d.matmul(&weights.t()?)?;
    let mut out_dims = lhs_dims;
    *out_dims.last_mut().unwrap() = n;
    out_2d.reshape(Shape::from(out_dims))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn dequantized_weight_matmul_reference(qweights: &QTensor, lhs: &Tensor) -> Result<Tensor> {
    let cpu = Device::Cpu;
    let weights = qweights.dequantize(&cpu)?;
    let (n, k) = qweights.shape().dims2()?;
    let lhs_dims = lhs.dims().to_vec();
    let lhs_rows = lhs_dims.iter().product::<usize>() / k;
    let lhs_2d = lhs.reshape((lhs_rows, k))?;
    let out_2d = lhs_2d.matmul(&weights.t()?)?;
    let mut out_dims = lhs_dims;
    *out_dims.last_mut().unwrap() = n;
    out_2d.reshape(Shape::from(out_dims))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn quantized_rel_tol(dtype: GgmlDType) -> f32 {
    match dtype {
        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8K => 1e-3,
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 => 2e-3,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 5e-3,
        _ => 1e-3,
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_quantized_close(
    actual: &Tensor,
    expected: &Tensor,
    dtype: GgmlDType,
    case: &str,
) -> Result<()> {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "{case} shape mismatch for {dtype:?}"
    );
    let actual_vals = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected_vals = expected.flatten_all()?.to_vec1::<f32>()?;
    let rel_tol = quantized_rel_tol(dtype);
    let mut max_rel = 0f32;
    let mut mse_diff = 0f64;
    let mut mse_ref = 0f64;
    let mut first_bad = None;
    for (idx, (actual, expected)) in actual_vals.iter().zip(expected_vals.iter()).enumerate() {
        let rel = (actual - expected).abs() / expected.abs().max(1.0);
        max_rel = max_rel.max(rel);
        let diff = (*actual as f64) - (*expected as f64);
        mse_diff += diff * diff;
        mse_ref += (*expected as f64) * (*expected as f64);
        if rel >= rel_tol && first_bad.is_none() {
            first_bad = Some((idx, *actual, *expected));
        }
    }
    let nmse = if mse_ref > 0.0 {
        mse_diff / mse_ref
    } else {
        0.0
    };
    if case != "matvec" && nmse <= 5e-4 {
        return Ok(());
    }
    if let Some((idx, actual, expected)) = first_bad {
        panic!(
            "{case} quantized matmul mismatch for {dtype:?} at linear idx {idx}: got {actual}, expected {expected}, rel_tol {rel_tol}, max_rel {max_rel}, nmse {nmse}"
        );
    }
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn expected_quantized_matmul(
    device: &Device,
    rhs_cpu: &Tensor,
    lhs_cpu: &Tensor,
    dtype: GgmlDType,
) -> Result<Tensor> {
    let q_rhs_cpu = QTensor::quantize(&rhs_cpu.t()?, dtype)?;
    if dtype == GgmlDType::Q8K {
        return dequantized_weight_matmul_reference(&q_rhs_cpu, lhs_cpu);
    }
    let lhs_m = lhs_cpu.dims()[lhs_cpu.rank() - 2];
    let lhs_k = lhs_cpu.dims()[lhs_cpu.rank() - 1];
    if device.is_vulkan() && vulkan_uses_q8_1_rhs(device, dtype, lhs_m, lhs_k) {
        if lhs_m == 1 {
            q8_1_activation_matmul_reference(&q_rhs_cpu, lhs_cpu)
        } else {
            q8_1_activation_matmul_reference_general(&q_rhs_cpu, lhs_cpu)
        }
    } else {
        QMatMul::from_qtensor(q_rhs_cpu)?.forward(lhs_cpu)
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn q8_1_activation_indexed_moe_reference(
    qweights: &QTensor,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let cpu = Device::Cpu;
    let x_q8_1 = QTensor::quantize(x, GgmlDType::Q8_1)?.dequantize(&cpu)?;
    let weights = qweights.dequantize(&cpu)?;
    let (num_experts, n, k) = weights.shape().dims3()?;
    let (batch, topk) = ids.shape().dims2()?;
    let (input_dim1, x_vals) = match x_q8_1.rank() {
        2 => {
            let (xb, xk) = x_q8_1.dims2()?;
            assert_eq!(xb, batch);
            assert_eq!(xk, k);
            let x2 = x_q8_1.to_vec2::<f32>()?;
            let x3 = x2.into_iter().map(|row| vec![row]).collect::<Vec<_>>();
            (1, x3)
        }
        3 => {
            let (xb, xtopk, xk) = x_q8_1.dims3()?;
            assert_eq!(xb, batch);
            assert_eq!(xk, k);
            (xtopk, x_q8_1.to_vec3::<f32>()?)
        }
        rank => panic!("q8_1 indexed moe reference expects rank-2/3 input, got rank {rank}"),
    };
    let weight_vals = weights.to_vec3::<f32>()?;
    let id_vals = ids.to_vec2::<u32>()?;
    let mut out = vec![0f32; batch * topk * n];
    for batch_idx in 0..batch {
        for (topk_idx, expert_id) in id_vals[batch_idx].iter().take(topk).enumerate() {
            let expert = *expert_id as usize;
            assert!(expert < num_experts);
            let input_slot = if input_dim1 == topk { topk_idx } else { 0 };
            let out_base = (batch_idx * topk + topk_idx) * n;
            for row in 0..n {
                let mut acc = 0f32;
                for col in 0..k {
                    acc += x_vals[batch_idx][input_slot][col] * weight_vals[expert][row][col];
                }
                out[out_base + row] = acc;
            }
        }
    }
    Tensor::from_vec(out, (batch, topk, n), &cpu)
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_quantized_paths(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let qmatmul_dtypes = [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];
    let indexed_moe_dtypes = [
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];

    let k = 256;
    let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
    let rhs_vals = (0..(k * 4))
        .map(|v| (v as f32 - 384.0) / 64.0)
        .collect::<Vec<_>>();

    let lhs_cpu = Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
    let rhs_cpu = Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
    let lhs = Tensor::from_slice(&lhs_vals, (1, k), device)?;
    let rhs = Tensor::from_slice(&rhs_vals, (k, 4), device)?;
    let lhs_multi_vals = (0..(3 * k))
        .map(|v| (v as f32 - 1.5 * k as f32) / 48.0)
        .collect::<Vec<_>>();
    let lhs_multi_cpu = Tensor::from_slice(&lhs_multi_vals, (3, k), &cpu)?;
    let lhs_multi = Tensor::from_slice(&lhs_multi_vals, (3, k), device)?;
    let lhs_batched_vals = (0..(2 * 3 * k))
        .map(|v| (v as f32 - 3.0 * k as f32) / 96.0)
        .collect::<Vec<_>>();
    let lhs_batched_cpu = Tensor::from_slice(&lhs_batched_vals, (2, 3, k), &cpu)?;
    let lhs_batched = Tensor::from_slice(&lhs_batched_vals, (2, 3, k), device)?;
    for dtype in qmatmul_dtypes {
        let q_rhs = QTensor::quantize(&rhs.t()?, dtype)?;
        assert_eq!(q_rhs.shape().dims(), &[4, k]);
        assert_eq!(q_rhs.dequantize(device)?.shape().dims(), &[4, k]);

        let qmm = QMatMul::from_qtensor(q_rhs)?;

        let expected_mm = expected_quantized_matmul(device, &rhs_cpu, &lhs_cpu, dtype)?;
        let actual_mm = qmm.forward(&lhs)?;
        assert_quantized_close(&actual_mm, &expected_mm, dtype, "matvec")?;

        let expected_multi = expected_quantized_matmul(device, &rhs_cpu, &lhs_multi_cpu, dtype)?;
        let actual_multi = qmm.forward(&lhs_multi)?;
        assert_quantized_close(&actual_multi, &expected_multi, dtype, "matmul-2d")?;

        let expected_batched =
            expected_quantized_matmul(device, &rhs_cpu, &lhs_batched_cpu, dtype)?;
        let actual_batched = qmm.forward(&lhs_batched)?;
        assert_quantized_close(&actual_batched, &expected_batched, dtype, "matmul-3d")?;
    }

    let moe_w_vals = (0..(2 * 3 * k))
        .map(|v| (v as f32 - 3.0 * k as f32) / 128.0)
        .collect::<Vec<_>>();
    let moe_x_vals = (0..(2 * k))
        .map(|v| (v as f32 - k as f32 / 2.0) / 16.0)
        .collect::<Vec<_>>();
    let moe_w = Tensor::from_slice(&moe_w_vals, (2, 3, k), device)?;
    let moe_x = Tensor::from_slice(&moe_x_vals, (2, k), device)?;
    let moe_ids = Tensor::from_slice(&[0u32, 1, 1, 0], (2, 2), device)?;
    let moe_w_cpu = Tensor::from_slice(&moe_w_vals, (2, 3, k), &cpu)?;
    let moe_x_cpu = Tensor::from_slice(&moe_x_vals, (2, k), &cpu)?;
    let moe_ids_cpu = Tensor::from_slice(&[0u32, 1, 1, 0], (2, 2), &cpu)?;
    let moe_x3_vals = (0..(2 * 2 * k))
        .map(|v| (v as f32 - k as f32) / 24.0)
        .collect::<Vec<_>>();
    let moe_x3 = Tensor::from_slice(&moe_x3_vals, (2, 2, k), device)?;
    let moe_x3_cpu = Tensor::from_slice(&moe_x3_vals, (2, 2, k), &cpu)?;
    for dtype in indexed_moe_dtypes {
        let q_moe = QTensor::quantize(&moe_w, dtype)?;
        let q_moe_cpu = QTensor::quantize(&moe_w_cpu, dtype)?;
        let expected_moe = if device.is_vulkan()
            && vulkan_uses_q8_1_rhs_for_indexed_moe(device, dtype, moe_ids_cpu.dims()[0], k)
        {
            q8_1_activation_indexed_moe_reference(&q_moe_cpu, &moe_x_cpu, &moe_ids_cpu)?
        } else {
            q_moe_cpu.indexed_moe_forward(&moe_x_cpu, &moe_ids_cpu)?
        };
        let actual_moe = q_moe.indexed_moe_forward(&moe_x, &moe_ids)?;
        let rel_tol = match dtype {
            GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8K => 2e-3,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => {
                8e-3
            }
            _ => 2e-3,
        };
        for (actual_topk, expected_topk) in actual_moe
            .to_vec3::<f32>()?
            .iter()
            .zip(expected_moe.to_vec3::<f32>()?.iter())
        {
            for (topk_idx, (actual_row, expected_row)) in
                actual_topk.iter().zip(expected_topk.iter()).enumerate()
            {
                for (col_idx, (actual, expected)) in
                    actual_row.iter().zip(expected_row.iter()).enumerate()
                {
                    assert!(
                        (actual - expected).abs() < rel_tol * expected.abs().max(1.0),
                        "quantized indexed_moe mismatch for {dtype:?} at topk={topk_idx}, col={col_idx}: got {actual}, expected {expected}, rel_tol {rel_tol}"
                    );
                }
            }
        }

        let expected_moe_x3 = if device.is_vulkan()
            && vulkan_uses_q8_1_rhs_for_indexed_moe(device, dtype, moe_ids_cpu.dims()[0], k)
        {
            q8_1_activation_indexed_moe_reference(&q_moe_cpu, &moe_x3_cpu, &moe_ids_cpu)?
        } else {
            q_moe_cpu.indexed_moe_forward(&moe_x3_cpu, &moe_ids_cpu)?
        };
        let actual_moe_x3 = q_moe.indexed_moe_forward(&moe_x3, &moe_ids)?;
        for (actual_topk, expected_topk) in actual_moe_x3
            .to_vec3::<f32>()?
            .iter()
            .zip(expected_moe_x3.to_vec3::<f32>()?.iter())
        {
            for (topk_idx, (actual_row, expected_row)) in
                actual_topk.iter().zip(expected_topk.iter()).enumerate()
            {
                for (col_idx, (actual, expected)) in
                    actual_row.iter().zip(expected_row.iter()).enumerate()
                {
                    assert!(
                        (actual - expected).abs() < rel_tol * expected.abs().max(1.0),
                        "quantized indexed_moe rank3 mismatch for {dtype:?} at topk={topk_idx}, col={col_idx}: got {actual}, expected {expected}, rel_tol {rel_tol}"
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_close_vec(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: mismatch at idx {idx}: got {actual}, expected {expected}, tol {tol}"
        );
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn assert_close(actual: &[Vec<f32>], expected: &[[f32; 2]; 2], tol: f32) {
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        for (col_idx, (actual, expected)) in actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= tol,
                "mismatch at ({row_idx}, {col_idx}): got {actual}, expected {expected}, tol {tol}"
            );
        }
    }
}
