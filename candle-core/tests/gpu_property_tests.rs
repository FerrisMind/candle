mod support;

use candle_core::{DType, Result, Tensor};
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
use support::{assert_tensors_close, deterministic_f32_data, native_required, TestBackend};

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "heavy property suite; requires WGPU runtime/device"]
fn gpu_properties_wgpu() -> Result<()> {
    native_required("gpu_properties_wgpu", TestBackend::Wgpu, run_property_suite)
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "heavy property suite; requires Vulkan runtime/device"]
fn gpu_properties_vulkan() -> Result<()> {
    native_required(
        "gpu_properties_vulkan",
        TestBackend::Vulkan,
        run_property_suite,
    )
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn run_property_suite(device: &candle_core::Device) -> Result<()> {
    let cpu = candle_core::Device::Cpu;
    let mut runner = TestRunner::new(ProptestConfig {
        cases: 48,
        ..ProptestConfig::default()
    });

    runner
        .run(&(1usize..9, 1usize..9, 0u64..4096), |(m, n, seed)| {
            with_case_error(|| {
                let lhs_vals = deterministic_f32_data(m * n, seed);
                let rhs_vals = deterministic_f32_data(n, seed ^ 0x55AA);

                let lhs_cpu = Tensor::from_vec(lhs_vals.clone(), (m, n), &cpu)?;
                let rhs_cpu = Tensor::from_vec(rhs_vals.clone(), (1, n), &cpu)?;
                let lhs_dev = Tensor::from_vec(lhs_vals, (m, n), device)?;
                let rhs_dev = Tensor::from_vec(rhs_vals, (1, n), device)?;

                let expected = lhs_cpu.broadcast_add(&rhs_cpu)?;
                let actual = lhs_dev.broadcast_add(&rhs_dev)?;
                assert_tensors_close(&actual, &expected, DType::F32, "property broadcast_add")
            })
        })
        .map_err(to_candle_error)?;

    runner
        .run(&(1usize..9, 1usize..9, 0u64..4096), |(m, n, seed)| {
            with_case_error(|| {
                let vals = deterministic_f32_data(m * n, seed);
                let xs_cpu = Tensor::from_vec(vals.clone(), (m, n), &cpu)?;
                let xs_dev = Tensor::from_vec(vals, (m, n), device)?;

                let expected = xs_cpu.t()?.contiguous()?;
                let actual = xs_dev.t()?.contiguous()?;
                assert_tensors_close(
                    &actual,
                    &expected,
                    DType::F32,
                    "property transpose_contiguous",
                )
            })
        })
        .map_err(to_candle_error)?;

    runner
        .run(
            &(1usize..5, 1usize..5, 1usize..5, 0u64..4096),
            |(a, b, c, seed)| {
                with_case_error(|| {
                    let vals = deterministic_f32_data(a * b * c, seed);
                    let xs_cpu = Tensor::from_vec(vals.clone(), (a, b, c), &cpu)?;
                    let xs_dev = Tensor::from_vec(vals, (a, b, c), device)?;

                    let expected = xs_cpu.reshape((a * b, c))?.flatten_all()?;
                    let actual = xs_dev.reshape((a * b, c))?.flatten_all()?;
                    assert_tensors_close(&actual, &expected, DType::F32, "property reshape_flatten")
                })
            },
        )
        .map_err(to_candle_error)?;

    runner
        .run(&(1usize..9, 1usize..9, 0u64..4096), |(m, n, seed)| {
            with_case_error(|| {
                let vals = deterministic_f32_data(m * n, seed);
                let xs_cpu = Tensor::from_vec(vals.clone(), (m, n), &cpu)?;
                let xs_dev = Tensor::from_vec(vals, (m, n), device)?;

                let expected_sum = xs_cpu.sum_keepdim(1)?;
                let actual_sum = xs_dev.sum_keepdim(1)?;
                assert_tensors_close(
                    &actual_sum,
                    &expected_sum,
                    DType::F32,
                    "property sum_keepdim",
                )?;

                let expected_cumsum = xs_cpu.cumsum(1)?;
                let actual_cumsum = xs_dev.cumsum(1)?;
                assert_tensors_close(
                    &actual_cumsum,
                    &expected_cumsum,
                    DType::F32,
                    "property cumsum",
                )?;

                let expected_argmax = xs_cpu.argmax_keepdim(1)?;
                let actual_argmax = xs_dev.argmax_keepdim(1)?;
                let actual_argmax = actual_argmax.flatten_all()?.to_vec1::<u32>()?;
                let expected_argmax = expected_argmax.flatten_all()?.to_vec1::<u32>()?;
                assert_eq!(actual_argmax, expected_argmax, "property argmax_keepdim");
                Ok(())
            })
        })
        .map_err(to_candle_error)?;

    runner
        .run(&(2usize..9, 2usize..9, 0u64..4096), |(m, n, seed)| {
            with_case_error(|| {
                let vals = deterministic_f32_data(m * n, seed);
                let xs_cpu = Tensor::from_vec(vals.clone(), (m, n), &cpu)?;
                let xs_dev = Tensor::from_vec(vals, (m, n), device)?;

                let ids_vals = vec![0u32, (n - 1) as u32];
                let ids_cpu = Tensor::from_vec(ids_vals.clone(), (2,), &cpu)?;
                let ids_dev = Tensor::from_vec(ids_vals.clone(), (2,), device)?;
                let gather_ids = Tensor::from_vec(
                    (0..(m * 2))
                        .map(|idx| if idx % 2 == 0 { 0u32 } else { (n - 1) as u32 })
                        .collect::<Vec<_>>(),
                    (m, 2),
                    &cpu,
                )?;
                let gather_ids_dev =
                    Tensor::from_vec(gather_ids.flatten_all()?.to_vec1::<u32>()?, (m, 2), device)?;

                let expected_index = xs_cpu.index_select(&ids_cpu, 1)?;
                let actual_index = xs_dev.index_select(&ids_dev, 1)?;
                assert_tensors_close(
                    &actual_index,
                    &expected_index,
                    DType::F32,
                    "property index_select",
                )?;

                let expected_gather = xs_cpu.gather(&gather_ids, 1)?;
                let actual_gather = xs_dev.gather(&gather_ids_dev, 1)?;
                assert_tensors_close(
                    &actual_gather,
                    &expected_gather,
                    DType::F32,
                    "property gather",
                )?;

                let src_vals = deterministic_f32_data(m * 2, seed ^ 0xA5A5);
                let src_cpu = Tensor::from_vec(src_vals.clone(), (m, 2), &cpu)?;
                let src_dev = Tensor::from_vec(src_vals, (m, 2), device)?;
                let dst_cpu = Tensor::zeros((m, n), DType::F32, &cpu)?;
                let dst_dev = Tensor::zeros((m, n), DType::F32, device)?;
                dst_cpu.scatter_set(&gather_ids, &src_cpu, 1)?;
                dst_dev.scatter_set(&gather_ids_dev, &src_dev, 1)?;
                assert_tensors_close(&dst_dev, &dst_cpu, DType::F32, "property scatter_set")
            })
        })
        .map_err(to_candle_error)?;

    runner
        .run(
            &(1usize..9, 1usize..9, 1usize..9, 0u64..4096),
            |(m, k, n, seed)| {
                with_case_error(|| {
                    let lhs_vals = deterministic_f32_data(m * k, seed);
                    let rhs_vals = deterministic_f32_data(k * n, seed ^ 0xDEAD);

                    let lhs_cpu = Tensor::from_vec(lhs_vals.clone(), (m, k), &cpu)?;
                    let rhs_cpu = Tensor::from_vec(rhs_vals.clone(), (k, n), &cpu)?;
                    let lhs_dev = Tensor::from_vec(lhs_vals, (m, k), device)?;
                    let rhs_dev = Tensor::from_vec(rhs_vals, (k, n), device)?;

                    let expected = lhs_cpu.matmul(&rhs_cpu)?;
                    let actual = lhs_dev.matmul(&rhs_dev)?;
                    assert_tensors_close(&actual, &expected, DType::F32, "property matmul")
                })
            },
        )
        .map_err(to_candle_error)?;

    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn with_case_error<F>(check: F) -> std::result::Result<(), TestCaseError>
where
    F: FnOnce() -> Result<()>,
{
    check().map_err(|err| TestCaseError::fail(format!("{err:?}")))
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn to_candle_error<T: std::fmt::Debug>(
    err: proptest::test_runner::TestError<T>,
) -> candle_core::Error {
    candle_core::Error::Msg(format!("property suite failed: {err}"))
}
