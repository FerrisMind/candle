#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
use half::f16;

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
#[ignore = "requires a usable wgpu adapter and driver"]
#[cfg(feature = "wgpu")]
fn backend_smoke_wgpu_f32_upload_unary_binary_roundtrip() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    assert!(device.is_wgpu());

    smoke_f32_upload_unary_binary_roundtrip(&device)?;
    smoke_unsupported_matmul_is_explicit(&device, "wgpu backend op matmul not implemented")?;
    Ok(())
}

#[test]
#[ignore = "requires a usable Vulkan compute device and driver"]
#[cfg(feature = "vulkan")]
fn backend_smoke_vulkan_f32_upload_unary_binary_roundtrip() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    assert!(device.is_vulkan());

    smoke_f32_upload_unary_binary_roundtrip(&device)?;
    smoke_i32_to_f32_dtype_conversion(&device)?;
    smoke_unsupported_matmul_is_explicit(&device, "vulkan backend op matmul not implemented")?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_f32_upload_unary_binary_roundtrip(device: &Device) -> Result<()> {
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
    smoke_f16_elementwise_ops(device)?;
    smoke_f32_binary_broadcast_and_strided_layout(device)?;
    smoke_strided_contiguous_copy(device)?;
    smoke_f32_sum_last_dim(device)?;
    smoke_f32_extended_unary_ops(device)?;

    device.synchronize()?;
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
fn smoke_f32_sum_last_dim(device: &Device) -> Result<()> {
    let xs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    assert_eq!(xs.sum_keepdim(1)?.to_vec2::<f32>()?, [[6.0], [15.0]]);

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
    let elu_input = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], (2, 2), device)?;
    assert_close(
        &elu_input.elu(1.0)?.to_vec2::<f32>()?,
        &[[std::f32::consts::E.recip() - 1.0, 0.0], [1.0, 2.0]],
        1e-5,
    );

    Ok(())
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

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn smoke_unsupported_matmul_is_explicit(device: &Device, expected: &str) -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    let rhs = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (2, 2), device)?;
    let err = lhs
        .matmul(&rhs)
        .expect_err("matmul must be explicit unsupported");
    let msg = err.to_string();
    assert!(
        msg.contains(expected),
        "expected error containing {expected:?}, got {msg:?}"
    );
    Ok(())
}
