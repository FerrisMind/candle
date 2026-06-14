//! Structured CUDA-parity matrix: op × dtype × rank × layout with
//! `native_required_cuda_parity` and zero CPU-fallback count.
//!
//! Reference results come from CUDA, not CPU — see `cuda_parity_backlog.md`.
//!
//! **P0 CI gate (required, not optional):**
//! ```text
//! CANDLE_REQUIRE_CUDA_TEST_DEVICE=1 \
//! CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 \
//! CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 \
//! cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_parity_matrix_tests
//! ```
//! Without `--features cuda`, this suite compiles zero tests by design.

mod support;

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
use support::{
    assert_same_error_class, assert_tensors_close, cuda_cast_supported, deterministic_f32_tensor,
    native_required_cuda_parity, P0_CAST_DTYPES, TestBackend,
};

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_unary_dtype_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let vals = vec![0.0f32, 0.25, 1.0, -0.5, 2.0, -1.25];
    let shape = (2, 3);

    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
        let got_x = Tensor::from_vec(vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let want_x = Tensor::from_vec(vals.clone(), shape, cuda)?.to_dtype(dtype)?;

        assert_tensors_close(&got_x.erf()?, &want_x.erf()?, dtype, &format!("erf {dtype:?}"))?;
        assert_tensors_close(
            &got_x.recip()?,
            &want_x.recip()?,
            dtype,
            &format!("recip {dtype:?}"),
        )?;
        assert_tensors_close(&got_x.neg()?, &want_x.neg()?, dtype, &format!("neg {dtype:?}"))?;
        assert_tensors_close(&got_x.abs()?, &want_x.abs()?, dtype, &format!("abs {dtype:?}"))?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_binary_dtype_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
    let shape = (2, 3);

    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
        let got_a = Tensor::from_vec(a_vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let got_b = Tensor::from_vec(b_vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let want_a = Tensor::from_vec(a_vals.clone(), shape, cuda)?.to_dtype(dtype)?;
        let want_b = Tensor::from_vec(b_vals.clone(), shape, cuda)?.to_dtype(dtype)?;

        assert_tensors_close(
            &got_a.add(&got_b)?,
            &want_a.add(&want_b)?,
            dtype,
            &format!("add {dtype:?}"),
        )?;
        assert_tensors_close(
            &got_a.sub(&got_b)?,
            &want_a.sub(&want_b)?,
            dtype,
            &format!("sub {dtype:?}"),
        )?;
        assert_tensors_close(
            &got_a.mul(&got_b)?,
            &want_a.mul(&want_b)?,
            dtype,
            &format!("mul {dtype:?}"),
        )?;
        assert_tensors_close(
            &got_a.div(&got_b)?,
            &want_a.div(&want_b)?,
            dtype,
            &format!("div {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_rank_layout_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let scalar = Tensor::from_slice(&[3.5f32], (1,), under_test)?;
    let scalar_cuda = Tensor::from_slice(&[3.5f32], (1,), cuda)?;
    assert_tensors_close(
        &scalar.sum_all()?,
        &scalar_cuda.sum_all()?,
        DType::F32,
        "scalar sum_all",
    )?;

    let got = deterministic_f32_tensor(&[1, 2, 3, 2, 2], 7, under_test)?
        .transpose(0, 4)?
        .to_dtype(DType::BF16)?;
    let want = deterministic_f32_tensor(&[1, 2, 3, 2, 2], 7, cuda)?
        .transpose(0, 4)?
        .to_dtype(DType::BF16)?;
    assert_tensors_close(
        &got.relu()?,
        &want.relu()?,
        DType::BF16,
        "rank5 strided bf16 relu",
    )?;

    let u8_a = Tensor::from_vec((0..24u8).collect::<Vec<_>>(), (2, 3, 4), under_test)?
        .transpose(0, 2)?;
    let u8_b = Tensor::from_vec((1..=24u8).collect::<Vec<_>>(), (2, 3, 4), under_test)?
        .transpose(0, 2)?;
    let u8_a_cuda = Tensor::from_vec((0..24u8).collect::<Vec<_>>(), (2, 3, 4), cuda)?
        .transpose(0, 2)?;
    let u8_b_cuda = Tensor::from_vec((1..=24u8).collect::<Vec<_>>(), (2, 3, 4), cuda)?
        .transpose(0, 2)?;
    assert_eq!(
        u8_a.add(&u8_b)?.flatten_all()?.to_vec1::<u8>()?,
        u8_a_cuda.add(&u8_b_cuda)?.flatten_all()?.to_vec1::<u8>()?
    );
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_conv_transpose(under_test: &Device, cuda: &Device) -> Result<()> {
    let input_vals = vec![1.0f32, 2.0, 3.0];
    let kernel_vals = vec![0.5f32, 1.0, 0.5];
    for dtype in [DType::F32, DType::F16] {
        let input_got =
            Tensor::from_vec(input_vals.clone(), (1, 1, 3), under_test)?.to_dtype(dtype)?;
        let kernel_got =
            Tensor::from_vec(kernel_vals.clone(), (1, 1, 3), under_test)?.to_dtype(dtype)?;
        let input_want =
            Tensor::from_vec(input_vals.clone(), (1, 1, 3), cuda)?.to_dtype(dtype)?;
        let kernel_want =
            Tensor::from_vec(kernel_vals.clone(), (1, 1, 3), cuda)?.to_dtype(dtype)?;
        let got = input_got.conv_transpose1d(&kernel_got, 0, 0, 1, 1, 1)?;
        let want = input_want.conv_transpose1d(&kernel_want, 0, 0, 1, 1, 1)?;
        assert_tensors_close(
            &got,
            &want,
            dtype,
            &format!("conv_transpose1d {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_rng_determinism(under_test: &Device, _cuda: &Device) -> Result<()> {
    under_test.set_seed(0xC0FFEE)?;
    let a = Tensor::rand(0.0f32, 1.0f32, (4, 8), under_test)?;
    under_test.set_seed(0xC0FFEE)?;
    let b = Tensor::rand(0.0f32, 1.0f32, (4, 8), under_test)?;
    assert_tensors_close(&a, &b, DType::F32, "rand seed reset determinism")?;

    under_test.set_seed(0xDEAD_BEEF)?;
    let first = Tensor::rand(0.0f32, 1.0f32, (4, 8), under_test)?;
    let second = Tensor::rand(0.0f32, 1.0f32, (4, 8), under_test)?;
    let first_vals = first.flatten_all()?.to_vec1::<f32>()?;
    let second_vals = second.flatten_all()?.to_vec1::<f32>()?;
    assert_ne!(
        first_vals, second_vals,
        "two consecutive rand() calls without set_seed must advance the RNG stream"
    );

    under_test.set_seed(0xBEEF)?;
    let n1 = Tensor::randn(0.0f32, 1.0f32, (3, 5), under_test)?;
    under_test.set_seed(0xBEEF)?;
    let n2 = Tensor::randn(0.0f32, 1.0f32, (3, 5), under_test)?;
    assert_tensors_close(&n1, &n2, DType::F32, "randn seed reset determinism")?;

    under_test.set_seed(0xCAFE_BABE)?;
    let n_first = Tensor::randn(0.0f32, 1.0f32, (3, 5), under_test)?;
    let n_second = Tensor::randn(0.0f32, 1.0f32, (3, 5), under_test)?;
    assert_ne!(
        n_first.flatten_all()?.to_vec1::<f32>()?,
        n_second.flatten_all()?.to_vec1::<f32>()?,
        "two consecutive randn() calls without set_seed must advance the RNG stream"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_int_binary_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let a_i64 = [-3_000_000_000i64, 6_700_417, -7, 42, -1, 100];
    let b_i64 = [3_000_000_000i64, 641, 100, -42, -1, 25];
    let got_a = Tensor::from_slice(&a_i64, (2, 3), under_test)?;
    let got_b = Tensor::from_slice(&b_i64, (2, 3), under_test)?;
    let want_a = Tensor::from_slice(&a_i64, (2, 3), cuda)?;
    let want_b = Tensor::from_slice(&b_i64, (2, 3), cuda)?;
    assert_eq!(
        (&got_a / &got_b)?.to_vec2::<i64>()?,
        (&want_a / &want_b)?.to_vec2::<i64>()?,
        "i64 div"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_argsort_dtypes(under_test: &Device, cuda: &Device) -> Result<()> {
    let shape = (2, 4);

    let f32_vals: Vec<f32> = (0..8).map(|v| (v as f32 - 3.5) * 2.0).collect();
    let got = Tensor::from_vec(f32_vals.clone(), shape, under_test)?;
    let want = Tensor::from_vec(f32_vals, shape, cuda)?;
    assert_eq!(
        got.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        want.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        "F32 argsort"
    );

    let u8_vals: Vec<u8> = (0..8).collect();
    let got = Tensor::from_vec(u8_vals.clone(), shape, under_test)?;
    let want = Tensor::from_vec(u8_vals, shape, cuda)?;
    assert_eq!(
        got.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        want.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        "U8 argsort"
    );

    let i64_vals: Vec<i64> = (0..8).map(|v| (v as i64 - 3) * 2).collect();
    let got = Tensor::from_vec(i64_vals.clone(), shape, under_test)?;
    let want = Tensor::from_vec(i64_vals, shape, cuda)?;
    assert_eq!(
        got.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        want.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
        "I64 argsort"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_conv_pool_bf16(under_test: &Device, cuda: &Device) -> Result<()> {
    let image_vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel_vals = [1.0f32, 0.0, 0.0, 1.0];
    let image = Tensor::from_slice(&image_vals, (1, 1, 3, 3), under_test)?.to_dtype(DType::BF16)?;
    let kernel =
        Tensor::from_slice(&kernel_vals, (1, 1, 2, 2), under_test)?.to_dtype(DType::BF16)?;
    let image_cuda =
        Tensor::from_slice(&image_vals, (1, 1, 3, 3), cuda)?.to_dtype(DType::BF16)?;
    let kernel_cuda =
        Tensor::from_slice(&kernel_vals, (1, 1, 2, 2), cuda)?.to_dtype(DType::BF16)?;
    assert_tensors_close(
        &image.conv2d(&kernel, 0, 1, 1, 1)?,
        &image_cuda.conv2d(&kernel_cuda, 0, 1, 1, 1)?,
        DType::BF16,
        "bf16 conv2d",
    )?;

    let pool_vals = [1.0f32, 2.0, 3.0, 4.0];
    let pool_image =
        Tensor::from_slice(&pool_vals, (1, 1, 2, 2), under_test)?.to_dtype(DType::BF16)?;
    let pool_cuda = Tensor::from_slice(&pool_vals, (1, 1, 2, 2), cuda)?.to_dtype(DType::BF16)?;
    assert_tensors_close(
        &pool_image.avg_pool2d((2, 2))?,
        &pool_cuda.avg_pool2d((2, 2))?,
        DType::BF16,
        "bf16 avg_pool2d",
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_rank0_reduce(under_test: &Device, cuda: &Device) -> Result<()> {
    let scalar = Tensor::new(7.25f32, under_test)?;
    let scalar_cuda = Tensor::new(7.25f32, cuda)?;
    assert_tensors_close(
        &scalar.sum_all()?,
        &scalar_cuda.sum_all()?,
        DType::F32,
        "rank0 sum_all",
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_reduce_multi_dim_sum(under_test: &Device, cuda: &Device) -> Result<()> {
    let shape = (2, 3, 4);
    let vals: Vec<f32> = (0..24).map(|v| (v as f32 - 11.0) * 0.5).collect();
    let got = Tensor::from_vec(vals.clone(), shape, under_test)?;
    let want = Tensor::from_vec(vals, shape, cuda)?;
    assert_tensors_close(
        &got.sum([0, 1])?,
        &want.sum([0, 1])?,
        DType::F32,
        "multi-dim sum [0,1]",
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_argsort_bf16_f16(under_test: &Device, cuda: &Device) -> Result<()> {
    let shape = (2, 4);
    for dtype in [DType::BF16, DType::F16] {
        let vals: Vec<f32> = (0..8).map(|v| (v as f32 - 3.0) * 2.0).collect();
        let got = Tensor::from_vec(vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let want = Tensor::from_vec(vals, shape, cuda)?.to_dtype(dtype)?;
        assert_eq!(
            got.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
            want.arg_sort_last_dim(true)?.to_vec2::<u32>()?,
            "{dtype:?} argsort"
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_max_pool_bf16(under_test: &Device, cuda: &Device) -> Result<()> {
    let pool_vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let pool_image =
        Tensor::from_slice(&pool_vals, (1, 1, 3, 3), under_test)?.to_dtype(DType::BF16)?;
    let pool_cuda = Tensor::from_slice(&pool_vals, (1, 1, 3, 3), cuda)?.to_dtype(DType::BF16)?;
    assert_tensors_close(
        &pool_image.max_pool2d((2, 2))?,
        &pool_cuda.max_pool2d((2, 2))?,
        DType::BF16,
        "bf16 max_pool2d",
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_matmul_bf16(under_test: &Device, cuda: &Device) -> Result<()> {
    let lhs_vals = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.0];
    let rhs_vals = [2.0f32, -1.0, 0.25, 3.0, -2.0, 1.5];
    let lhs = Tensor::from_slice(&lhs_vals, (2, 3), under_test)?.to_dtype(DType::BF16)?;
    let rhs = Tensor::from_slice(&rhs_vals, (3, 2), under_test)?.to_dtype(DType::BF16)?;
    let lhs_cuda = Tensor::from_slice(&lhs_vals, (2, 3), cuda)?.to_dtype(DType::BF16)?;
    let rhs_cuda = Tensor::from_slice(&rhs_vals, (3, 2), cuda)?.to_dtype(DType::BF16)?;
    assert_tensors_close(
        &lhs.matmul(&rhs)?,
        &lhs_cuda.matmul(&rhs_cuda)?,
        DType::BF16,
        "bf16 matmul",
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_cast_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let vals: Vec<f32> = (0..64).map(|v| v as f32 / 8.0).collect();
    for &src in P0_CAST_DTYPES {
        for &dst in P0_CAST_DTYPES {
            if !cuda_cast_supported(cuda, src, dst) {
                continue;
            }
            let got_src = Tensor::from_vec(vals.clone(), 64, under_test)?.to_dtype(src)?;
            let want_src = Tensor::from_vec(vals.clone(), 64, cuda)?.to_dtype(src)?;
            let got = got_src.to_dtype(dst);
            let want = want_src.to_dtype(dst);
            if want.is_err() {
                assert_same_error_class(&got, &want, &format!("cast {src:?}->{dst:?}"))?;
                continue;
            }
            assert_tensors_close(
                &got?.to_dtype(DType::F32)?,
                &want?.to_dtype(DType::F32)?,
                DType::F32,
                &format!("cast {src:?}->{dst:?}"),
            )?;
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_cmp_where_dtypes(under_test: &Device, cuda: &Device) -> Result<()> {
    let shape = (2, 3);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8, DType::U32] {
        let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_vals = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
        let got_a = Tensor::from_vec(a_vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let got_b = Tensor::from_vec(b_vals.clone(), shape, under_test)?.to_dtype(dtype)?;
        let want_a = Tensor::from_vec(a_vals, shape, cuda)?.to_dtype(dtype)?;
        let want_b = Tensor::from_vec(b_vals, shape, cuda)?.to_dtype(dtype)?;
        let cond = got_a.gt(&got_b)?;
        let cond_want = want_a.gt(&want_b)?;
        assert_eq!(
            cond.flatten_all()?.to_vec1::<u8>()?,
            cond_want.flatten_all()?.to_vec1::<u8>()?,
            "cmp gt {dtype:?}"
        );
        let where_got = cond.where_cond(&got_a, &got_b)?;
        let where_want = cond_want.where_cond(&want_a, &want_b)?;
        assert_tensors_close(&where_got, &where_want, dtype, &format!("where {dtype:?}"))?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_indexing_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    let src_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let ids_0 = Tensor::from_vec(vec![0u32, 2, 1], 3, under_test)?;
    let ids_0_cuda = Tensor::from_vec(vec![0u32, 2, 1], 3, cuda)?;
    let ids_1 = Tensor::from_vec(vec![2u32, 0, 1, 1], (2, 2), under_test)?;
    let ids_1_cuda = Tensor::from_vec(vec![2u32, 0, 1, 1], (2, 2), cuda)?;
    for dtype in [DType::F32, DType::BF16, DType::U8] {
        let got_src = Tensor::from_vec(src_vals.clone(), (3, 4), under_test)?.to_dtype(dtype)?;
        let want_src = Tensor::from_vec(src_vals.clone(), (3, 4), cuda)?.to_dtype(dtype)?;
        assert_tensors_close(
            &got_src.index_select(&ids_0, 0)?,
            &want_src.index_select(&ids_0_cuda, 0)?,
            dtype,
            &format!("index_select dim0 {dtype:?}"),
        )?;
        let got_gather = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            (2, 3),
            under_test,
        )?
        .to_dtype(dtype)?;
        let want_gather = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            (2, 3),
            cuda,
        )?
        .to_dtype(dtype)?;
        assert_tensors_close(
            &got_gather.gather(&ids_1, 1)?,
            &want_gather.gather(&ids_1_cuda, 1)?,
            dtype,
            &format!("gather dim1 {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_matmul_f16_f64(under_test: &Device, cuda: &Device) -> Result<()> {
    let lhs_vals = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.0];
    let rhs_vals = [2.0f32, -1.0, 0.25, 3.0, -2.0, 1.5];
    for dtype in [DType::F16, DType::F64] {
        let lhs = Tensor::from_slice(&lhs_vals, (2, 3), under_test)?.to_dtype(dtype)?;
        let rhs = Tensor::from_slice(&rhs_vals, (3, 2), under_test)?.to_dtype(dtype)?;
        let lhs_cuda = Tensor::from_slice(&lhs_vals, (2, 3), cuda)?.to_dtype(dtype)?;
        let rhs_cuda = Tensor::from_slice(&rhs_vals, (3, 2), cuda)?.to_dtype(dtype)?;
        assert_tensors_close(
            &lhs.matmul(&rhs)?,
            &lhs_cuda.matmul(&rhs_cuda)?,
            dtype,
            &format!("matmul {dtype:?}"),
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_f64_matmul_precision(under_test: &Device, cuda: &Device) -> Result<()> {
    // Values that diverge when matmul accumulates via F32 hub (1e16 + 1 rounds to 1e16 in f32).
    let lhs_vals = [1.0e16f64, 1.0];
    let rhs_vals = [1.0f64, 1.0];
    let lhs = Tensor::from_slice(&lhs_vals, (1, 2), under_test)?;
    let rhs = Tensor::from_slice(&rhs_vals, (2, 1), under_test)?;
    let lhs_cuda = Tensor::from_slice(&lhs_vals, (1, 2), cuda)?;
    let rhs_cuda = Tensor::from_slice(&rhs_vals, (2, 1), cuda)?;
    let got = lhs.matmul(&rhs)?;
    let want = lhs_cuda.matmul(&rhs_cuda)?;
    assert_tensors_close(&got, &want, DType::F64, "f64 matmul precision")?;
    let got_val = got.to_vec2::<f64>()?[0][0];
    let want_val = want.to_vec2::<f64>()?[0][0];
    if (got_val - want_val).abs() > 0.5 {
        candle_core::bail!(
            "f64 matmul precision: expected ~1e16+1, got {got_val}, want {want_val}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn parity_f64_surface(under_test: &Device, cuda: &Device) -> Result<()> {
    let shape = (2, 3);
    let vals = vec![0.5f64, -1.25, 2.0, 0.0, 3.5, -0.75];
    let got = Tensor::from_vec(vals.clone(), shape, under_test)?;
    let want = Tensor::from_vec(vals, shape, cuda)?;
    assert_tensors_close(&got.neg()?, &want.neg()?, DType::F64, "f64 neg")?;
    assert_tensors_close(&got.abs()?, &want.abs()?, DType::F64, "f64 abs")?;

    let lhs_vals = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_vals = [0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5];
    let lhs = Tensor::from_slice(&lhs_vals, (2, 3), under_test)?;
    let rhs = Tensor::from_slice(&rhs_vals, (2, 3), under_test)?;
    let lhs_cuda = Tensor::from_slice(&lhs_vals, (2, 3), cuda)?;
    let rhs_cuda = Tensor::from_slice(&rhs_vals, (2, 3), cuda)?;
    assert_tensors_close(
        &lhs.matmul(&rhs.transpose(0, 1)?)?,
        &lhs_cuda.matmul(&rhs_cuda.transpose(0, 1)?)?,
        DType::F64,
        "f64 matmul",
    )?;

    under_test.set_seed(0xF64_0EAD)?;
    let r1 = Tensor::rand(0.0f64, 1.0f64, (8,), under_test)?;
    under_test.set_seed(0xF64_0EAD)?;
    let r2 = Tensor::rand(0.0f64, 1.0f64, (8,), under_test)?;
    assert_tensors_close(&r1, &r2, DType::F64, "f64 rand seed determinism")?;
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "wgpu", feature = "vulkan")))]
fn run_parity_matrix(under_test: &Device, cuda: &Device) -> Result<()> {
    parity_unary_dtype_matrix(under_test, cuda)?;
    parity_binary_dtype_matrix(under_test, cuda)?;
    parity_cast_matrix(under_test, cuda)?;
    parity_cmp_where_dtypes(under_test, cuda)?;
    parity_indexing_matrix(under_test, cuda)?;
    parity_matmul_f16_f64(under_test, cuda)?;
    parity_f64_matmul_precision(under_test, cuda)?;
    parity_int_binary_matrix(under_test, cuda)?;
    parity_rank_layout_matrix(under_test, cuda)?;
    parity_rank0_reduce(under_test, cuda)?;
    parity_reduce_multi_dim_sum(under_test, cuda)?;
    parity_conv_transpose(under_test, cuda)?;
    parity_conv_pool_bf16(under_test, cuda)?;
    parity_max_pool_bf16(under_test, cuda)?;
    parity_matmul_bf16(under_test, cuda)?;
    parity_argsort_dtypes(under_test, cuda)?;
    parity_argsort_bf16_f16(under_test, cuda)?;
    parity_f64_surface(under_test, cuda)?;
    parity_rng_determinism(under_test, cuda)?;
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "wgpu"))]
fn gpu_parity_matrix_wgpu() -> Result<()> {
    native_required_cuda_parity(
        "gpu_parity_matrix_wgpu",
        TestBackend::Wgpu,
        run_parity_matrix,
    )
}

#[test]
#[cfg(all(feature = "cuda", feature = "vulkan"))]
fn gpu_parity_matrix_vulkan() -> Result<()> {
    native_required_cuda_parity(
        "gpu_parity_matrix_vulkan",
        TestBackend::Vulkan,
        run_parity_matrix,
    )
}
