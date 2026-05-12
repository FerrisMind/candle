mod support;

use candle_core::{DType, Device, Result, Tensor};
use serde::Deserialize;
use support::{fallback_allowed, TestBackend};

#[derive(Debug, Deserialize)]
struct FixtureRoot {
    source: String,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    name: String,
    op: String,
    dtype: String,
    shape: Vec<i64>,
    rhs_shape: Option<Vec<i64>>,
    output_shape: Vec<i64>,
    a: Vec<f32>,
    b: Option<Vec<f32>>,
    ids: Option<Vec<i32>>,
    clamp_min: Option<f32>,
    clamp_max: Option<f32>,
    expected: Option<Vec<f32>>,
    expected_u32: Option<Vec<u32>>,
}

fn fixtures() -> FixtureRoot {
    serde_json::from_str(include_str!("fixtures/ggml_f16_f32_ops.json"))
        .expect("invalid ggml fixture json")
}

fn dims_to_usize(dims: &[i64]) -> Vec<usize> {
    dims.iter().map(|&d| d as usize).collect()
}

fn dtype_of(case: &FixtureCase) -> DType {
    match case.dtype.as_str() {
        "f16" => DType::F16,
        "f32" => DType::F32,
        other => panic!("unsupported fixture dtype {other}"),
    }
}

fn assert_close(case: &FixtureCase, got: &[f32], expected: &[f32]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "len mismatch for case {} op {}",
        case.name,
        case.op
    );
    let (atol, rtol) = if case.dtype == "f16" {
        (5e-2f32, 5e-2f32)
    } else {
        (2e-4f32, 2e-4f32)
    };
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        let delta = (g - e).abs();
        assert!(
            delta <= tol,
            "case={} op={} idx={} got={} expected={} delta={} tol={}",
            case.name,
            case.op,
            i,
            g,
            e,
            delta,
            tol
        );
    }
}

fn run_case(case: &FixtureCase, device: &Device) -> Result<()> {
    let dtype = dtype_of(case);
    let shape = dims_to_usize(&case.shape);
    let out_shape = dims_to_usize(&case.output_shape);
    let a = Tensor::from_vec(case.a.clone(), shape, device)?.to_dtype(dtype)?;

    let out = match case.op.as_str() {
        "neg" => a.neg()?,
        "abs" => a.abs()?,
        "exp" => a.exp()?,
        "log" => a.log()?,
        "sin" => a.sin()?,
        "cos" => a.cos()?,
        "tanh" => a.tanh()?,
        "sqr" => a.sqr()?,
        "sqrt" => a.sqrt()?,
        "relu" => a.relu()?,
        "ceil" => a.ceil()?,
        "floor" => a.floor()?,
        "round" => a.round()?,
        "sign" => a.sign()?,
        "add" => {
            let b_shape = dims_to_usize(case.rhs_shape.as_ref().expect("missing rhs_shape"));
            let b = Tensor::from_vec(case.b.clone().expect("missing b"), b_shape, device)?
                .to_dtype(dtype)?;
            a.add(&b)?
        }
        "sub" => {
            let b_shape = dims_to_usize(case.rhs_shape.as_ref().expect("missing rhs_shape"));
            let b = Tensor::from_vec(case.b.clone().expect("missing b"), b_shape, device)?
                .to_dtype(dtype)?;
            a.sub(&b)?
        }
        "mul" => {
            let b_shape = dims_to_usize(case.rhs_shape.as_ref().expect("missing rhs_shape"));
            let b = Tensor::from_vec(case.b.clone().expect("missing b"), b_shape, device)?
                .to_dtype(dtype)?;
            a.mul(&b)?
        }
        "div" => {
            let b_shape = dims_to_usize(case.rhs_shape.as_ref().expect("missing rhs_shape"));
            let b = Tensor::from_vec(case.b.clone().expect("missing b"), b_shape, device)?
                .to_dtype(dtype)?;
            a.div(&b)?
        }
        "clamp" => {
            let shape = dims_to_usize(&case.shape);
            let min_t = Tensor::full(
                case.clamp_min.expect("missing clamp_min"),
                shape.clone(),
                device,
            )?
            .to_dtype(dtype)?;
            let max_t = Tensor::full(case.clamp_max.expect("missing clamp_max"), shape, device)?
                .to_dtype(dtype)?;
            a.maximum(&min_t)?.minimum(&max_t)?
        }
        "sum_keepdim_last" => a.sum_keepdim(1)?,
        "mean_keepdim_last" => a.mean_keepdim(1)?,
        "argmax_keepdim_last" => a.argmax_keepdim(1)?,
        "cumsum_last" => a.cumsum(1)?,
        // ggml MUL_MAT case stores RHS as transposed matrix (B^T) in Candle layout.
        // Candle expects B directly, so we transpose RHS before matmul.
        "mul_mat_ggml" => {
            let rhs_shape = dims_to_usize(case.rhs_shape.as_ref().expect("missing rhs_shape"));
            let rhs_t = Tensor::from_vec(case.b.clone().expect("missing b"), rhs_shape, device)?
                .to_dtype(dtype)?;
            let rhs = rhs_t.t()?;
            a.matmul(&rhs)?.t()?
        }
        "index_select_dim0" => {
            let ids = case.ids.as_ref().expect("missing ids");
            let ids_u32: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
            let ids_t = Tensor::from_vec(ids_u32, ids.len(), device)?;
            a.index_select(&ids_t, 0)?
        }
        other => panic!("unsupported fixture op {other}"),
    };

    assert_eq!(
        out.dims(),
        out_shape.as_slice(),
        "shape mismatch for case {} op {}",
        case.name,
        case.op
    );

    if let Some(expected_u32) = &case.expected_u32 {
        let got = out.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
        assert_eq!(
            got, *expected_u32,
            "arg mismatch for case {} op {}",
            case.name, case.op
        );
    } else {
        let expected = case.expected.as_ref().expect("missing expected");
        let got = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_close(case, &got, expected);
    }
    Ok(())
}

fn run_all_cases_on_device(device: &Device) -> Result<()> {
    let fx = fixtures();
    assert_eq!(fx.source, "ggml");
    for case in &fx.cases {
        run_case(case, device)?;
    }
    Ok(())
}

#[test]
fn ggml_fixture_file_is_valid() {
    let fx = fixtures();
    assert_eq!(fx.source, "ggml");
    assert!(!fx.cases.is_empty());
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires WGPU runtime/device"]
fn ggml_fixture_ops_wgpu() -> Result<()> {
    fallback_allowed("ggml_fixture_ops_wgpu", TestBackend::Wgpu, |device| {
        run_all_cases_on_device(device)
    })
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Vulkan runtime/device"]
fn ggml_fixture_ops_vulkan() -> Result<()> {
    fallback_allowed("ggml_fixture_ops_vulkan", TestBackend::Vulkan, |device| {
        run_all_cases_on_device(device)
    })
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA runtime/device"]
fn ggml_fixture_ops_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    run_all_cases_on_device(&dev)
}
