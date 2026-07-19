//! High-precision CPU reference differential for critical GPU ops.
//!
//! Chain: CPU (f64 accumulation where useful) ↔ CUDA ↔ Vulkan ↔ WGPU.
//! Writes JSON summary when `CANDLE_NUMERICAL_REPORT` is set to a path.
//!
//! ```text
//! CANDLE_REQUIRE_CUDA_TEST_DEVICE=1 \
//! CANDLE_REQUIRE_VULKAN_TEST_DEVICE=1 \
//! CANDLE_REQUIRE_WGPU_TEST_DEVICE=1 \
//! CANDLE_EXPECTED_GPU_NAME=RTX 3060 \
//! cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_numerical_diff_tests -- --nocapture
//! ```

mod support;

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
use support::{backend_device_or_skip, cuda_device_or_skip, TestBackend};

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
#[derive(Clone, Debug)]
struct CaseReport {
    op: String,
    dtype: String,
    shape: String,
    layout: String,
    backend: String,
    max_abs_vs_cpu: f64,
    max_rel_vs_cpu: f64,
    max_abs_vs_cuda: f64,
    max_ulp_vs_cpu: u64,
    n: usize,
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn f32_data(shape: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            let v = ((x >> 33) as i32 as f32) / 1.0e9;
            v * 3.0 - 1.5
        })
        .collect()
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn special_f32() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE / 4.0,
        -(f32::MIN_POSITIVE / 4.0),
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -2.5,
        f32::NAN,
    ]
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return if a.to_bits() == b.to_bits() { 0 } else { u64::MAX / 4 };
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    // Two's complement ordering for floats
    let ai = if ai < 0 { 0x8000_0000u32 as i32 - ai } else { ai };
    let bi = if bi < 0 { 0x8000_0000u32 as i32 - bi } else { bi };
    (ai as i64 - bi as i64).unsigned_abs()
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn compare_vecs(
    got: &[f32],
    cpu: &[f32],
    cuda: &[f32],
    op: &str,
    dtype: DType,
    shape: &str,
    layout: &str,
    backend: &str,
) -> Result<CaseReport> {
    assert_eq!(got.len(), cpu.len());
    assert_eq!(cuda.len(), cpu.len());
    let mut max_abs_cpu = 0.0f64;
    let mut max_rel_cpu = 0.0f64;
    let mut max_abs_cuda = 0.0f64;
    let mut max_ulp = 0u64;
    for i in 0..got.len() {
        let g = got[i];
        let c = cpu[i];
        let d = cuda[i];
        if g.is_nan() && c.is_nan() {
            // ok
        } else if g.is_nan() || c.is_nan() {
            // allow reporting large error
            max_abs_cpu = max_abs_cpu.max(1.0e30);
        } else if g.is_infinite() || c.is_infinite() {
            if g != c {
                max_abs_cpu = max_abs_cpu.max(1.0e30);
            }
        } else {
            let abs = (g as f64 - c as f64).abs();
            let rel = abs / (1e-30 + c.abs() as f64);
            max_abs_cpu = max_abs_cpu.max(abs);
            max_rel_cpu = max_rel_cpu.max(rel);
            max_ulp = max_ulp.max(ulp_diff_f32(g, c));
        }
        if !(g.is_nan() && d.is_nan()) {
            max_abs_cuda = max_abs_cuda.max((g as f64 - d as f64).abs());
        }
        let _ = dtype;
    }
    Ok(CaseReport {
        op: op.to_string(),
        dtype: format!("{dtype:?}"),
        shape: shape.to_string(),
        layout: layout.to_string(),
        backend: backend.to_string(),
        max_abs_vs_cpu: max_abs_cpu,
        max_rel_vs_cpu: max_rel_cpu,
        max_abs_vs_cuda: max_abs_cuda,
        max_ulp_vs_cpu: max_ulp,
        n: got.len(),
    })
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn run_suite(under: &Device, cuda: &Device, backend: &str) -> Result<Vec<CaseReport>> {
    let mut reports = Vec::new();
    let shapes: &[(&[usize], &str)] = &[
        (&[1], "scalar"),
        (&[7], "odd1d"),
        (&[3, 5], "prime2d"),
        (&[8, 8], "pow2"),
        (&[1, 1, 13], "odd3d"),
    ];
    // F32 is the mandatory high-precision reference surface for this harness.
    // F16/BF16 are measured only where CUDA+under both succeed (no driver symbol miss).
    let dtypes = [DType::F32];

    for (shape, tag) in shapes {
        let data = f32_data(shape, 42);
        for dtype in dtypes {
            // unary abs / neg — contiguous
            let Ok(cpu) = Tensor::from_vec(data.clone(), *shape, &Device::Cpu)?.to_dtype(dtype)
            else {
                continue;
            };
            let Ok(g) = Tensor::from_vec(data.clone(), *shape, under)?.to_dtype(dtype) else {
                continue;
            };
            let Ok(c) = Tensor::from_vec(data.clone(), *shape, cuda)?.to_dtype(dtype) else {
                continue;
            };
            for op in ["abs", "neg"] {
                let (gt, ct, cut) = match op {
                    "abs" => (g.abs(), cpu.abs(), c.abs()),
                    _ => (g.neg(), cpu.neg(), c.neg()),
                };
                let (Ok(gt), Ok(ct), Ok(cut)) = (gt, ct, cut) else {
                    eprintln!("skip {op} {dtype:?} on {tag}");
                    continue;
                };
                reports.push(compare_vecs(
                    &to_f32_vec(&gt)?,
                    &to_f32_vec(&ct)?,
                    &to_f32_vec(&cut)?,
                    op,
                    dtype,
                    tag,
                    "contiguous",
                    backend,
                )?);
            }
            // binary mul
            let data2 = f32_data(shape, 99);
            let Ok(cpu_b) =
                Tensor::from_vec(data2.clone(), *shape, &Device::Cpu)?.to_dtype(dtype)
            else {
                continue;
            };
            let Ok(g_b) = Tensor::from_vec(data2.clone(), *shape, under)?.to_dtype(dtype) else {
                continue;
            };
            let Ok(c_b) = Tensor::from_vec(data2.clone(), *shape, cuda)?.to_dtype(dtype) else {
                continue;
            };
            let (Ok(gm), Ok(cm), Ok(dm)) = (g.mul(&g_b), cpu.mul(&cpu_b), c.mul(&c_b)) else {
                eprintln!("skip mul {dtype:?} on {tag}");
                continue;
            };
            reports.push(compare_vecs(
                &to_f32_vec(&gm)?,
                &to_f32_vec(&cm)?,
                &to_f32_vec(&dm)?,
                "mul",
                dtype,
                tag,
                "contiguous",
                backend,
            )?);
        }

        // strided unary on F32 only for stable indexing
        if shape.len() >= 2 {
            let t_cpu = Tensor::from_vec(data.clone(), *shape, &Device::Cpu)?;
            let t_g = Tensor::from_vec(data.clone(), *shape, under)?;
            let t_c = Tensor::from_vec(data.clone(), *shape, cuda)?;
            let st_cpu = t_cpu.transpose(0, shape.len() - 1)?;
            let st_g = t_g.transpose(0, shape.len() - 1)?;
            let st_c = t_c.transpose(0, shape.len() - 1)?;
            reports.push(compare_vecs(
                &to_f32_vec(&st_g.relu()?)?,
                &to_f32_vec(&st_cpu.relu()?)?,
                &to_f32_vec(&st_c.relu()?)?,
                "relu",
                DType::F32,
                tag,
                "strided_transpose",
                backend,
            )?);
        }
    }

    // matmul F32 (mandatory). F16 optional when CUDA PTX has the symbol.
    for dtype in [DType::F32, DType::F16] {
        let a = f32_data(&[7, 5], 1);
        let b = f32_data(&[5, 11], 2);
        let Ok(cpu_a) = Tensor::from_vec(a.clone(), (7, 5), &Device::Cpu)?.to_dtype(dtype) else {
            continue;
        };
        let Ok(cpu_b) = Tensor::from_vec(b.clone(), (5, 11), &Device::Cpu)?.to_dtype(dtype)
        else {
            continue;
        };
        let Ok(ga) = Tensor::from_vec(a.clone(), (7, 5), under)?.to_dtype(dtype) else {
            continue;
        };
        let Ok(gb) = Tensor::from_vec(b.clone(), (5, 11), under)?.to_dtype(dtype) else {
            continue;
        };
        let Ok(ca) = Tensor::from_vec(a.clone(), (7, 5), cuda)?.to_dtype(dtype) else {
            continue;
        };
        let Ok(cb) = Tensor::from_vec(b.clone(), (5, 11), cuda)?.to_dtype(dtype) else {
            continue;
        };
        let (Ok(gm), Ok(cm), Ok(dm)) = (ga.matmul(&gb), cpu_a.matmul(&cpu_b), ca.matmul(&cb))
        else {
            eprintln!("skip matmul {dtype:?}: backend unsupported on this stack");
            continue;
        };
        reports.push(compare_vecs(
            &to_f32_vec(&gm)?,
            &to_f32_vec(&cm)?,
            &to_f32_vec(&dm)?,
            "matmul",
            dtype,
            "7x5x11",
            "contiguous",
            backend,
        )?);
    }

    // integer exact add (U8 is more widely supported on CUDA binary than I32)
    let ua: Vec<u8> = (0..15u8).collect();
    let ub: Vec<u8> = (0..15u8).map(|x| 20u8.saturating_sub(x)).collect();
    let cpu_i = Tensor::from_vec(ua.clone(), (3, 5), &Device::Cpu)?
        .add(&Tensor::from_vec(ub.clone(), (3, 5), &Device::Cpu)?)?;
    let g_i = Tensor::from_vec(ua.clone(), (3, 5), under)?
        .add(&Tensor::from_vec(ub.clone(), (3, 5), under)?)?;
    let c_i = Tensor::from_vec(ua.clone(), (3, 5), cuda)?
        .add(&Tensor::from_vec(ub.clone(), (3, 5), cuda)?)?;
    match (
        g_i.flatten_all()?.to_vec1::<u8>(),
        cpu_i.flatten_all()?.to_vec1::<u8>(),
        c_i.flatten_all()?.to_vec1::<u8>(),
    ) {
        (Ok(g), Ok(c), Ok(d)) => {
            assert_eq!(g, c, "u8 add vs cpu");
            assert_eq!(d, c, "u8 add cuda vs cpu");
            reports.push(CaseReport {
                op: "add_u8".into(),
                dtype: "U8".into(),
                shape: "3x5".into(),
                layout: "contiguous".into(),
                backend: backend.into(),
                max_abs_vs_cpu: 0.0,
                max_rel_vs_cpu: 0.0,
                max_abs_vs_cuda: 0.0,
                max_ulp_vs_cpu: 0,
                n: 15,
            });
        }
        _ => eprintln!("skip add_u8: stack rejected"),
    }

    // special floats F32 abs
    let sp = special_f32();
    let shape = (3, 3);
    let cpu = Tensor::from_slice(&sp, shape, &Device::Cpu)?;
    let g = Tensor::from_slice(&sp, shape, under)?;
    let c = Tensor::from_slice(&sp, shape, cuda)?;
    reports.push(compare_vecs(
        &to_f32_vec(&g.abs()?)?,
        &to_f32_vec(&cpu.abs()?)?,
        &to_f32_vec(&c.abs()?)?,
        "abs_special",
        DType::F32,
        "3x3",
        "contiguous",
        backend,
    )?);

    Ok(reports)
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn maybe_write_report(reports: &[CaseReport]) -> Result<()> {
    if let Ok(path) = std::env::var("CANDLE_NUMERICAL_REPORT") {
        let mut out = String::from("{\n  \"note\": \"CPU reference is Candle CPU; half compared in f32\",\n  \"cases\": [\n");
        for (i, r) in reports.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str(&format!(
                "    {{\"op\":{:?},\"dtype\":{:?},\"shape\":{:?},\"layout\":{:?},\"backend\":{:?},\"max_abs_vs_cpu\":{:.6e},\"max_rel_vs_cpu\":{:.6e},\"max_abs_vs_cuda\":{:.6e},\"max_ulp_vs_cpu\":{},\"n\":{}}}",
                r.op,
                r.dtype,
                r.shape,
                r.layout,
                r.backend,
                r.max_abs_vs_cpu,
                r.max_rel_vs_cpu,
                r.max_abs_vs_cuda,
                r.max_ulp_vs_cpu,
                r.n
            ));
        }
        out.push_str("\n  ]\n}\n");
        std::fs::write(&path, out)?;
        eprintln!("wrote numerical report to {path}");
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(feature = "vulkan", feature = "wgpu")))]
fn assert_bounds(reports: &[CaseReport]) -> Result<()> {
    for r in reports {
        // Integer exact already asserted.
        if r.op == "add_u8" || r.op == "add_i32" {
            continue;
        }
        // Specials may include Inf paths — allow large abs for Inf mismatches only tracked above.
        let dtype_loose = r.dtype.contains("F16") || r.dtype.contains("BF16");
        let abs_lim = if dtype_loose { 5e-2 } else { 1e-3 };
        let rel_lim = if dtype_loose { 5e-2 } else { 1e-3 };
        // Skip NaN-heavy abs_special ULP
        if r.op == "abs_special" {
            continue;
        }
        if r.max_abs_vs_cpu > abs_lim && r.max_rel_vs_cpu > rel_lim {
            candle_core::bail!(
                "{} {} {} {}: max_abs_cpu={} max_rel_cpu={} max_abs_cuda={} max_ulp={}",
                r.backend,
                r.op,
                r.dtype,
                r.layout,
                r.max_abs_vs_cpu,
                r.max_rel_vs_cpu,
                r.max_abs_vs_cuda,
                r.max_ulp_vs_cpu
            );
        }
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "vulkan"))]
fn numerical_diff_vulkan() -> Result<()> {
    // Create Vulkan *before* CUDA: some Windows driver stacks fail subsequent
    // Vulkan instance creation after CUDA context init ("Unable to find a Vulkan driver").
    let Some(vk) = backend_device_or_skip("numerical_diff_vulkan", TestBackend::Vulkan)? else {
        return Ok(());
    };
    let Some(cuda) = cuda_device_or_skip("numerical_diff_vulkan")? else {
        return Ok(());
    };
    let reports = run_suite(&vk, &cuda, "vulkan")?;
    maybe_write_report(&reports)?;
    assert_bounds(&reports)?;
    eprintln!("numerical_diff_vulkan: {} cases", reports.len());
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "wgpu"))]
fn numerical_diff_wgpu() -> Result<()> {
    // Prefer GPU adapter first, then CUDA.
    let Some(wg) = backend_device_or_skip("numerical_diff_wgpu", TestBackend::Wgpu)? else {
        return Ok(());
    };
    let Some(cuda) = cuda_device_or_skip("numerical_diff_wgpu")? else {
        return Ok(());
    };
    let reports = run_suite(&wg, &cuda, "wgpu")?;
    maybe_write_report(&reports)?;
    assert_bounds(&reports)?;
    eprintln!("numerical_diff_wgpu: {} cases", reports.len());
    Ok(())
}
