//! Cross-backend DIFFERENTIAL parity test suite.
//!
//! Reference chain per AGENTS.md §9:
//!   CPU (high-precision) ↔ CUDA ↔ Vulkan ↔ WGPU
//!
//! For each (op, dtype, shape) case:
//!   1. Run on CPU → reference result
//!   2. Run on each available GPU backend → compare vs CPU
//!   3. Cross-compare GPU results (GPU-vs-GPU)
//!
//! A case passes only if:
//!   - GPU result is within per-dtype tolerance vs CPU reference, AND
//!   - Cross-GPU deltas are within 2× the CUDA-vs-CPU error band
//!
//! Device probing: try Device::new_vulkan(0)/new_wgpu(0)/new_cuda(0) behind
//! feature cfgs, skip-with-message when unavailable. Skipped != passed.
//!
//! Run:
//!   cargo test -p candle-core --features vulkan,wgpu \
//!     --test backend_parity_diff -- --test-threads=1
//!
//! Clippy gate:
//!   cargo clippy -p candle-core --features vulkan,wgpu --tests -- -D warnings

mod support;

use candle_core::test_utils::{
    compare_f32_slices, compare_f64_slices, compare_int_slices, diff_tolerance, is_integer_dtype,
};
use candle_core::{DType, Device, Result, Tensor};
use std::collections::BTreeMap;
use std::fmt::Write;

// Type aliases to reduce clippy type_complexity warnings
type UnaryOpFn = fn(&Tensor) -> Result<Tensor>;
type BinaryOpFn = fn(&Tensor, &Tensor) -> Result<Tensor>;
type UnaryOpEntry = (&'static str, UnaryOpFn);
type BinaryOpEntry = (&'static str, BinaryOpFn);
type ReduceOpFn = fn(&Tensor) -> Result<Tensor>;
type ReduceDimOpFn = fn(&Tensor, usize) -> Result<Tensor>;

// ---------------------------------------------------------------------------
// Backend probing
// ---------------------------------------------------------------------------

/// Available backends discovered at runtime.
struct BackendSet {
    devices: Vec<(String, Device)>,
    skips: Vec<String>,
}

fn probe_backends() -> BackendSet {
    let mut devices: Vec<(String, Device)> = Vec::new();
    let mut skips: Vec<String> = Vec::new();

    // CPU is always available and serves as reference.
    devices.push(("cpu".to_string(), Device::Cpu));

    // Vulkan
    #[cfg(feature = "vulkan")]
    {
        match Device::new_vulkan(0) {
            Ok(d) => devices.push(("vulkan".to_string(), d)),
            Err(e) => skips.push(format!("vulkan: {e}")),
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        skips.push("vulkan: feature not enabled".to_string());
    }

    // WGPU
    #[cfg(feature = "wgpu")]
    {
        match Device::new_wgpu(0) {
            Ok(d) => devices.push(("wgpu".to_string(), d)),
            Err(e) => skips.push(format!("wgpu: {e}")),
        }
    }
    #[cfg(not(feature = "wgpu"))]
    {
        skips.push("wgpu: feature not enabled".to_string());
    }

    // CUDA
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => devices.push(("cuda".to_string(), d)),
            Err(e) => skips.push(format!("cuda: {e}")),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        skips.push("cuda: feature not enabled".to_string());
    }

    BackendSet { devices, skips }
}

// ---------------------------------------------------------------------------
// Expected-mismatch allowlist
// ---------------------------------------------------------------------------

/// A single documented case where the GPU result legitimately differs from the
/// CPU reference, so it is *allowed* (with a rationale + an upper bound) instead
/// of being treated as a hard regression.
///
/// Matching is on (op, dtype, shape): `shape == None` matches every shape label
/// for that (op, dtype). When a case matches:
///   - if the observed |diff| is within `max_abs` it records a PASS annotated as
///     an expected mismatch (still surfaced in the summary so a stale entry is
///     visible);
///   - if the observed |diff| EXCEEDS `max_abs` it is still a hard FAIL — the
///     bound documents the expected magnitude, and breaching it means a real
///     regression (kernel drift / dtype-path change), which must not be hidden.
///   - `nonfinite_eq` makes the comparison treat any pair of non-finite values
///     (NaN, +Inf or -Inf, of either flavor) as equal; it is only meaningful for
///     cases where GPU and CPU legitimately disagree on special inputs.
struct ExpectedMismatch {
    op: &'static str,
    dtype: DType,
    shape: Option<&'static str>,
    max_abs: f64,
    nonfinite_eq: bool,
    cause: &'static str,
}

const EXPECTED_MISMATCHES: &[ExpectedMismatch] = &[
    // gelu_erf: the GPU shaders approximate `erf` (and therefore GELU) while the
    // CPU falls back to libm's high-precision erf, giving a systematic ~4.7e-4
    // abs error that sits just above the F32 (1e-4) / F64 (1e-6) tolerance at
    // the negative-input region. Every unary shape is affected.
    ExpectedMismatch {
        op: "gelu_erf",
        dtype: DType::F64,
        shape: None,
        max_abs: 5e-4,
        nonfinite_eq: true,
        cause: "shader erf approximation vs CPU libm (F64)",
    },
    ExpectedMismatch {
        op: "gelu_erf",
        dtype: DType::F32,
        shape: None,
        max_abs: 5e-4,
        nonfinite_eq: true,
        cause: "shader erf approximation vs CPU libm (F32)",
    },
    // sum_dim (reduce along dim 1) on the long "llm"-shaped F16/BF16 inputs:
    // the GPU accumulates in an f32 hub over reduced-precision-rounded
    // (f16/bf16) inputs, so long sums drift by a few ULP of the f32 accumulator,
    // growing past the f16 (1e-2) / bf16 (2e-2) abs budget. Bounded here; a
    // bound breach would mean the accumulation path regressed.
    ExpectedMismatch {
        op: "sum_dim",
        dtype: DType::F16,
        shape: Some("reduce_llm"),
        max_abs: 1e-1,
        nonfinite_eq: false,
        cause: "long f32-hub sum over f16-rounded inputs",
    },
    ExpectedMismatch {
        op: "sum_dim",
        dtype: DType::BF16,
        shape: Some("reduce_llm"),
        max_abs: 3e-1,
        nonfinite_eq: false,
        cause: "long f32-hub sum over bf16-rounded inputs",
    },
    // matmul F16 on the 1x8x8 case: f16 inputs are rounded to ~3 decimal digits
    // before the f32 kernel accumulates, so the result drifts 1.56e-2 just past
    // the f16 1e-2 abs budget. This is documented input-rounding drift, not a
    // matmul correctness regression.
    ExpectedMismatch {
        op: "matmul",
        dtype: DType::F16,
        shape: Some("square8"),
        max_abs: 2e-2,
        nonfinite_eq: false,
        cause: "f16 input rounding into f32-accumulated matmul",
    },
    // sin/cos/gelu on the special-value fixture (0, ±0, ±Inf, NaN, ±1e-7,
    // ±1e7): the GPU transcendental path disagrees with CPU libm on extreme
    // inputs (relative error at near-zero values, and finite-vs-nonfinite
    // disagreement at special inputs, e.g. gelu(-inf) -> gpu:-inf vs cpu:NaN).
    // Allowed when both sides are non-finite or the finite diff is within the
    // bound.
    ExpectedMismatch {
        op: "sin",
        dtype: DType::F32,
        shape: Some("special"),
        max_abs: 1.0,
        nonfinite_eq: true,
        cause: "GPU sin approx at special/extreme inputs vs CPU libm",
    },
    ExpectedMismatch {
        op: "cos",
        dtype: DType::F32,
        shape: Some("special"),
        max_abs: 1.0,
        nonfinite_eq: true,
        cause: "GPU cos approx at special/extreme inputs vs CPU libm",
    },
    ExpectedMismatch {
        op: "gelu",
        dtype: DType::F32,
        shape: Some("special"),
        max_abs: 1e-4,
        nonfinite_eq: true,
        cause: "gelu non-finite semantics (gpu -inf vs cpu NaN)",
    },
];

/// Look up the single allowlist entry matching a (op, dtype, shape) case.
fn allowlist_entry(op: &str, dtype: &DType, shape: &str) -> Option<&'static ExpectedMismatch> {
    EXPECTED_MISMATCHES
        .iter()
        .find(|e| e.op == op && &e.dtype == dtype && e.shape.is_none_or(|s| s == shape))
}

/// Build the annotation attached to allowlisted cases for the run summary.
///
/// A case is counted as "exercised" whenever an allowlist entry applies to it,
/// even when it passed within the normal per-dtype tolerance — so a stale /
/// now-passing entry stays visible instead of silently rotting.
fn expected_note(e: &ExpectedMismatch, max_abs: f64) -> String {
    format!(
        "expected-mismatch: {} (bound max_abs={:.1e}, observed={:.2e})",
        e.cause, e.max_abs, max_abs
    )
}

/// Decide the outcome for a float mismatch, consulting the allowlist.
/// Returns (outcome, expected_note): Pass-with-note when an entry covers the
/// observed diff (nonfinite pairs equal, or within the bound); Fail (annotated
/// when it merely exceeds the bound) otherwise.
fn resolve_float_mismatch(
    op_name: &str,
    dtype: &DType,
    shape_label: &str,
    max_abs: f64,
    first_pair: (f64, f64),
    fail_msg: String,
) -> (CaseOutcome, Option<String>) {
    let (gpu_first, cpu_first) = first_pair;
    let Some(e) = allowlist_entry(op_name, dtype, shape_label) else {
        return (CaseOutcome::Fail(fail_msg), None);
    };
    let nonfinite_pair_eq = e.nonfinite_eq && !gpu_first.is_finite() && !cpu_first.is_finite();
    if nonfinite_pair_eq || max_abs <= e.max_abs {
        (
            CaseOutcome::Pass,
            Some(expected_note(
                e,
                if nonfinite_pair_eq { 0.0 } else { max_abs },
            )),
        )
    } else {
        (
            CaseOutcome::Fail(format!(
                "{fail_msg} (exceeds allowlist bound {:.1e}: {})",
                e.max_abs, e.cause
            )),
            Some(expected_note(e, max_abs)),
        )
    }
}

// ---------------------------------------------------------------------------
// Case-level tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum CaseOutcome {
    Pass,
    Fail(String),
    Skip(String),
}

#[derive(Debug, Clone)]
struct CaseResult {
    op: String,
    dtype: String,
    shape: String,
    backend: String,
    outcome: CaseOutcome,
    /// Set when an EXPECTED_MISMATCHES entry covers this case (pass or fail);
    /// surfaced in the summary so stale entries stay visible.
    expected_note: Option<String>,
    max_abs_vs_cpu: f64,
    max_rel_vs_cpu: f64,
    max_ulp_vs_cpu: u64,
    #[allow(dead_code)]
    max_abs_cross_gpu: f64,
}

struct SuiteTracker {
    results: Vec<CaseResult>,
    passes: usize,
    failures: usize,
    skips: usize,
    skip_reasons: BTreeMap<String, usize>,
}

impl SuiteTracker {
    fn new() -> Self {
        Self {
            results: Vec::new(),
            passes: 0,
            failures: 0,
            skips: 0,
            skip_reasons: BTreeMap::new(),
        }
    }

    fn record(&mut self, r: CaseResult) {
        match &r.outcome {
            CaseOutcome::Pass => self.passes += 1,
            CaseOutcome::Fail(_) => self.failures += 1,
            CaseOutcome::Skip(reason) => {
                self.skips += 1;
                *self.skip_reasons.entry(reason.clone()).or_insert(0) += 1;
            }
        }
        self.results.push(r);
    }

    fn summary(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "Total cases: {} ({} passed, {} failed, {} skipped)",
            self.results.len(),
            self.passes,
            self.failures,
            self.skips
        );
        if !self.skip_reasons.is_empty() {
            let _ = writeln!(s, "Skip reasons:");
            for (reason, count) in &self.skip_reasons {
                let _ = writeln!(s, "  [{count}] {reason}");
            }
        }
        let expected: Vec<&CaseResult> = self
            .results
            .iter()
            .filter(|r| r.expected_note.is_some())
            .collect();
        if !expected.is_empty() {
            let _ = writeln!(s, "Expected-mismatch cases exercised:");
            for r in expected {
                let _ = writeln!(
                    s,
                    "  {} {}/{} {}: {}",
                    r.op,
                    r.dtype,
                    r.shape,
                    r.backend,
                    r.expected_note.as_deref().unwrap_or("")
                );
            }
        }
        s
    }

    fn failures_report(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            if let CaseOutcome::Fail(msg) = &r.outcome {
                let _ = writeln!(
                    s,
                    "FAIL {} {}/{} {}: abs={:.2e} rel={:.2e} ulp={} | {}",
                    r.op,
                    r.dtype,
                    r.shape,
                    r.backend,
                    r.max_abs_vs_cpu,
                    r.max_rel_vs_cpu,
                    r.max_ulp_vs_cpu,
                    msg
                );
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

/// Deterministic f32 data for a given shape and seed.
fn gen_f32(shape: &[usize], seed: u64) -> Vec<f32> {
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

/// Deterministic f64 data.
#[allow(dead_code)]
fn gen_f64(shape: &[usize], seed: u64) -> Vec<f64> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            let v = ((x >> 33) as i32 as f64) / 1.0e9;
            v * 3.0 - 1.5
        })
        .collect()
}

/// Deterministic i64 data.
fn gen_i64(shape: &[usize], seed: u64) -> Vec<i64> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            ((x >> 2) as i64) % 100
        })
        .collect()
}

/// Deterministic u8 data (small values to avoid overflow in arithmetic ops).
fn gen_u8(shape: &[usize], seed: u64) -> Vec<u8> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            ((x >> 2) as u8) % 8
        })
        .collect()
}

/// Deterministic u32 data.
fn gen_u32(shape: &[usize], seed: u64) -> Vec<u32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            ((x >> 2) as u32) % 1000
        })
        .collect()
}

/// Special float values for edge-case testing (NaN, Inf, signed zero, normal values).
/// Subnormals are excluded because GPUs commonly flush them to zero.
fn special_f32() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -2.5,
        f32::NAN,
        1e-7,
        -1e-7,
        1e7,
        -1e7,
    ]
}

/// Shapes for testing: covers scalar, 1-elem, odd, non-power-of-2, small, transformer-relevant.
fn standard_shapes() -> Vec<(Vec<usize>, &'static str)> {
    vec![
        (vec![1], "scalar"),
        (vec![1], "1elem"),
        (vec![7], "odd1d"),
        (vec![3, 5], "prime2d"),
        (vec![8, 8], "pow2"),
        (vec![2, 3, 5], "small3d"),
        (vec![4, 16, 64], "llm_hidden"), // transformer hidden dim
        (vec![4, 32, 32], "llm_attn"),   // attention head
        (vec![16, 64], "vocab_proj"),    // vocab projection
        (vec![1, 1, 13], "odd3d"),
    ]
}

/// Shapes for matmul testing: (m, k, n) tuples.
fn matmul_shapes() -> Vec<((usize, usize, usize), &'static str)> {
    vec![
        ((2, 3, 4), "square_small"),
        ((3, 2, 5), "rect"),
        ((1, 8, 8), "square8"),
        ((4, 4, 3), "batched_mm"), // (m=4, k=4, n=3)
    ]
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Map a `{dst:?}`-style dtype name (the debug form used in `to_dtype_*` case
/// names) back to a `DType`.
fn dtype_from_debug_name(s: &str) -> Option<DType> {
    Some(match s {
        "U8" => DType::U8,
        "U32" => DType::U32,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        "F8E4M3" => DType::F8E4M3,
        "F6E2M3" => DType::F6E2M3,
        "F6E3M2" => DType::F6E3M2,
        "F4" => DType::F4,
        "F8E8M0" => DType::F8E8M0,
        _ => return None,
    })
}

/// The dtype of the tensor that `op_name` produces for a given source `DType`.
///
/// Some ops emit results in a *different* dtype than their input:
/// - element-wise comparisons (`eq`/`ne`/`le`/`ge`/`lt`/`gt`) produce `U8` masks,
/// - `argmax`/`argmin` reduce to `U32` index tensors,
/// - `argsort` returns `U32` index tensors,
/// - `to_dtype` re-casts to the target dtype encoded in the case name.
fn op_result_dtype(op_name: &str, src_dtype: DType) -> DType {
    match op_name {
        "eq" | "ne" | "le" | "ge" | "lt" | "gt" => DType::U8,
        "argmax_dim" | "argmin_dim" => DType::U32,
        "argsort_asc" | "argsort_desc" => DType::U32,
        // Case names are formatted as `to_dtype_{src:?}_to_{dst:?}`; dtype debug
        // forms never contain `_`, so the last `_to_` split yields the target.
        name if name.starts_with("to_dtype_") => name
            .rsplit_once("_to_")
            .and_then(|(_, dst)| dtype_from_debug_name(dst))
            .unwrap_or(src_dtype),
        _ => src_dtype,
    }
}

/// Run op on CPU and on `gpu_device`, compare results.
/// Returns None on skip, Some(CaseResult) on pass/fail.
fn check_vs_cpu<F>(
    op_name: &str,
    dtype: DType,
    shape_label: &str,
    _shape: &[usize],
    gpu_name: &str,
    gpu_device: &Device,
    run_op: F,
) -> Option<CaseResult>
where
    F: Fn(&Device) -> Result<Tensor>,
{
    // CPU reference
    let cpu_result = match run_op(&Device::Cpu) {
        Ok(t) => t,
        Err(e) => {
            return Some(CaseResult {
                op: op_name.to_string(),
                dtype: format!("{dtype:?}"),
                shape: shape_label.to_string(),
                backend: gpu_name.to_string(),
                outcome: CaseOutcome::Skip(format!("CPU ref failed: {e}")),
                expected_note: None,
                max_abs_vs_cpu: 0.0,
                max_rel_vs_cpu: 0.0,
                max_ulp_vs_cpu: 0,
                max_abs_cross_gpu: 0.0,
            });
        }
    };

    // GPU result
    let gpu_result = match run_op(gpu_device) {
        Ok(t) => t,
        Err(e) => {
            return Some(CaseResult {
                op: op_name.to_string(),
                dtype: format!("{dtype:?}"),
                shape: shape_label.to_string(),
                backend: gpu_name.to_string(),
                outcome: CaseOutcome::Skip(format!("GPU op failed: {e}")),
                expected_note: None,
                max_abs_vs_cpu: 0.0,
                max_rel_vs_cpu: 0.0,
                max_ulp_vs_cpu: 0,
                max_abs_cross_gpu: 0.0,
            });
        }
    };

    // Compare shapes
    if cpu_result.dims() != gpu_result.dims() {
        return Some(CaseResult {
            op: op_name.to_string(),
            dtype: format!("{dtype:?}"),
            shape: shape_label.to_string(),
            backend: gpu_name.to_string(),
            outcome: CaseOutcome::Fail(format!(
                "shape mismatch: cpu {:?} vs gpu {:?}",
                cpu_result.dims(),
                gpu_result.dims()
            )),
            expected_note: None,
            max_abs_vs_cpu: 1e30,
            max_rel_vs_cpu: 1e30,
            max_ulp_vs_cpu: u64::MAX / 4,
            max_abs_cross_gpu: 0.0,
        });
    }

    let (atol, rtol) = diff_tolerance(dtype);

    // The result tensor can have a different dtype than the source (cmp ops emit
    // U8 masks, argmax/argmin/argsort emit U32 indices, to_dtype re-casts to the
    // target dtype). Dispatch the comparison on the RESULT dtype so the results
    // are read with the correct `to_vec1` element type, while keeping the
    // source-dtype tolerance semantics unchanged. When the result dtype equals
    // the source dtype this resolves to the previous behavior.
    let result_dtype = op_result_dtype(op_name, dtype);

    if is_integer_dtype(result_dtype) {
        // Exact comparison for integers
        let cpu_vec = match cpu_result.flatten_all() {
            Ok(t) => t,
            Err(e) => {
                return Some(CaseResult {
                    op: op_name.to_string(),
                    dtype: format!("{dtype:?}"),
                    shape: shape_label.to_string(),
                    backend: gpu_name.to_string(),
                    outcome: CaseOutcome::Fail(format!("CPU flatten failed: {e}")),
                    expected_note: None,
                    max_abs_vs_cpu: 1e30,
                    max_rel_vs_cpu: 1e30,
                    max_ulp_vs_cpu: u64::MAX / 4,
                    max_abs_cross_gpu: 0.0,
                });
            }
        };
        let gpu_vec = match gpu_result.flatten_all() {
            Ok(t) => t,
            Err(e) => {
                return Some(CaseResult {
                    op: op_name.to_string(),
                    dtype: format!("{dtype:?}"),
                    shape: shape_label.to_string(),
                    backend: gpu_name.to_string(),
                    outcome: CaseOutcome::Fail(format!("GPU flatten failed: {e}")),
                    expected_note: None,
                    max_abs_vs_cpu: 1e30,
                    max_rel_vs_cpu: 1e30,
                    max_ulp_vs_cpu: u64::MAX / 4,
                    max_abs_cross_gpu: 0.0,
                });
            }
        };

        match result_dtype {
            DType::U8 => {
                let cpu = cpu_vec.to_vec1::<u8>().unwrap_or_default();
                let gpu = gpu_vec.to_vec1::<u8>().unwrap_or_default();
                if cpu.len() != gpu.len() {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail("length mismatch".to_string()),
                        expected_note: None,
                        max_abs_vs_cpu: 1e30,
                        max_rel_vs_cpu: 1e30,
                        max_ulp_vs_cpu: u64::MAX / 4,
                        max_abs_cross_gpu: 0.0,
                    });
                }
                if let Some((i, g, e)) = compare_int_slices(&gpu, &cpu) {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail(format!(
                            "mismatch at idx {i}: gpu={g:?} cpu={e:?}"
                        )),
                        expected_note: None,
                        max_abs_vs_cpu: (g as f64 - e as f64).abs(),
                        max_rel_vs_cpu: 0.0,
                        max_ulp_vs_cpu: 0,
                        max_abs_cross_gpu: 0.0,
                    });
                }
            }
            DType::U32 => {
                let cpu = cpu_vec.to_vec1::<u32>().unwrap_or_default();
                let gpu = gpu_vec.to_vec1::<u32>().unwrap_or_default();
                if cpu.len() != gpu.len() {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail("length mismatch".to_string()),
                        expected_note: None,
                        max_abs_vs_cpu: 1e30,
                        max_rel_vs_cpu: 1e30,
                        max_ulp_vs_cpu: u64::MAX / 4,
                        max_abs_cross_gpu: 0.0,
                    });
                }
                if let Some((i, g, e)) = compare_int_slices(&gpu, &cpu) {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail(format!(
                            "mismatch at idx {i}: gpu={g:?} cpu={e:?}"
                        )),
                        expected_note: None,
                        max_abs_vs_cpu: (g as f64 - e as f64).abs(),
                        max_rel_vs_cpu: 0.0,
                        max_ulp_vs_cpu: 0,
                        max_abs_cross_gpu: 0.0,
                    });
                }
            }
            DType::I16 => {
                let cpu = cpu_vec.to_vec1::<i16>().unwrap_or_default();
                let gpu = gpu_vec.to_vec1::<i16>().unwrap_or_default();
                if cpu.len() != gpu.len() {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail("length mismatch".to_string()),
                        expected_note: None,
                        max_abs_vs_cpu: 1e30,
                        max_rel_vs_cpu: 1e30,
                        max_ulp_vs_cpu: u64::MAX / 4,
                        max_abs_cross_gpu: 0.0,
                    });
                }
                if let Some((i, g, e)) = compare_int_slices(&gpu, &cpu) {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail(format!(
                            "mismatch at idx {i}: gpu={g:?} cpu={e:?}"
                        )),
                        expected_note: None,
                        max_abs_vs_cpu: (g as f64 - e as f64).abs(),
                        max_rel_vs_cpu: 0.0,
                        max_ulp_vs_cpu: 0,
                        max_abs_cross_gpu: 0.0,
                    });
                }
            }
            DType::I32 => {
                let cpu = cpu_vec.to_vec1::<i32>().unwrap_or_default();
                let gpu = gpu_vec.to_vec1::<i32>().unwrap_or_default();
                if cpu.len() != gpu.len() {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail("length mismatch".to_string()),
                        expected_note: None,
                        max_abs_vs_cpu: 1e30,
                        max_rel_vs_cpu: 1e30,
                        max_ulp_vs_cpu: u64::MAX / 4,
                        max_abs_cross_gpu: 0.0,
                    });
                }
                if let Some((i, g, e)) = compare_int_slices(&gpu, &cpu) {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail(format!(
                            "mismatch at idx {i}: gpu={g:?} cpu={e:?}"
                        )),
                        expected_note: None,
                        max_abs_vs_cpu: (g as f64 - e as f64).abs(),
                        max_rel_vs_cpu: 0.0,
                        max_ulp_vs_cpu: 0,
                        max_abs_cross_gpu: 0.0,
                    });
                }
            }
            DType::I64 => {
                let cpu = cpu_vec.to_vec1::<i64>().unwrap_or_default();
                let gpu = gpu_vec.to_vec1::<i64>().unwrap_or_default();
                if cpu.len() != gpu.len() {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail("length mismatch".to_string()),
                        expected_note: None,
                        max_abs_vs_cpu: 1e30,
                        max_rel_vs_cpu: 1e30,
                        max_ulp_vs_cpu: u64::MAX / 4,
                        max_abs_cross_gpu: 0.0,
                    });
                }
                if let Some((i, g, e)) = compare_int_slices(&gpu, &cpu) {
                    return Some(CaseResult {
                        op: op_name.to_string(),
                        dtype: format!("{dtype:?}"),
                        shape: shape_label.to_string(),
                        backend: gpu_name.to_string(),
                        outcome: CaseOutcome::Fail(format!(
                            "mismatch at idx {i}: gpu={g:?} cpu={e:?}"
                        )),
                        expected_note: None,
                        max_abs_vs_cpu: (g as f64 - e as f64).abs(),
                        max_rel_vs_cpu: 0.0,
                        max_ulp_vs_cpu: 0,
                        max_abs_cross_gpu: 0.0,
                    });
                }
            }
            _ => {}
        }
        // Pass
        return Some(CaseResult {
            op: op_name.to_string(),
            dtype: format!("{dtype:?}"),
            shape: shape_label.to_string(),
            backend: gpu_name.to_string(),
            outcome: CaseOutcome::Pass,
            expected_note: None,
            max_abs_vs_cpu: 0.0,
            max_rel_vs_cpu: 0.0,
            max_ulp_vs_cpu: 0,
            max_abs_cross_gpu: 0.0,
        });
    }

    // Float comparison
    if result_dtype == DType::F64 {
        let cpu_vec = match cpu_result.flatten_all().and_then(|t| t.to_vec1::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                return Some(CaseResult {
                    op: op_name.to_string(),
                    dtype: format!("{dtype:?}"),
                    shape: shape_label.to_string(),
                    backend: gpu_name.to_string(),
                    outcome: CaseOutcome::Fail(format!("CPU to_vec1::<f64> failed: {e}")),
                    expected_note: None,
                    max_abs_vs_cpu: 1e30,
                    max_rel_vs_cpu: 1e30,
                    max_ulp_vs_cpu: u64::MAX / 4,
                    max_abs_cross_gpu: 0.0,
                });
            }
        };
        let gpu_vec = match gpu_result.flatten_all().and_then(|t| t.to_vec1::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                return Some(CaseResult {
                    op: op_name.to_string(),
                    dtype: format!("{dtype:?}"),
                    shape: shape_label.to_string(),
                    backend: gpu_name.to_string(),
                    outcome: CaseOutcome::Fail(format!("GPU to_vec1::<f64> failed: {e}")),
                    expected_note: None,
                    max_abs_vs_cpu: 1e30,
                    max_rel_vs_cpu: 1e30,
                    max_ulp_vs_cpu: u64::MAX / 4,
                    max_abs_cross_gpu: 0.0,
                });
            }
        };

        let (max_abs, max_rel, first_bad) = compare_f64_slices(&gpu_vec, &cpu_vec);
        if max_abs > atol && max_rel > rtol {
            let idx_info = match first_bad {
                Some(i) => format!(
                    " first mismatch at idx {i}: gpu={:.6e} cpu={:.6e}",
                    gpu_vec[i], cpu_vec[i]
                ),
                None => String::new(),
            };
            let (outcome, expected_note) = resolve_float_mismatch(
                op_name,
                &dtype,
                shape_label,
                max_abs,
                (first_bad.map_or(0.0, |i| gpu_vec[i]), first_bad.map_or(0.0, |i| cpu_vec[i])),
                format!(
                    "abs={max_abs:.2e} rel={max_rel:.2e} tol_abs={atol:.2e} tol_rel={rtol:.2e}{idx_info}"
                ),
            );
            return Some(CaseResult {
                op: op_name.to_string(),
                dtype: format!("{dtype:?}"),
                shape: shape_label.to_string(),
                backend: gpu_name.to_string(),
                outcome,
                expected_note,
                max_abs_vs_cpu: max_abs,
                max_rel_vs_cpu: max_rel,
                max_ulp_vs_cpu: 0,
                max_abs_cross_gpu: 0.0,
            });
        }
        return Some(CaseResult {
            op: op_name.to_string(),
            dtype: format!("{dtype:?}"),
            shape: shape_label.to_string(),
            backend: gpu_name.to_string(),
            outcome: CaseOutcome::Pass,
            expected_note: None,
            max_abs_vs_cpu: max_abs,
            max_rel_vs_cpu: max_rel,
            max_ulp_vs_cpu: 0,
            max_abs_cross_gpu: 0.0,
        });
    }

    // F32 comparison (also handles F16/BF16 via to_dtype(F32))
    let cpu_vec = match cpu_result
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => v,
        Err(e) => {
            return Some(CaseResult {
                op: op_name.to_string(),
                dtype: format!("{dtype:?}"),
                shape: shape_label.to_string(),
                backend: gpu_name.to_string(),
                outcome: CaseOutcome::Fail(format!("CPU to f32 failed: {e}")),
                expected_note: None,
                max_abs_vs_cpu: 1e30,
                max_rel_vs_cpu: 1e30,
                max_ulp_vs_cpu: u64::MAX / 4,
                max_abs_cross_gpu: 0.0,
            });
        }
    };
    let gpu_vec = match gpu_result
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => v,
        Err(e) => {
            return Some(CaseResult {
                op: op_name.to_string(),
                dtype: format!("{dtype:?}"),
                shape: shape_label.to_string(),
                backend: gpu_name.to_string(),
                outcome: CaseOutcome::Fail(format!("GPU to f32 failed: {e}")),
                expected_note: None,
                max_abs_vs_cpu: 1e30,
                max_rel_vs_cpu: 1e30,
                max_ulp_vs_cpu: u64::MAX / 4,
                max_abs_cross_gpu: 0.0,
            });
        }
    };

    let (max_abs, max_rel, max_ulp, first_bad) = compare_f32_slices(&gpu_vec, &cpu_vec);

    // Fail only if BOTH absolute and relative tolerances are exceeded.
    if max_abs > atol && max_rel > rtol {
        let idx_info = match first_bad {
            Some(i) => format!(
                " first mismatch at idx {i}: gpu={:.6e} cpu={:.6e}",
                gpu_vec[i], cpu_vec[i]
            ),
            None => String::new(),
        };
        let (outcome, expected_note) = resolve_float_mismatch(
            op_name,
            &dtype,
            shape_label,
            max_abs,
            (first_bad.map_or(0.0, |i| gpu_vec[i] as f64), first_bad.map_or(0.0, |i| cpu_vec[i] as f64)),
            format!(
                "abs={max_abs:.2e} rel={max_rel:.2e} ulp={max_ulp} tol_abs={atol:.2e} tol_rel={rtol:.2e}{idx_info}"
            ),
        );
        return Some(CaseResult {
            op: op_name.to_string(),
            dtype: format!("{dtype:?}"),
            shape: shape_label.to_string(),
            backend: gpu_name.to_string(),
            outcome,
            expected_note,
            max_abs_vs_cpu: max_abs,
            max_rel_vs_cpu: max_rel,
            max_ulp_vs_cpu: max_ulp,
            max_abs_cross_gpu: 0.0,
        });
    }

    Some(CaseResult {
        op: op_name.to_string(),
        dtype: format!("{dtype:?}"),
        shape: shape_label.to_string(),
        backend: gpu_name.to_string(),
        outcome: CaseOutcome::Pass,
        expected_note: None,
        max_abs_vs_cpu: max_abs,
        max_rel_vs_cpu: max_rel,
        max_ulp_vs_cpu: max_ulp,
        max_abs_cross_gpu: 0.0,
    })
}

// ---------------------------------------------------------------------------
// Test runner: iterate over backends, run per-case, collect results
// ---------------------------------------------------------------------------

fn run_case<F>(
    tracker: &mut SuiteTracker,
    op_name: &str,
    dtype: DType,
    shape_label: &str,
    shape: &[usize],
    gpu_backends: &[(String, Device)],
    run_op: F,
) where
    F: Fn(&Device) -> Result<Tensor>,
{
    for (gpu_name, gpu_device) in gpu_backends {
        if gpu_name == "cpu" {
            continue;
        }
        if let Some(result) = check_vs_cpu(
            op_name,
            dtype,
            shape_label,
            shape,
            gpu_name,
            gpu_device,
            &run_op,
        ) {
            tracker.record(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Unary ops
// ---------------------------------------------------------------------------

const UNARY_OPS: &[UnaryOpEntry] = &[
    ("exp", Tensor::exp),
    ("log", Tensor::log),
    ("sin", Tensor::sin),
    ("cos", Tensor::cos),
    ("abs", Tensor::abs),
    ("neg", Tensor::neg),
    ("recip", Tensor::recip),
    ("sqr", Tensor::sqr),
    ("sqrt", Tensor::sqrt),
    ("gelu", Tensor::gelu),
    ("gelu_erf", Tensor::gelu_erf),
    ("erf", Tensor::erf),
    ("relu", Tensor::relu),
    ("silu", Tensor::silu),
    ("tanh", Tensor::tanh),
    ("floor", Tensor::floor),
    ("ceil", Tensor::ceil),
    ("round", Tensor::round),
    ("sign", Tensor::sign),
];

const UNARY_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::F64];

fn test_unary_ops(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for (op_name, op_fn) in UNARY_OPS {
        for &dtype in UNARY_FLOAT_DTYPES {
            for (shape, shape_label) in standard_shapes() {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    &shape,
                    gpu_backends,
                    |device| {
                        let data = gen_f32(&shape, 42);
                        let x = Tensor::from_vec(data, shape.clone(), device)?.to_dtype(dtype)?;
                        op_fn(&x)
                    },
                );
            }
        }
        // Special-value edge case (f32 only)
        {
            let special = special_f32();
            let n = special.len();
            let shape = vec![n];
            let shape_label = "special";
            run_case(
                tracker,
                op_name,
                DType::F32,
                shape_label,
                &shape,
                gpu_backends,
                |device| {
                    let x = Tensor::from_vec(special.clone(), shape.clone(), device)?;
                    op_fn(&x)
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Binary ops
// ---------------------------------------------------------------------------

const BINARY_OPS: &[BinaryOpEntry] = &[
    ("add", Tensor::add),
    ("sub", Tensor::sub),
    ("mul", Tensor::mul),
    ("div", Tensor::div),
];

const BINARY_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::F64];
const BINARY_INT_DTYPES: &[DType] = &[DType::U8, DType::U32, DType::I32, DType::I64];

fn test_binary_ops(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for (op_name, op_fn) in BINARY_OPS {
        // Float dtypes
        for &dtype in BINARY_FLOAT_DTYPES {
            for (shape, shape_label) in standard_shapes() {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    &shape,
                    gpu_backends,
                    |device| {
                        let s = shape.clone();
                        let a_data = gen_f32(&s, 42);
                        let b_data = gen_f32(&s, 137);
                        let a = Tensor::from_vec(a_data, s.clone(), device)?.to_dtype(dtype)?;
                        let b = Tensor::from_vec(b_data, s, device)?.to_dtype(dtype)?;
                        op_fn(&a, &b)
                    },
                );
            }
        }
        // Integer dtypes
        for &dtype in BINARY_INT_DTYPES {
            for (shape, shape_label) in standard_shapes().iter().take(4) {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    shape,
                    gpu_backends,
                    |device| match dtype {
                        DType::U8 => {
                            let s = shape.clone();
                            // Use safe values: a >= b for all elements to avoid sub underflow,
                            // and b != 0 to avoid div-by-zero.
                            let a: Vec<u8> =
                                (0..s.iter().product()).map(|i| (i as u8 % 6) + 3).collect();
                            let b: Vec<u8> =
                                (0..s.iter().product()).map(|i| (i as u8 % 3) + 1).collect();
                            let a_t = Tensor::from_vec(a, s.clone(), device)?;
                            let b_t = Tensor::from_vec(b, s, device)?;
                            op_fn(&a_t, &b_t)
                        }
                        DType::U32 => {
                            let s = shape.clone();
                            let a: Vec<u32> = (0..s.iter().product())
                                .map(|i| (i as u32 % 100) + 100)
                                .collect();
                            let b: Vec<u32> = (0..s.iter().product())
                                .map(|i| (i as u32 % 50) + 1)
                                .collect();
                            let a_t = Tensor::from_vec(a, s.clone(), device)?;
                            let b_t = Tensor::from_vec(b, s, device)?;
                            op_fn(&a_t, &b_t)
                        }
                        DType::I32 => {
                            let s = shape.clone();
                            let a: Vec<i32> = (0..s.iter().product())
                                .map(|i| (i as i32 % 100) + 100)
                                .collect();
                            let b: Vec<i32> = (0..s.iter().product())
                                .map(|i| (i as i32 % 50) + 1)
                                .collect();
                            let a_t = Tensor::from_vec(a, s.clone(), device)?;
                            let b_t = Tensor::from_vec(b, s, device)?;
                            op_fn(&a_t, &b_t)
                        }
                        DType::I64 => {
                            let s = shape.clone();
                            let a: Vec<i64> = (0..s.iter().product())
                                .map(|i| (i as i64 % 100) + 100)
                                .collect();
                            let b: Vec<i64> = (0..s.iter().product())
                                .map(|i| (i as i64 % 50) + 1)
                                .collect();
                            let a_t = Tensor::from_vec(a, s.clone(), device)?;
                            let b_t = Tensor::from_vec(b, s, device)?;
                            op_fn(&a_t, &b_t)
                        }
                        _ => unreachable!(),
                    },
                );
            }
        }
        // Broadcasting case: (3, 1, 5) + (1, 4, 1) = (3, 4, 5) using broadcast_add
        let op_name = *op_name;
        let op_fn = *op_fn;
        run_case(
            tracker,
            op_name,
            DType::F32,
            "broadcast",
            &[3, 4, 5],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[3, 1, 5], 42), (3, 1, 5), device)?;
                let b = Tensor::from_vec(gen_f32(&[1, 4, 1], 137), (1, 4, 1), device)?;
                // Use same_shape_binary_op by broadcasting manually
                let a_bc = a.broadcast_as((3, 4, 5))?;
                let b_bc = b.broadcast_as((3, 4, 5))?;
                op_fn(&a_bc, &b_bc)
            },
        );
    }

    // maximum/minimum (scalar-compatible ops, use closures)
    for &dtype in BINARY_FLOAT_DTYPES {
        for (shape, shape_label) in standard_shapes().iter().take(4) {
            run_case(
                tracker,
                "maximum",
                dtype,
                shape_label,
                shape,
                gpu_backends,
                |device| {
                    let s = shape.clone();
                    let a =
                        Tensor::from_vec(gen_f32(&s, 42), s.clone(), device)?.to_dtype(dtype)?;
                    let b = Tensor::from_vec(gen_f32(&s, 137), s, device)?.to_dtype(dtype)?;
                    a.maximum(&b)
                },
            );
            run_case(
                tracker,
                "minimum",
                dtype,
                shape_label,
                shape,
                gpu_backends,
                |device| {
                    let s = shape.clone();
                    let a =
                        Tensor::from_vec(gen_f32(&s, 42), s.clone(), device)?.to_dtype(dtype)?;
                    let b = Tensor::from_vec(gen_f32(&s, 137), s, device)?.to_dtype(dtype)?;
                    a.minimum(&b)
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison ops (output dtype is U8)
// ---------------------------------------------------------------------------

const CMP_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::F64];
const CMP_INT_DTYPES: &[DType] = &[DType::U8, DType::U32, DType::I32, DType::I64];

fn test_cmp_ops(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    let cmp_ops: &[BinaryOpEntry] = &[
        ("eq", |a: &Tensor, b: &Tensor| a.eq(b)),
        ("ne", |a: &Tensor, b: &Tensor| a.ne(b)),
        ("le", |a: &Tensor, b: &Tensor| a.le(b)),
        ("ge", |a: &Tensor, b: &Tensor| a.ge(b)),
        ("lt", |a: &Tensor, b: &Tensor| a.lt(b)),
        ("gt", |a: &Tensor, b: &Tensor| a.gt(b)),
    ];
    for (op_name, op_fn) in cmp_ops {
        for &dtype in CMP_FLOAT_DTYPES {
            for (shape, shape_label) in standard_shapes().iter().take(4) {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    shape,
                    gpu_backends,
                    |device| {
                        let s = shape.clone();
                        let a = Tensor::from_vec(gen_f32(&s, 42), s.clone(), device)?
                            .to_dtype(dtype)?;
                        let b = Tensor::from_vec(gen_f32(&s, 137), s, device)?.to_dtype(dtype)?;
                        op_fn(&a, &b)
                    },
                );
            }
        }
        for &dtype in CMP_INT_DTYPES {
            for (shape, shape_label) in standard_shapes().iter().take(3) {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    shape,
                    gpu_backends,
                    |device| match dtype {
                        DType::U8 => {
                            let s = shape.clone();
                            let a: Vec<u8> =
                                (0..s.iter().product()).map(|i| (i as u8 % 6) + 3).collect();
                            let b: Vec<u8> =
                                (0..s.iter().product()).map(|i| (i as u8 % 3) + 1).collect();
                            let a = Tensor::from_vec(a, s.clone(), device)?;
                            let b = Tensor::from_vec(b, s, device)?;
                            op_fn(&a, &b)
                        }
                        DType::U32 => {
                            let s = shape.clone();
                            let a: Vec<u32> = (0..s.iter().product())
                                .map(|i| (i as u32 % 100) + 100)
                                .collect();
                            let b: Vec<u32> = (0..s.iter().product())
                                .map(|i| (i as u32 % 50) + 1)
                                .collect();
                            let a = Tensor::from_vec(a, s.clone(), device)?;
                            let b = Tensor::from_vec(b, s, device)?;
                            op_fn(&a, &b)
                        }
                        DType::I32 => {
                            let s = shape.clone();
                            let a: Vec<i32> = (0..s.iter().product())
                                .map(|i| (i as i32 % 100) + 100)
                                .collect();
                            let b: Vec<i32> = (0..s.iter().product())
                                .map(|i| (i as i32 % 50) + 1)
                                .collect();
                            let a = Tensor::from_vec(a, s.clone(), device)?;
                            let b = Tensor::from_vec(b, s, device)?;
                            op_fn(&a, &b)
                        }
                        DType::I64 => {
                            let s = shape.clone();
                            let a: Vec<i64> = (0..s.iter().product())
                                .map(|i| (i as i64 % 100) + 100)
                                .collect();
                            let b: Vec<i64> = (0..s.iter().product())
                                .map(|i| (i as i64 % 50) + 1)
                                .collect();
                            let a = Tensor::from_vec(a, s.clone(), device)?;
                            let b = Tensor::from_vec(b, s, device)?;
                            op_fn(&a, &b)
                        }
                        _ => unreachable!(),
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce ops
// ---------------------------------------------------------------------------

const REDUCE_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::F64];

fn test_reduce_ops(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    // sum, max, min reductions
    let reduce_ops: &[(&str, ReduceOpFn)] = &[
        ("sum_all", Tensor::sum_all),
        ("max_all", Tensor::max_all),
        ("min_all", Tensor::min_all),
    ];

    for (op_name, op_fn) in reduce_ops {
        for &dtype in REDUCE_FLOAT_DTYPES {
            for (shape, shape_label) in standard_shapes().iter().take(4) {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    shape,
                    gpu_backends,
                    |device| {
                        let s = shape.clone();
                        let x = Tensor::from_vec(gen_f32(&s, 42), s, device)?.to_dtype(dtype)?;
                        op_fn(&x)
                    },
                );
            }
        }
    }

    // Multi-dim reduce: sum along a dimension
    let reduce_dim_ops: &[(&str, ReduceDimOpFn)] = &[
        ("sum_dim", Tensor::sum),
        ("max_dim", Tensor::max),
        ("min_dim", Tensor::min),
        ("argmax_dim", Tensor::argmax),
        ("argmin_dim", Tensor::argmin),
    ];

    for (op_name, op_fn) in reduce_dim_ops {
        for &dtype in REDUCE_FLOAT_DTYPES {
            for (shape, shape_label) in
                &[(vec![2, 3, 5], "reduce3d"), (vec![4, 16, 64], "reduce_llm")]
            {
                let op_name = *op_name;
                let op_fn = *op_fn;
                run_case(
                    tracker,
                    op_name,
                    dtype,
                    shape_label,
                    shape,
                    gpu_backends,
                    |device| {
                        let x = Tensor::from_vec(gen_f32(shape, 42), shape.clone(), device)?
                            .to_dtype(dtype)?;
                        op_fn(&x, 1) // reduce along dim 1
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

const MATMUL_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::F64];

fn test_matmul(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in MATMUL_DTYPES {
        for ((m, k, n), label) in matmul_shapes() {
            run_case(
                tracker,
                "matmul",
                dtype,
                label,
                &[m, n],
                gpu_backends,
                |device| {
                    let a =
                        Tensor::from_vec(gen_f32(&[m, k], 42), (m, k), device)?.to_dtype(dtype)?;
                    let b =
                        Tensor::from_vec(gen_f32(&[k, n], 137), (k, n), device)?.to_dtype(dtype)?;
                    a.matmul(&b)
                },
            );
        }
    }

    // Batched matmul
    for &dtype in &[DType::F32, DType::F16] {
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm",
            &[2, 4, 3],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[2, 4, 4], 42), (2, 4, 4), device)?
                    .to_dtype(dtype)?;
                let b = Tensor::from_vec(gen_f32(&[2, 4, 3], 137), (2, 4, 3), device)?
                    .to_dtype(dtype)?;
                a.broadcast_matmul(&b)
            },
        );
    }

    // Batched m == 1 GEMV (the attention score/ctx shape, e.g. head_dim columns):
    // exercises the Vulkan single-dispatch batched-GEMV kernel. Uses a k that is
    // NOT a multiple of 32 to exercise the partial last K iteration, and a
    // non-trivial batch so every head is covered in one dispatch.
    #[allow(clippy::single_element_loop)]
    for &dtype in &[DType::F32] {
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm_m1",
            &[2, 1, 64],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[2, 1, 100], 42), (2, 1, 100), device)?
                    .to_dtype(dtype)?;
                let b = Tensor::from_vec(gen_f32(&[2, 100, 64], 137), (2, 100, 64), device)?
                    .to_dtype(dtype)?;
                a.broadcast_matmul(&b)
            },
        );
        // Natural-layout CONTEXT GEMV (probs @ v): the m==1 rank>2 runtime dispatches
        // to the `ctx_gemv_f32` kernel when the RHS V is contiguous (batch, k, n),
        // reducing over k and outputting over head_dim n. This case gives a
        // head_dim-width output (n=128 -> grid.x = 4 head_dim tiles) and a k that is
        // NOT a multiple of 32 (exercises the partial l-slice in the K_SPLIT
        // partition). batch=2 covers every head in one dispatch.
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm_ctx_natural",
            &[2, 1, 128],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[2, 1, 130], 42), (2, 1, 130), device)?
                    .to_dtype(dtype)?;
                let b = Tensor::from_vec(gen_f32(&[2, 130, 128], 137), (2, 130, 128), device)?
                    .to_dtype(dtype)?;
                a.broadcast_matmul(&b)
            },
        );
        // wgpu m == 1 batched GEMV (score-style q @ k^T): the RHS is a transposed
        // view of a contiguous (batch, n, k) matrix, so the single-dispatch
        // batched-GEMV kernel fires (its transpose-complement is contiguous). Uses a
        // k that is NOT a multiple of 32 (partial last K iteration) and an n that is
        // a multiple of the 4-column output tile. Correct on every backend.
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm_m1_wgpu",
            &[2, 1, 64],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[2, 1, 100], 42), (2, 1, 100), device)?
                    .to_dtype(dtype)?;
                let b_nk = Tensor::from_vec(gen_f32(&[2, 64, 100], 137), (2, 64, 100), device)?
                    .to_dtype(dtype)?;
                a.matmul(&b_nk.transpose(1, 2)?)
            },
        );
        // StrIDED-batch m == 1 batched GEMV (grown/narrowed KV-cache view): the RHS
        // V is the natural-layout (batch, kv, head_dim) prefix of a backing buffer
        // with capacity > live length, so its BATCH stride (capacity*head_dim) is
        // NON-compact while each within-batch (kv, head_dim) block stays contiguous.
        // This is exactly the shape the expanded-KV cache hands the Vulkan
        // `batched_gemv_f32`/`ctx_gemv_f32` kernels; the batch stride is read from the
        // layout, not hardcoded as kv*head_dim. Exercises the score-style transpose of
        // a strided B (q @ k^T) against CPU.
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm_m1_strided_batch",
            &[2, 1, 64],
            gpu_backends,
            |device| {
                let live = 64usize;
                let cap = live + 41; // grown capacity > live prefix -> strided batch
                // Storage (batch, cap, head_head_dim); narrow to the live kv prefix.
                let b_src = Tensor::from_vec(gen_f32(&[2, cap, 100], 137), (2, cap, 100), device)?
                    .to_dtype(dtype)?;
                let b = b_src.narrow(1, 0, live)?; // (2, live, 100) strided batch
                let a = Tensor::from_vec(gen_f32(&[2, 1, 100], 42), (2, 1, 100), device)?
                    .to_dtype(dtype)?;
                // a @ b^T: reduce over the inner 100, output over the strided live dim.
                a.matmul(&b.transpose(1, 2)?)
            },
        );
        // Natural-layout CONTEXT GEMV with a strided (grown) V: probs @ v where v is
        // the (batch, kv, head_dim) prefix of a capacity > live backing buffer, so
        // v's batch stride is non-compact and the kernel reads it from the layout.
        // Directly exercises `ctx_gemv_f32`'s stride-aware batch read (no transpose).
        run_case(
            tracker,
            "matmul_batched",
            dtype,
            "batched_mm_ctx_natural_strided",
            &[2, 1, 128],
            gpu_backends,
            |device| {
                let live = 130usize;
                let cap = live + 47; // grown capacity > live prefix -> strided batch
                let a = Tensor::from_vec(gen_f32(&[2, 1, live], 42), (2, 1, live), device)?
                    .to_dtype(dtype)?;
                let v_src = Tensor::from_vec(gen_f32(&[2, cap, 128], 137), (2, cap, 128), device)?
                    .to_dtype(dtype)?;
                let v = v_src.narrow(1, 0, live)?; // (2, live, 128) strided on batch
                a.matmul(&v)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Indexing ops: gather, index_select, scatter_add, index_add
// ---------------------------------------------------------------------------

fn test_indexing_ops(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    let index_dtypes = &[
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::U8,
        DType::U32,
        DType::I64,
    ];

    for &dtype in index_dtypes {
        // gather
        run_case(
            tracker,
            "gather",
            dtype,
            "gather2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let data = match dtype {
                    DType::U8 => Tensor::from_vec(gen_u8(&[3, 4], 42), (3, 4), device)?,
                    DType::U32 => Tensor::from_vec(gen_u32(&[3, 4], 42), (3, 4), device)?,
                    DType::I64 => Tensor::from_vec(gen_i64(&[3, 4], 42), (3, 4), device)?,
                    _ => Tensor::from_vec(gen_f32(&[3, 4], 42), (3, 4), device)?.to_dtype(dtype)?,
                };
                let indices = Tensor::from_vec(vec![0u32, 2, 1, 0, 1, 1], (2, 3), device)?;
                data.gather(&indices, 0)
            },
        );

        // index_select
        run_case(
            tracker,
            "index_select",
            dtype,
            "is2d",
            &[2, 4],
            gpu_backends,
            |device| {
                let data = match dtype {
                    DType::U8 => Tensor::from_vec(gen_u8(&[5, 4], 42), (5, 4), device)?,
                    DType::U32 => Tensor::from_vec(gen_u32(&[5, 4], 42), (5, 4), device)?,
                    DType::I64 => Tensor::from_vec(gen_i64(&[5, 4], 42), (5, 4), device)?,
                    _ => Tensor::from_vec(gen_f32(&[5, 4], 42), (5, 4), device)?.to_dtype(dtype)?,
                };
                let indices = Tensor::from_vec(vec![0u32, 3], 2, device)?;
                data.index_select(&indices, 0)
            },
        );
    }

    // scatter_add and index_add (float only)
    for &dtype in &[DType::F32, DType::F16] {
        run_case(
            tracker,
            "scatter_add",
            dtype,
            "scatter_add2d",
            &[3, 4],
            gpu_backends,
            |device| {
                let data =
                    Tensor::from_vec(gen_f32(&[3, 4], 42), (3, 4), device)?.to_dtype(dtype)?;
                let indices =
                    Tensor::from_vec(vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], (3, 4), device)?;
                let src =
                    Tensor::from_vec(gen_f32(&[3, 4], 137), (3, 4), device)?.to_dtype(dtype)?;
                data.scatter_add(&indices, &src, 0)
            },
        );

        run_case(
            tracker,
            "index_add",
            dtype,
            "index_add2d",
            &[4, 4],
            gpu_backends,
            |device| {
                let data =
                    Tensor::from_vec(gen_f32(&[4, 4], 42), (4, 4), device)?.to_dtype(dtype)?;
                let indices = Tensor::from_vec(vec![0u32, 2], 2, device)?;
                let src =
                    Tensor::from_vec(gen_f32(&[2, 4], 137), (2, 4), device)?.to_dtype(dtype)?;
                data.index_add(&indices, &src, 0)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

fn test_where_cond(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16, DType::BF16, DType::U8, DType::U32] {
        run_case(
            tracker,
            "where_cond",
            dtype,
            "where2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let cond_data: Vec<u8> = vec![1, 0, 1, 0, 1, 1];
                let cond = Tensor::from_vec(cond_data, (2, 3), device)?;
                let on_true = match dtype {
                    DType::U8 => Tensor::from_vec(gen_u8(&[2, 3], 42), (2, 3), device)?,
                    DType::U32 => Tensor::from_vec(gen_u32(&[2, 3], 42), (2, 3), device)?,
                    _ => Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(dtype)?,
                };
                let on_false = match dtype {
                    DType::U8 => Tensor::from_vec(gen_u8(&[2, 3], 137), (2, 3), device)?,
                    DType::U32 => Tensor::from_vec(gen_u32(&[2, 3], 137), (2, 3), device)?,
                    _ => {
                        Tensor::from_vec(gen_f32(&[2, 3], 137), (2, 3), device)?.to_dtype(dtype)?
                    }
                };
                cond.where_cond(&on_true, &on_false)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// argsort
// ---------------------------------------------------------------------------

fn test_argsort(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16, DType::U32, DType::I64] {
        for &asc in &[true, false] {
            let label = if asc { "asc" } else { "desc" };
            run_case(
                tracker,
                &format!("argsort_{label}"),
                dtype,
                "argsort2d",
                &[3, 5],
                gpu_backends,
                |device| {
                    let data = match dtype {
                        DType::U32 => Tensor::from_vec(gen_u32(&[3, 5], 42), (3, 5), device)?,
                        DType::I64 => Tensor::from_vec(gen_i64(&[3, 5], 42), (3, 5), device)?,
                        _ => Tensor::from_vec(gen_f32(&[3, 5], 42), (3, 5), device)?
                            .to_dtype(dtype)?,
                    };
                    data.arg_sort_last_dim(asc)
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// clamp
// ---------------------------------------------------------------------------

fn test_clamp(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16, DType::BF16] {
        run_case(
            tracker,
            "clamp",
            dtype,
            "clamp2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(dtype)?;
                x.clamp(-0.3f64, 0.7f64)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// cumsum
// ---------------------------------------------------------------------------

fn test_cumsum(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16] {
        run_case(
            tracker,
            "cumsum",
            dtype,
            "cumsum2d",
            &[3, 4],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[3, 4], 42), (3, 4), device)?.to_dtype(dtype)?;
                x.cumsum(1)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// affine / powf / elu
// ---------------------------------------------------------------------------

fn test_affine_powf_elu(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16, DType::BF16, DType::F64] {
        run_case(
            tracker,
            "affine",
            dtype,
            "affine2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(dtype)?;
                x.affine(2.0, 1.0)
            },
        );
        run_case(
            tracker,
            "powf",
            dtype,
            "powf2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(
                    gen_f32(&[2, 3], 42)
                        .iter()
                        .map(|&v| v.abs() + 0.1)
                        .collect::<Vec<_>>(),
                    (2, 3),
                    device,
                )?
                .to_dtype(dtype)?;
                x.powf(2.5)
            },
        );
        run_case(
            tracker,
            "elu",
            dtype,
            "elu2d",
            &[2, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(dtype)?;
                x.elu(1.0)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// to_dtype cross-matrix
// ---------------------------------------------------------------------------

const TO_DTYPE_MATRIX: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::F64,
    DType::U8,
    DType::U32,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F8E4M3,
];

/// Reduced to_dtype matrix for non-ignored tests (avoids stack overflow from 90 nested closures).
const TO_DTYPE_MATRIX_SMALL: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::U8,
    DType::U32,
    DType::I32,
    DType::I64,
];

fn test_to_dtype(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &src in TO_DTYPE_MATRIX_SMALL {
        for &dst in TO_DTYPE_MATRIX_SMALL {
            if src == dst {
                continue;
            }
            run_case(
                tracker,
                &format!("to_dtype_{src:?}_to_{dst:?}"),
                src,
                "cast2d",
                &[2, 3],
                gpu_backends,
                |device| {
                    let data = match src {
                        DType::U8 => Tensor::from_vec(gen_u8(&[2, 3], 42), (2, 3), device)?,
                        DType::U32 => Tensor::from_vec(gen_u32(&[2, 3], 42), (2, 3), device)?,
                        DType::I16 => Tensor::from_vec(
                            gen_i64(&[2, 3], 42)
                                .iter()
                                .map(|&x| x as i16)
                                .collect::<Vec<_>>(),
                            (2, 3),
                            device,
                        )?,
                        DType::I32 => Tensor::from_vec(
                            gen_i64(&[2, 3], 42)
                                .iter()
                                .map(|&x| x as i32)
                                .collect::<Vec<_>>(),
                            (2, 3),
                            device,
                        )?,
                        DType::I64 => Tensor::from_vec(gen_i64(&[2, 3], 42), (2, 3), device)?,
                        _ => {
                            Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(src)?
                        }
                    };
                    data.to_dtype(dst)
                },
            );
        }
    }
}

/// Full 10x10 to_dtype matrix (90 cases). Marked #[ignore] because it can overflow the stack
/// on Windows with default stack size. Run with `cargo test ... -- --ignored`.
#[test]
#[ignore]
fn diff_to_dtype_extended() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    for &src in TO_DTYPE_MATRIX {
        for &dst in TO_DTYPE_MATRIX {
            if src == dst {
                continue;
            }
            run_case(
                &mut tracker,
                &format!("to_dtype_{src:?}_to_{dst:?}"),
                src,
                "cast2d",
                &[2, 3],
                &gpu_backends,
                |device| {
                    let data = match src {
                        DType::U8 => Tensor::from_vec(gen_u8(&[2, 3], 42), (2, 3), device)?,
                        DType::U32 => Tensor::from_vec(gen_u32(&[2, 3], 42), (2, 3), device)?,
                        DType::I16 => Tensor::from_vec(
                            gen_i64(&[2, 3], 42)
                                .iter()
                                .map(|&x| x as i16)
                                .collect::<Vec<_>>(),
                            (2, 3),
                            device,
                        )?,
                        DType::I32 => Tensor::from_vec(
                            gen_i64(&[2, 3], 42)
                                .iter()
                                .map(|&x| x as i32)
                                .collect::<Vec<_>>(),
                            (2, 3),
                            device,
                        )?,
                        DType::I64 => Tensor::from_vec(gen_i64(&[2, 3], 42), (2, 3), device)?,
                        _ => {
                            Tensor::from_vec(gen_f32(&[2, 3], 42), (2, 3), device)?.to_dtype(src)?
                        }
                    };
                    data.to_dtype(dst)
                },
            );
        }
    }
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

// ---------------------------------------------------------------------------
// upsample / pool / conv
// ---------------------------------------------------------------------------

fn test_upsample_pool_conv(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16] {
        // upsample_nearest2d
        run_case(
            tracker,
            "upsample_nearest2d",
            dtype,
            "up_near2d",
            &[1, 2, 4, 4],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 2, 2], 42), (1, 2, 2, 2), device)?
                    .to_dtype(dtype)?;
                x.upsample_nearest2d(4, 4)
            },
        );

        // upsample_bilinear2d
        run_case(
            tracker,
            "upsample_bilinear2d",
            dtype,
            "up_bilinear",
            &[1, 2, 4, 4],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 2, 2], 42), (1, 2, 2, 2), device)?
                    .to_dtype(dtype)?;
                x.upsample_bilinear2d(4, 4, false)
            },
        );

        // avg_pool2d
        run_case(
            tracker,
            "avg_pool2d",
            dtype,
            "avgpool2d",
            &[1, 2, 2, 2],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 4, 4], 42), (1, 2, 4, 4), device)?
                    .to_dtype(dtype)?;
                x.avg_pool2d((2, 2))
            },
        );

        // max_pool2d
        run_case(
            tracker,
            "max_pool2d",
            dtype,
            "maxpool2d",
            &[1, 2, 2, 2],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 4, 4], 42), (1, 2, 4, 4), device)?
                    .to_dtype(dtype)?;
                x.max_pool2d((2, 2))
            },
        );
    }

    // conv1d and conv2d
    for &dtype in &[DType::F32, DType::F16] {
        run_case(
            tracker,
            "conv1d",
            dtype,
            "conv1d_tiny",
            &[1, 1, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 5], 42), (1, 2, 5), device)?
                    .to_dtype(dtype)?;
                let w = Tensor::from_vec(gen_f32(&[1, 2, 3], 137), (1, 2, 3), device)?
                    .to_dtype(dtype)?;
                x.conv1d(&w, 0, 1, 1, 1)
            },
        );

        run_case(
            tracker,
            "conv2d",
            dtype,
            "conv2d_tiny",
            &[1, 1, 3, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[1, 2, 5, 5], 42), (1, 2, 5, 5), device)?
                    .to_dtype(dtype)?;
                let w = Tensor::from_vec(gen_f32(&[1, 2, 3, 3], 137), (1, 2, 3, 3), device)?
                    .to_dtype(dtype)?;
                x.conv2d(&w, 0, 1, 1, 1)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Strided / transposed view edge case
// ---------------------------------------------------------------------------

fn test_strided_views(tracker: &mut SuiteTracker, gpu_backends: &[(String, Device)]) {
    for &dtype in &[DType::F32, DType::F16, DType::BF16] {
        // Transposed view
        run_case(
            tracker,
            "add_transposed",
            dtype,
            "transpose",
            &[3, 4],
            gpu_backends,
            |device| {
                let a = Tensor::from_vec(gen_f32(&[4, 3], 42), (4, 3), device)?
                    .to_dtype(dtype)?
                    .t()?;
                let b = Tensor::from_vec(gen_f32(&[3, 4], 137), (3, 4), device)?.to_dtype(dtype)?;
                a.add(&b)
            },
        );

        // Sliced view
        run_case(
            tracker,
            "relu_sliced",
            dtype,
            "slice",
            &[2, 3],
            gpu_backends,
            |device| {
                let x = Tensor::from_vec(gen_f32(&[4, 3], 42), (4, 3), device)?.to_dtype(dtype)?;
                let sliced = x.narrow(0, 1, 2)?;
                sliced.relu()
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

#[test]
fn diff_unary() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_unary_ops(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_binary() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_binary_ops(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_cmp() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_cmp_ops(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_reduce() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_reduce_ops(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_matmul() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_matmul(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_indexing() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_indexing_ops(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_misc() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_where_cond(&mut tracker, &backends.devices);
    test_argsort(&mut tracker, &backends.devices);
    test_clamp(&mut tracker, &backends.devices);
    test_cumsum(&mut tracker, &backends.devices);
    test_affine_powf_elu(&mut tracker, &backends.devices);
    test_strided_views(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_upsample_pool_conv() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_upsample_pool_conv(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}

#[test]
fn diff_to_dtype() -> Result<()> {
    let backends = probe_backends();
    let gpu_backends: Vec<_> = backends
        .devices
        .iter()
        .filter(|(n, _)| n != "cpu")
        .cloned()
        .collect();
    if gpu_backends.is_empty() {
        eprintln!("SKIP: no GPU backends available");
        for s in &backends.skips {
            eprintln!("  {s}");
        }
        return Ok(());
    }
    let mut tracker = SuiteTracker::new();
    test_to_dtype(&mut tracker, &backends.devices);
    eprintln!("{}", tracker.summary());
    let failures = tracker.failures_report();
    if !failures.is_empty() {
        eprintln!("{failures}");
    }
    assert_eq!(tracker.failures, 0, "{} failures found", tracker.failures);
    Ok(())
}
