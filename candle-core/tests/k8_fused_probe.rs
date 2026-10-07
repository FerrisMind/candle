//! Fused Q8K dp4a correctness probe for the vulkan backend (campaign C2).
//!
//! For every k-quant dtype (Q2K, Q3K, Q4K, Q5K, Q6K) this probe runs
//! `QMatMul::forward` on the vulkan device and compares the output against two
//! CPU references:
//!
//!   (a) f32 matmul of CPU-dequantized weights against the SAME random
//!       activation rounded onto the CPU `BlockQ8K` grid (`from_float` +
//!       `to_float`) — exactly what the vulkan raw-f32 fallback path (Q8K
//!       producer + raw-f32 kernel) computes.
//!   (b) CPU `QMatMul` — the native k-quant contract (integer dot products on
//!       the BlockQ8K grid with per-block f32 scales).
//!
//! Both bounds are checked separately: rel_rms (sqrt of nmse) must be <= 1e-5.
//!
//! Route evidence (anti-silent-fallback check): the probe prints the predicted
//! kernel branch per case, replicated from the locked routing code in
//! `vulkan_backend.rs`, plus the bit-identical output fraction vs each
//! reference. A bit-identical match to reference (a) on a large GEMM is a
//! fallback smell (printed as a warning, not a failure) — the fused dp4a
//! kernels accumulate in a different order than a plain f32 matmul. k8-perf's
//! timing is the independent confirmation of the fused route.
//!
//! Note on the "k = 300" fallback-gate case from the brief: `QTensor::quantize`
//! (`check_shape`) rejects any k-quant weight whose last dim is not a multiple
//! of the 256 block, so a k % 256 != 0 k-quant weight cannot exist. The raw-f32
//! fallback is instead exercised on this device through the matvec routing
//! gates (Q3K/Q6K always; Q4K/Q5K with input_m == 1 and k <= 4096 on NVIDIA)
//! and through padded non-contiguous activation slices (`k8_fused_probe_padded_slice`,
//! see `k300_k_quant_weight_quantize_is_rejected` for the exact rejection).
//!
//! Run (GPU lock mandatory):
//!   bash G:/jobs/candle-based-projects/.swarm-verify/tools/with_gpu_lock.sh \
//!     cargo test -p candle-core --features vulkan --test k8_fused_probe -- \
//!     --test-threads=1 --nocapture

use candle_core::quantized::k_quants::{BlockQ8K, GgmlType};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

/// Tolerance from the locked design (C2): rel <= 1e-5 against both references.
const REL_TOL: f64 = 1e-5;

/// Fixed seed so every run and every device sees identical data.
const SEED: u64 = 0x5EED_00C7;

// ---------------------------------------------------------------------------
// deterministic RNG (xorshift64*) — identical stream on every device
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------

struct CaseMetrics {
    rel_rms: f64,
    nmse: f64,
    max_abs: f64,
    bit_eq: usize,
    total: usize,
}

fn compare(actual: &[f32], reference: &[f32]) -> CaseMetrics {
    assert_eq!(actual.len(), reference.len(), "output length mismatch");
    let n = actual.len() as f64;
    let mut se = 0f64;
    let mut ref_pow = 0f64;
    let mut max_abs = 0f64;
    let mut bit_eq = 0usize;
    for (a, b) in actual.iter().zip(reference.iter()) {
        let d = (*a - *b) as f64;
        se += d * d;
        ref_pow += (*b as f64) * (*b as f64);
        max_abs = max_abs.max(d.abs());
        if a.to_bits() == b.to_bits() {
            bit_eq += 1;
        }
    }
    let mse = se / n;
    let nmse = mse / ref_pow.max(1e-30);
    CaseMetrics {
        rel_rms: nmse.sqrt(),
        nmse,
        max_abs,
        bit_eq,
        total: actual.len(),
    }
}

/// Round an activation stream onto the CPU `BlockQ8K` grid and dequantize back
/// to f32 (per 256-element block: signed extreme, f32 scale).
fn q8k_roundtrip(xs: &[f32]) -> Vec<f32> {
    assert!(xs.len() % 256 == 0, "q8k roundtrip needs len % 256 == 0");
    let mut quant = vec![BlockQ8K::zeros(); xs.len() / 256];
    BlockQ8K::from_float(xs, &mut quant);
    let mut out = vec![0f32; xs.len()];
    BlockQ8K::to_float(&quant, &mut out);
    out
}

// ---------------------------------------------------------------------------
// route prediction (replicated from the locked vulkan_backend.rs routing)
// ---------------------------------------------------------------------------

const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    FusedGemm,
    RawF32Gemm,
    FusedMatvec,
    RawF32Matvec,
}

struct DevInfo {
    name: String,
    vendor_id: u32,
    subgroup_size: u32,
    subgroup_min: u32,
    subgroup_max: u32,
    integer_dot: bool,
    subgroup_arithmetic: bool,
    subgroup_size_control: bool,
}

fn is_k_quant(dt: GgmlDType) -> bool {
    matches!(
        dt,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    )
}

/// Replicates `vulkan_should_use_mmvq` (the fused/legacy matvec gate).
fn should_use_mmvq(dev: &DevInfo, dt: GgmlDType, input_m: usize, k: usize) -> bool {
    if matches!(dt, GgmlDType::Q3K | GgmlDType::Q6K) {
        return false;
    }
    if input_m > 1 {
        return true;
    }
    match dev.vendor_id {
        VENDOR_ID_NVIDIA => {
            if dt == GgmlDType::Q2K {
                return true;
            }
            if k <= 4096 {
                return false;
            }
            dt != GgmlDType::Q8_0
        }
        VENDOR_ID_AMD => {
            if k < 2048 {
                return false;
            }
            dt != GgmlDType::Q8_0
        }
        VENDOR_ID_INTEL => {
            if k < 2048 {
                return false;
            }
            !matches!(dt, GgmlDType::Q4_0 | GgmlDType::Q5_1)
        }
        _ => true,
    }
}

/// Predicted branch for a k-quant matmul with the given shape. The spirv
/// existence checks of the real gates are treated as satisfied (the 21 fused
/// modules are registered by build.rs; verified by k8-compile).
fn predicted_route(dev: &DevInfo, dt: GgmlDType, input_m: usize, k: usize) -> Route {
    let fused_gate = is_k_quant(dt)
        && dev.integer_dot
        && dev.subgroup_size <= 32;
    if input_m > 8 {
        return if fused_gate {
            Route::FusedGemm
        } else {
            Route::RawF32Gemm
        };
    }
    if fused_gate && should_use_mmvq(dev, dt, input_m, k) {
        Route::FusedMatvec
    } else {
        Route::RawF32Matvec
    }
}

// ---------------------------------------------------------------------------
// probe cases
// ---------------------------------------------------------------------------

struct Case {
    label: &'static str,
    dtype: GgmlDType,
    /// activation rows (input_m): > 8 -> GEMM branch, <= 8 -> matvec branch
    m: usize,
    /// reduction dim; the weight is (n, k), k % 256 == 0
    k: usize,
    /// weight rows (output dim)
    n: usize,
    /// feed the activation as a non-contiguous narrow of a wider padded buffer
    padded: bool,
}

fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let dtypes = [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];
    for &dt in &dtypes {
        cases.push(Case {
            label: "gemm-large",
            dtype: dt,
            m: 256,
            k: 512,
            n: 128,
            padded: false,
        });
        cases.push(Case {
            label: "gemm-small",
            dtype: dt,
            m: 64,
            k: 768,
            n: 24,
            padded: false,
        });
        cases.push(Case {
            label: "matvec-m1",
            dtype: dt,
            m: 1,
            k: 512,
            n: 96,
            padded: false,
        });
        cases.push(Case {
            label: "matvec-m4",
            dtype: dt,
            m: 4,
            k: 768,
            n: 96,
            padded: false,
        });
    }
    // Padded non-contiguous activation slices (Q4K): exercise the strided copy
    // path on both branches. m=16 > 8 -> GEMM, m=4 <= 8 -> matvec.
    cases.push(Case {
        label: "gemm-padded-slice",
        dtype: GgmlDType::Q4K,
        m: 16,
        k: 512,
        n: 64,
        padded: true,
    });
    cases.push(Case {
        label: "matvec-padded-slice",
        dtype: GgmlDType::Q4K,
        m: 4,
        k: 512,
        n: 64,
        padded: true,
    });
    cases
}

fn run_case(dev: &Device, info: &DevInfo, case: &Case, rng: &mut Rng) -> Result<()> {
    let cpu = Device::Cpu;
    let Case {
        label,
        dtype,
        m,
        k,
        n,
        padded,
    } = *case;
    assert!(k % 256 == 0, "probe requires k % 256 == 0, got k={k}");
    let route = predicted_route(info, dtype, m, k);

    // Deterministic random data, identical on every device. The activation
    // carries a +0.25 DC offset so per-block integer sums (the k-quant min
    // term / bsums) are non-zero.
    let w_vals: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 0.5).collect();
    let x_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32() + 0.25).collect();

    let w_cpu = Tensor::from_slice(&w_vals, (n, k), &cpu)?;
    let x_cpu = Tensor::from_slice(&x_vals, (m, k), &cpu)?;

    // Same weights quantized on both devices: the k-quant weight quantize
    // contract must be bit-exact across devices.
    let qt_cpu = QTensor::quantize(&w_cpu, dtype)?;
    let w_vk = Tensor::from_slice(&w_vals, (n, k), dev)?;
    let qt_vk = QTensor::quantize(&w_vk, dtype)?;
    assert_eq!(
        qt_cpu.data()?.as_ref(),
        qt_vk.data()?.as_ref(),
        "{label} {dtype:?}: quantized weight bytes diverged between cpu and vulkan"
    );

    // Reference (b): CPU QMatMul (integer dot products on the Q8K grid).
    // The QTensor is kept alive behind an Arc so the weights can be
    // dequantized for reference (a) without a second quantize pass.
    let arc_cpu = std::sync::Arc::new(qt_cpu);
    let qmm_cpu = QMatMul::from_arc(arc_cpu.clone())?;
    let ref_b = qmm_cpu.forward(&x_cpu)?;

    // Reference (a): Q8K-rounded activation, f32 matmul with dequantized
    // weights — exactly what the vulkan raw-f32 fallback path computes.
    let w_f32 = arc_cpu.dequantize(&cpu)?.to_dtype(DType::F32)?;
    let x_q8k = q8k_roundtrip(&x_vals);
    let x_q8k_cpu = Tensor::from_slice(&x_q8k, (m, k), &cpu)?;
    let ref_a = x_q8k_cpu.matmul(&w_f32.t()?.contiguous()?)?;

    // Vulkan forward; optionally through a padded non-contiguous slice.
    let x_vk_input = if padded {
        let k_pad = k + 64;
        let mut padded_vals = vec![0f32; m * k_pad];
        for r in 0..m {
            for c in 0..k {
                padded_vals[r * k_pad + c] = x_vals[r * k + c];
            }
            for c in k..k_pad {
                padded_vals[r * k_pad + c] = rng.next_f32();
            }
        }
        let t = Tensor::from_slice(&padded_vals, (m, k_pad), dev)?;
        t.narrow(1, 0, k)?
    } else {
        Tensor::from_slice(&x_vals, (m, k), dev)?
    };

    let qmm_vk = QMatMul::from_qtensor(qt_vk)?;
    let out_vk = qmm_vk.forward(&x_vk_input)?;
    let out = out_vk.to_device(&cpu)?.to_dtype(DType::F32)?;
    let out_v = out.flatten_all()?.to_vec1::<f32>()?;
    let ref_a_v = ref_a.flatten_all()?.to_vec1::<f32>()?;
    let ref_b_v = ref_b
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let ma = check(label, "ref-a(Q8K f32)", &out_v, &ref_a_v)?;
    let mb = check(label, "ref-b(CPU QMatMul)", &out_v, &ref_b_v)?;

    // Bit-identity vs each reference.
    let bit_pct = |mm: &CaseMetrics| (mm.bit_eq as f64 / mm.total as f64) * 100.0;
    println!(
        "CASE {label} {dtype:?} m={m} k={k} n={n} route_pred={route:?} | vs(a) rel_rms={:.3e} nmse={:.3e} max_abs={:.3e} bit_eq={}/{} ({:.2}%) | vs(b) rel_rms={:.3e} nmse={:.3e} max_abs={:.3e} bit_eq={}/{} ({:.2}%)",
        ma.rel_rms, ma.nmse, ma.max_abs, ma.bit_eq, ma.total, bit_pct(&ma),
        mb.rel_rms, mb.nmse, mb.max_abs, mb.bit_eq, mb.total, bit_pct(&mb),
    );

    // Anti-silent-fallback check: the fused dp4a kernels accumulate in a
    // different order than a plain f32 matmul, so a bit-identical match to the
    // Q8K-rounded f32 reference on a large GEMM would smell like the raw-f32
    // fallback (or a plain f32 matmul) silently running instead.
    if route == Route::FusedGemm && ma.bit_eq == ma.total && ma.total >= 4096 {
        println!(
            "    WARNING {label} {dtype:?}: vulkan output BIT-identical to reference (a) on a large GEMM — fallback smell"
        );
    }

    // Padded-slice determinism: the strided copy must feed the kernel the exact
    // same contiguous buffer, so the output must be bit-identical to a run on
    // the contiguous twin.
    if padded {
        let x_cont = Tensor::from_slice(&x_vals, (m, k), dev)?;
        let out_cont = qmm_vk.forward(&x_cont)?;
        let out_cont_v = out_cont
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let eq = out_v
            .iter()
            .zip(out_cont_v.iter())
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        assert_eq!(
            eq,
            out_v.len(),
            "{label}: padded-slice output diverged from the contiguous twin"
        );
        println!("    padded-slice output bit-identical to contiguous twin: {eq}/{}", out_v.len());
    }
    Ok(())
}

fn check(label: &str, tag: &str, actual: &[f32], reference: &[f32]) -> Result<CaseMetrics> {
    let mm = compare(actual, reference);
    assert!(
        mm.rel_rms <= REL_TOL,
        "{label} vs {tag}: rel_rms={:e} nmse={:e} max_abs={:e} exceeds tol {REL_TOL}",
        mm.rel_rms,
        mm.nmse,
        mm.max_abs
    );
    Ok(mm)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
#[test]
fn k8_fused_probe_vulkan() -> Result<()> {
    use candle_core::backend::BackendDevice;
    let vdev = candle_core::vulkan_backend::VulkanDevice::new(0)?;
    let info = DevInfo {
        name: vdev.physical_device_name().to_string(),
        vendor_id: vdev.vendor_id(),
        subgroup_size: vdev.subgroup_size(),
        subgroup_min: vdev.subgroup_min_size(),
        subgroup_max: vdev.subgroup_max_size(),
        integer_dot: vdev.integer_dot_product_supported(),
        subgroup_arithmetic: vdev.subgroup_arithmetic_supported(),
        subgroup_size_control: vdev.subgroup_size_control_supported(),
    };
    println!(
        "device: {} vendor_id={:#06x} subgroup_size={} subgroup=[{},{}] integer_dot={} subgroup_arithmetic={} subgroup_size_control={}",
        info.name,
        info.vendor_id,
        info.subgroup_size,
        info.subgroup_min,
        info.subgroup_max,
        info.integer_dot,
        info.subgroup_arithmetic,
        info.subgroup_size_control,
    );
    let dev = Device::Vulkan(vdev);
    let mut rng = Rng::new(SEED);
    for case in all_cases() {
        run_case(&dev, &info, &case, &mut rng)?;
    }
    println!("k8_fused_probe: all cases passed (rel_rms <= {REL_TOL} vs both references)");
    Ok(())
}

/// The brief asks for a k % 256 != 0 case (k = 300). The CPU contract
/// (`check_shape`) rejects such weights outright — prove it with the exact
/// error so the report can document why the raw-f32 fallback is instead
/// exercised through the matvec routing gates and the padded-slice cases.
#[test]
fn k300_k_quant_weight_quantize_is_rejected() {
    let vals: Vec<f32> = (0..4 * 300).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let w = Tensor::from_slice(&vals, (4, 300), &Device::Cpu).unwrap();
    let err = QTensor::quantize(&w, GgmlDType::Q4K).unwrap_err();
    println!("k=300 Q4K quantize rejected with: {err}");
    assert!(
        err.to_string().contains("divisible by block size"),
        "unexpected error message: {err}"
    );
}
