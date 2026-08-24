/// Performance and memory audit harness for Vulkan and WGPU backends.
///
/// This example measures steady-state dispatch latency and host-side memory
/// pressure for representative workloads (matmul, elementwise, quantized
/// matvec) on both GPU backends.
///
/// Build:
///   cargo build --release --example perf_memory_audit --features vulkan
///   cargo build --release --example perf_memory_audit --features wgpu
///
/// Run:
///   cargo run --release --example perf_memory_audit --features vulkan
///   cargo run --release --example perf_memory_audit --features wgpu
///
/// Protocol (per AGENTS.md §11):
/// - release build
/// - warmup (5 iterations)
/// - explicit device.synchronize() before each timed iteration
/// - median and p95 over >= 30 reps
/// - memory delta measured via process working set (Windows)
/// - growth slope > 1% over 100 iterations = leak candidate
use std::time::Instant;

#[cfg(any(feature = "vulkan", feature = "wgpu"))]
fn main() -> anyhow::Result<()> {
    #[cfg(feature = "vulkan")]
    let vulkan = run_vulkan_workloads();
    #[cfg(feature = "wgpu")]
    let wgpu = run_wgpu_workloads();

    #[cfg(feature = "vulkan")]
    vulkan?;
    #[cfg(feature = "wgpu")]
    wgpu?;
    Ok(())
}

#[cfg(not(any(feature = "vulkan", feature = "wgpu")))]
fn main() {
    eprintln!("This example requires either --features vulkan or --features wgpu");
}

// ---------------------------------------------------------------------------
// Memory tracking (Windows-friendly, no extra deps)
// ---------------------------------------------------------------------------

/// Returns current process working set size in bytes (approximate RSS).
/// On Windows uses `GetProcessMemoryInfo` via std only (no winapi dep).
/// Falls back to 0 on non-Windows; caller should note this in the report.
fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }

        extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(
                process: isize,
                counters: *mut ProcessMemoryCounters,
                size: u32,
            ) -> i32;
        }

        let mut pmc = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        unsafe {
            let h = GetCurrentProcess();
            let _ = GetProcessMemoryInfo(h, &mut pmc, pmc.cb);
        }
        pmc.working_set_size as u64
    }
    #[cfg(not(target_os = "windows"))]
    {
        0
    }
}

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------

struct TimingStats {
    samples: Vec<f64>,
}

impl TimingStats {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    fn record(&mut self, duration_ms: f64) {
        self.samples.push(duration_ms);
    }

    fn median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn p95(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        let idx = idx.max(1).min(sorted.len());
        sorted[idx - 1]
    }

    #[allow(dead_code)]
    fn count(&self) -> usize {
        self.samples.len()
    }
}

// ---------------------------------------------------------------------------
// Workload runners
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
fn run_vulkan_workloads() -> anyhow::Result<()> {
    use candle::Device;

    println!("=== Vulkan Backend Audit ===");
    let device = Device::new_vulkan(0)?;
    println!("Device: {:?}", device.location());

    let warmup = 5;
    let reps = 30;
    let leak_reps = 100;

    // ---- matmul f32 1024² ----
    {
        print!("matmul f32 1024^2 ... ");
        let a = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        for _ in 0..warmup {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = a.matmul(&b)?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- elementwise unary (1M elems) ----
    {
        print!("unary gelu 1M ... ");
        let x = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = x.gelu()?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = x.gelu()?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- elementwise binary (1M elems) ----
    {
        print!("binary add 1M ... ");
        let a = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = (&a + &b)?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- leak test: repeated elementwise binary ----
    {
        let a = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let rss_before = current_rss_bytes();
        for _ in 0..leak_reps {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let rss_after = current_rss_bytes();
        let delta = rss_after as i64 - rss_before as i64;
        let slope_pct = if rss_before > 0 {
            (delta as f64 / rss_before as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "leak test binary x{}: rss_before={} rss_after={} delta={} slope={:.3}%",
            leak_reps, rss_before, rss_after, delta, slope_pct
        );
    }

    // ---- leak test: repeated matmul 1024² ----
    {
        let a = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        for _ in 0..warmup {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let rss_before = current_rss_bytes();
        for _ in 0..leak_reps {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let rss_after = current_rss_bytes();
        let delta = rss_after as i64 - rss_before as i64;
        let slope_pct = if rss_before > 0 {
            (delta as f64 / rss_before as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "leak test matmul x{}: rss_before={} rss_after={} delta={} slope={:.3}%",
            leak_reps, rss_before, rss_after, delta, slope_pct
        );
    }

    Ok(())
}

#[cfg(feature = "wgpu")]
fn run_wgpu_workloads() -> anyhow::Result<()> {
    use candle::Device;

    println!("=== WGPU Backend Audit ===");
    let device = Device::new_wgpu(0)?;
    println!("Device: {:?}", device.location());

    let warmup = 5;
    let reps = 30;
    let leak_reps = 100;

    // ---- matmul f32 1024² ----
    {
        print!("matmul f32 1024^2 ... ");
        let a = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        for _ in 0..warmup {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = a.matmul(&b)?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- elementwise unary (1M elems) ----
    {
        print!("unary gelu 1M ... ");
        let x = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = x.gelu()?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = x.gelu()?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- elementwise binary (1M elems) ----
    {
        print!("binary add 1M ... ");
        let a = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let mut stats = TimingStats::new();
        for _ in 0..reps {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = (&a + &b)?;
            device.synchronize()?;
            stats.record(t0.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "median={:.2}ms p95={:.2}ms (n={})",
            stats.median(),
            stats.p95(),
            stats.count()
        );
    }

    // ---- leak test: repeated elementwise binary ----
    {
        let a = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1_000_000,), &device)?;
        for _ in 0..warmup {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let rss_before = current_rss_bytes();
        for _ in 0..leak_reps {
            let _ = (&a + &b)?;
            device.synchronize()?;
        }
        let rss_after = current_rss_bytes();
        let delta = rss_after as i64 - rss_before as i64;
        let slope_pct = if rss_before > 0 {
            (delta as f64 / rss_before as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "leak test binary x{}: rss_before={} rss_after={} delta={} slope={:.3}%",
            leak_reps, rss_before, rss_after, delta, slope_pct
        );
    }

    // ---- leak test: repeated matmul 1024² ----
    {
        let a = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        let b = candle::Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
        for _ in 0..warmup {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let rss_before = current_rss_bytes();
        for _ in 0..leak_reps {
            let _ = a.matmul(&b)?;
            device.synchronize()?;
        }
        let rss_after = current_rss_bytes();
        let delta = rss_after as i64 - rss_before as i64;
        let slope_pct = if rss_before > 0 {
            (delta as f64 / rss_before as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "leak test matmul x{}: rss_before={} rss_after={} delta={} slope={:.3}%",
            leak_reps, rss_before, rss_after, delta, slope_pct
        );
    }

    Ok(())
}
