use crate::benchmarks::BenchDevice;
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn vulkan_device_or_skip() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("Skipping Vulkan bench track: {err}");
            None
        }
    }
}

fn run_upload_download_bench(c: &mut Criterion, device: &Device) {
    let rows = 512;
    let cols = 512;
    let elem_count = rows * cols;
    let host_values = (0..elem_count)
        .map(|idx| idx as f32 * 0.001953125)
        .collect::<Vec<_>>();
    let moved_bytes = elem_count * std::mem::size_of::<f32>() * 2;

    let mut group = c.benchmark_group(device.bench_name("upload_download_f32"));
    group.throughput(Throughput::Bytes(moved_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tensor =
                    Tensor::from_slice(black_box(&host_values), (rows, cols), device).unwrap();
                black_box(tensor.to_vec2::<f32>().unwrap());
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_matmul_bench(c: &mut Criterion, device: &Device) {
    let m = 256;
    let k = 256;
    let n = 256;
    let lhs = Tensor::zeros((m, k), DType::F32, device).unwrap();
    let rhs = Tensor::zeros((k, n), DType::F32, device).unwrap();
    let flops = 2 * m * k * n;

    let mut group = c.benchmark_group(device.bench_name("matmul_f32_256"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(lhs.broadcast_matmul(black_box(&rhs)).unwrap());
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_copy2d_strided_bench(c: &mut Criterion, device: &Device) {
    let rows = 1024;
    let cols = 512;
    let src = Tensor::zeros((rows, cols), DType::F32, device).unwrap();
    let copied_bytes = rows * cols * std::mem::size_of::<f32>();

    let mut group = c.benchmark_group(device.bench_name("copy2d_strided_contiguous_f32"));
    group.throughput(Throughput::Bytes(copied_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let transposed = src.t().unwrap();
                black_box(transposed.contiguous().unwrap());
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let Some(device) = vulkan_device_or_skip() else {
        return;
    };
    run_upload_download_bench(c, &device);
    run_matmul_bench(c, &device);
    run_copy2d_strided_bench(c, &device);
}

criterion_group!(benches, criterion_benchmark);
