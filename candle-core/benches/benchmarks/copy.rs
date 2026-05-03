use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run_upload_benchmark(c: &mut Criterion, device: &Device) {
    let rows = 512usize;
    let cols = 1024usize;
    let data = (0..rows * cols)
        .map(|idx| idx as f32 / 1024.0)
        .collect::<Vec<_>>();
    let size_in_bytes = data.len() * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("copy_upload_f32"));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let t = Tensor::from_slice(black_box(&data), (rows, cols), device).unwrap();
                black_box(t);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_download_benchmark(c: &mut Criterion, device: &Device) {
    let rows = 512usize;
    let cols = 1024usize;
    let data = (0..rows * cols)
        .map(|idx| idx as f32 / 1024.0)
        .collect::<Vec<_>>();
    let t = Tensor::from_slice(&data, (rows, cols), device).unwrap();
    let size_in_bytes = data.len() * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("copy_download_f32"));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                black_box(v);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_copy_strided_benchmark(c: &mut Criterion, device: &Device) {
    let src = Tensor::zeros((1024, 1024), DType::F32, device).unwrap();
    let strided = src.t().unwrap();
    let size_in_bytes = src.elem_count() * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("copy_strided_src_f32"));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let out = strided.contiguous().unwrap();
                black_box(out);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_copy2d_benchmark(c: &mut Criterion, device: &Device) {
    let lhs = Tensor::zeros((256, 1024), DType::F32, device).unwrap();
    let rhs = Tensor::ones((256, 1024), DType::F32, device).unwrap();
    let size_in_bytes = (lhs.elem_count() + rhs.elem_count()) * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("copy2d_cat_f32"));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let out = Tensor::cat(&[black_box(&lhs), black_box(&rhs)], 1).unwrap();
                black_box(out);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_upload_benchmark(c, &device);
        run_download_benchmark(c, &device);
        run_copy_strided_benchmark(c, &device);
        run_copy2d_benchmark(c, &device);
    }
}

criterion_group!(benches, criterion_benchmark);
