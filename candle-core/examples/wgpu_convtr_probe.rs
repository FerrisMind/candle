use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn run(dtype: DType) -> candle_core::Result<()> {
    let dev = Device::new_wgpu(0)?;
    let input_shape = [1usize, 4, 50, 50];
    let kernel_shape = [4usize, 1, 5, 5];
    let n = input_shape.iter().product::<usize>();
    let x_f32: Vec<f32> = (0..n).map(|v| v as f32 / n as f32).collect();
    let kn = kernel_shape.iter().product::<usize>();
    let w_f32: Vec<f32> = (0..kn).map(|v| (v as f32 / kn as f32) * 0.5).collect();
    let to_t = |v: &Vec<f32>, sh: &[usize]| -> candle_core::Result<Tensor> {
        match dtype {
            DType::F32 => Tensor::from_vec(v.clone(), sh.to_vec(), &dev),
            DType::F16 => Tensor::from_vec(v.clone(), sh.to_vec(), &Device::Cpu)?
                .to_dtype(DType::F16)?
                .to_device(&dev),
            DType::BF16 => Tensor::from_vec(v.clone(), sh.to_vec(), &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .to_device(&dev),
            _ => unreachable!(),
        }
    };
    let x = to_t(&x_f32, &input_shape)?;
    let w = to_t(&w_f32, &kernel_shape)?;

    // Warmup + correctness vs CPU.
    let out = x.conv_transpose2d(&w, 1, 0, 2, 1)?;
    let out_cpu = x.to_device(&Device::Cpu)?.conv_transpose2d(
        &w.to_device(&Device::Cpu)?,
        1,
        0,
        2,
        1,
    )?;
    let g = out
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let c = out_cpu
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max = 0f32;
    for (a, b) in g.iter().zip(c.iter()) {
        max = max.max((a - b).abs());
    }
    println!("{dtype:?} shape={:?} max_abs_diff={max:e}", out.dims());

    // Timed.
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        let _o = x.conv_transpose2d(&w, 1, 0, 2, 1)?;
    }
    dev.synchronize()?;
    let el = start.elapsed();
    println!("{dtype:?} {:?}/iter", el / iters);
    Ok(())
}

fn main() -> candle_core::Result<()> {
    for d in [DType::F32, DType::F16, DType::BF16] {
        run(d)?;
    }
    Ok(())
}
