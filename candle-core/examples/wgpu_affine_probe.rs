use candle_core::{DType, Device, Tensor};
use half::bf16;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_wgpu(0)?;
    let cpu = Device::Cpu;
    let vals: Vec<f32> = vec![-1.5, 2.25, -0.75, 3.5, 0.125, -2.875];
    let x_cpu = Tensor::from_slice(&vals, (2, 3), &cpu)?.to_dtype(DType::BF16)?;
    let x_gpu = Tensor::from_slice(&vals, (2, 3), &dev)?.to_dtype(DType::BF16)?;
    let y_cpu = x_cpu.affine(2.0, 1.0)?;
    let y_gpu = x_gpu.affine(2.0, 1.0)?;
    println!("x   cpu: {:?}", x_cpu.to_vec2::<bf16>()?);
    println!("x   gpu: {:?}", x_gpu.to_vec2::<bf16>()?);
    println!("aff cpu: {:?}", y_cpu.to_vec2::<bf16>()?);
    println!("aff gpu: {:?}", y_gpu.to_vec2::<bf16>()?);
    Ok(())
}
