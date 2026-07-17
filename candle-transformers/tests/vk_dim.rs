use candle::{Device, Result, Tensor};
fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
#[test]
fn dim() -> Result<()> {
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    let k=1500usize; let n=64usize;
    // 2d
    let a = Tensor::randn(0f32, 0.1, (k,k), &cpu)?;
    let v = Tensor::randn(0f32, 0.1, (k,n), &cpu)?;
    println!("2d {}", md(&a.matmul(&v)?, &a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?)?);
    // 3d batch 1
    let a3 = a.reshape((1,k,k))?;
    let v3 = v.reshape((1,k,n))?;
    println!("3d b1 {}", md(&a3.matmul(&v3)?, &a3.to_device(&vk)?.matmul(&v3.to_device(&vk)?)?)?);
    // 4d like attention
    let a4 = Tensor::randn(0f32, 0.1, (1,6,k,k), &cpu)?;
    let v4 = Tensor::randn(0f32, 0.1, (1,6,k,n), &cpu)?;
    println!("4d {}", md(&a4.matmul(&v4)?, &a4.to_device(&vk)?.matmul(&v4.to_device(&vk)?)?)?);
    // 4d N=256
    let v4b = Tensor::randn(0f32, 0.1, (1,6,k,256), &cpu)?;
    println!("4d N256 {}", md(&a4.matmul(&v4b)?, &a4.to_device(&vk)?.matmul(&v4b.to_device(&vk)?)?)?);
    Ok(())
}
