use candle::{DType, Device, Result, Tensor};

#[test]
fn fresh_paths() -> Result<()> {
    let cpu = Device::Cpu;
    let k = Tensor::randn(0f32, 1.0, (64, 16, 64), &cpu)?;
    let noise = Tensor::randn(0f32, 0.5, (64, 64), &cpu)?;
    let cos = ((noise * 0.01)? + 1.0)?;

    let vk = Device::new_vulkan(0)?;
    let y1 = k.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    let y1c = k.broadcast_mul(&cos.unsqueeze(1)?)?;
    let d1 = {
        let a=y1.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b=y1c.flatten_all()?.to_vec1::<f32>()?;
        a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max)
    };
    println!("fresh simple to_device bmul md={d1}");

    for _ in 0..100 {
        let t = Tensor::randn(0f32, 1.0, (1024, 1024), &vk)?;
        let _ = (t.relu()? + 1.0)?;
    }
    vk.synchronize()?;
    let y2 = k.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    let d2 = {
        let a=y2.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b=y1c.flatten_all()?.to_vec1::<f32>()?;
        a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max)
    };
    println!("after 100 ops bmul md={d2}");
    Ok(())
}
