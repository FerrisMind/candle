use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn mat() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    for (as_, vs, name) in [
        (0.1f32, 0.1f32, "0.1/0.1"),
        (0.1, 1.0, "0.1/1"),
        (1.0, 0.1, "1/0.1"),
        (1.0, 1.0, "1/1"),
        (0.01, 1.0, "0.01/1"),
    ] {
        let a = Tensor::randn(0f32, as_, (1, 6, 1500, 1500), &cpu)?;
        let v = Tensor::randn(0f32, vs, (1, 6, 1500, 64), &cpu)?;
        let av = a.to_device(&vk)?;
        let vv = v.to_device(&vk)?;
        let y_c = a.matmul(&v)?;
        let y_v = av.matmul(&vv)?;
        vk.synchronize()?;
        println!("{name} {}", md(&y_c, &y_v)?);
    }
    // smaller K
    let a = Tensor::randn(0f32, 1.0, (1, 6, 256, 256), &cpu)?;
    let v = Tensor::randn(0f32, 1.0, (1, 6, 256, 64), &cpu)?;
    let y_c = a.matmul(&v)?;
    let y_v = a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("256x256@256x64 {}", md(&y_c, &y_v)?);
    // 1500 with batch 1 head 1
    let a = Tensor::randn(0f32, 1.0, (1, 1, 1500, 1500), &cpu)?;
    let v = Tensor::randn(0f32, 1.0, (1, 1, 1500, 64), &cpu)?;
    let y_c = a.matmul(&v)?;
    let y_v = a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("1head 1500 {}", md(&y_c, &y_v)?);
    // 2d matmul 1500x1500 @ 1500x64
    let a = Tensor::randn(0f32, 1.0, (1500, 1500), &cpu)?;
    let v = Tensor::randn(0f32, 1.0, (1500, 64), &cpu)?;
    let y_c = a.matmul(&v)?;
    let y_v = a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("2d 1500 {}", md(&y_c, &y_v)?);
    Ok(())
}
