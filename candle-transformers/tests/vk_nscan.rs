use candle::{Device, Result, Tensor};
fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
#[test]
fn nscan() -> Result<()> {
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    let k = 1500usize;
    for n in [16usize, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024, 1500] {
        let a = Tensor::randn(0f32, 0.1, (k, k), &cpu)?;
        let v = Tensor::randn(0f32, 0.1, (k, n), &cpu)?;
        let d = md(&a.matmul(&v)?, &a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?)?;
        println!("K=1500 N={n} maxdiff={d:.6e}");
    }
    // square 1500
    let a = Tensor::randn(0f32, 0.1, (1500, 1500), &cpu)?;
    let d = md(&a.matmul(&a)?, &a.to_device(&vk)?.matmul(&a.to_device(&vk)?)?)?;
    println!("1500^3 maxdiff={d:.6e}");
    Ok(())
}
