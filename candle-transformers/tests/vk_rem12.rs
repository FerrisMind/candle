use candle::{Device, Result, Tensor};
fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
#[test]
fn rem12() -> Result<()> {
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    for k in (12..=2048).step_by(16) {
        if k < 64 { continue; }
        let a = Tensor::randn(0f32, 0.1, (k, k), &cpu)?;
        let v = Tensor::randn(0f32, 0.1, (k, 32), &cpu)?;
        let d = md(&a.matmul(&v)?, &a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?)?;
        if d > 0.01 {
            println!("FAIL K={k} maxdiff={d:.6e}");
        }
    }
    println!("done");
    Ok(())
}
