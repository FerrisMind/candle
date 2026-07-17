use candle::{Device, Result, Tensor};
fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
#[test]
fn remlo() -> Result<()> {
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    let n = 64usize;
    for rem in 0..32usize {
        let k = 1280 + rem;
        let a = Tensor::randn(0f32, 0.1, (k, k), &cpu)?;
        let v = Tensor::randn(0f32, 0.1, (k, n), &cpu)?;
        let d = md(&a.matmul(&v)?, &a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?)?;
        println!("K={k} rem={rem} d={d:.4e} {}", if d>0.01 {"FAIL"} else {"ok"});
    }
    Ok(())
}
