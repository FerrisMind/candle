use candle::{Device, Result, Tensor};
fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
#[test]
fn near() -> Result<()> {
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    for k in [1488usize, 1496, 1500, 1504, 1512, 1520, 1536] {
        let a = Tensor::randn(0f32, 0.1, (k, k), &cpu)?;
        let v = Tensor::randn(0f32, 0.1, (k, 64), &cpu)?;
        let d = md(&a.matmul(&v)?, &a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?)?;
        println!("K={k} k%16={} k%32={} maxdiff={d:.6e}", k%16, k%32, d=d);
    }
    Ok(())
}
