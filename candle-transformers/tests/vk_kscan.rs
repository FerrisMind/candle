use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn kscan() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    for k in [64usize, 128, 256, 512, 768, 1024, 1280, 1500, 1536, 2048] {
        let a = Tensor::randn(0f32, 0.1, (k, k), &cpu)?;
        let v = Tensor::randn(0f32, 0.1, (k, 64), &cpu)?;
        let y_c = a.matmul(&v)?;
        let y_v = a.to_device(&vk)?.matmul(&v.to_device(&vk)?)?;
        vk.synchronize()?;
        let d = md(&y_c, &y_v)?;
        // also relative-ish
        let yc = y_c.flatten_all()?.to_vec1::<f32>()?;
        let absmax = yc.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!("K={k} maxdiff={d:.6e} absmax_out={absmax:.4} rel={}", d/absmax.max(1e-6));
    }
    Ok(())
}
