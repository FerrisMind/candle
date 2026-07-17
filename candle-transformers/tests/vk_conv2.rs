use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn conv_stride2() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // whisper-like: in 80 -> 384, then 384->384 stride 2, len 3000
    let x = Tensor::randn(0f32, 1.0, (1, 80, 3000), &cpu)?;
    let w1 = Tensor::randn(0f32, 0.05, (384, 80, 3), &cpu)?;
    let w2 = Tensor::randn(0f32, 0.05, (384, 384, 3), &cpu)?;
    let xv = x.to_device(&vk)?;
    let w1v = w1.to_device(&vk)?;
    let w2v = w2.to_device(&vk)?;
    // pad=1 stride=1
    let c1c = x.conv1d(&w1, 1, 1, 1, 1)?.gelu()?;
    let c1v = xv.conv1d(&w1v, 1, 1, 1, 1)?.gelu()?;
    vk.synchronize()?;
    println!("c1 {}", md(&c1c, &c1v)?);
    // pad=1 stride=2
    let c2c = c1c.conv1d(&w2, 1, 2, 1, 1)?.gelu()?;
    let c2v = c1v.conv1d(&w2v, 1, 2, 1, 1)?.gelu()?;
    vk.synchronize()?;
    println!("c2 stride2 {}", md(&c2c, &c2v)?);
    // chain more
    let mut a = c2c.clone();
    let mut b = c2v.clone();
    for i in 0..4 {
        let w = Tensor::randn(0f32, 0.05, (384, 384), &cpu)?;
        let wv = w.to_device(&vk)?;
        // linear via matmul: (1, T, 384) @ (384, 384)
        let at = a.transpose(1,2)?.contiguous()?; // (1, 1500, 384) wait c2 is (1,384,1500)
        // reshape
        let (b_sz, c, t) = a.dims3()?;
        let a2 = a.transpose(1,2)?.reshape((b_sz*t, c))?;
        let b2 = b.transpose(1,2)?.reshape((b_sz*t, c))?;
        let a2 = a2.matmul(&w.t()?)?;
        let b2 = b2.matmul(&wv.t()?)?;
        a = a2.reshape((b_sz, t, c))?.transpose(1,2)?.contiguous()?;
        b = b2.reshape((b_sz, t, c))?.transpose(1,2)?.contiguous()?;
        vk.synchronize()?;
        println!("layer{i} {}", md(&a, &b)?);
    }
    Ok(())
}
