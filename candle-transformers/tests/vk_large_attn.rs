use candle::{Device, Result, Tensor, D};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn large_attn() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // whisper-like: batch=1, heads=6, seq=1500, dim=64 for tiny
    let q = Tensor::randn(0f32, 0.1, (1, 6, 1500, 64), &cpu)?;
    let k = Tensor::randn(0f32, 0.1, (1, 6, 1500, 64), &cpu)?;
    let v = Tensor::randn(0f32, 0.1, (1, 6, 1500, 64), &cpu)?;
    let qv=q.to_device(&vk)?; let kv=k.to_device(&vk)?; let vv=v.to_device(&vk)?;
    let att_c = (q.matmul(&k.t()?)? / 8.0)?;
    let att_v = (qv.matmul(&kv.t()?)? / 8.0)?;
    vk.synchronize()?;
    println!("scores {}", md(&att_c, &att_v)?);
    let s_c = candle_nn::ops::softmax(&att_c, D::Minus1)?;
    let s_v = candle_nn::ops::softmax(&att_v, D::Minus1)?;
    vk.synchronize()?;
    println!("softmax {}", md(&s_c, &s_v)?);
    let y_c = s_c.matmul(&v)?;
    let y_v = s_v.matmul(&vv)?;
    vk.synchronize()?;
    println!("attn out {}", md(&y_c, &y_v)?);
    Ok(())
}
