use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?.to_dtype(candle::DType::F32)?.unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)
}

#[test]
fn pos() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let p_c = sinusoids(1500, 384, &cpu)?;
    let p_v = sinusoids(1500, 384, &vk)?;
    vk.synchronize()?;
    println!("pos emb md {}", md(&p_c, &p_v)?);
    let a = Tensor::arange(0u32, 1500, &vk)?.to_dtype(candle::DType::F32)?;
    let ac = Tensor::arange(0u32, 1500, &cpu)?.to_dtype(candle::DType::F32)?;
    println!("arange md {}", md(&ac, &a)?);
    let s_c = ac.sin()?;
    let s_v = a.sin()?;
    println!("sin md {}", md(&s_c, &s_v)?);
    Ok(())
}
