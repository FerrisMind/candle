use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn silu_mul_detail() -> Result<()> {
    let wg = Device::new_wgpu(0)?;
    let cpu = Device::Cpu;
    let a = Tensor::randn(0f32, 1.0, (1, 6, 768), &cpu)?;
    let b = Tensor::randn(0f32, 1.0, (1, 6, 768), &cpu)?;
    let ag = a.to_device(&wg)?;
    let bg = b.to_device(&wg)?;
    let s_c = (&a / (&a.neg()?.exp()? + 1.0)?)?;
    let m_c = (&s_c * &b)?;

    let s_g = (&ag / (&ag.neg()?.exp()? + 1.0)?)?;
    // force read silu
    let s_read = s_g.to_device(&Device::Cpu)?;
    println!("silu md {}", md(&s_c, &s_g)?);
    println!("silu vs readback {}", md(&s_c, &s_read)?);

    // mul using live s_g
    let m_g = (&s_g * &bg)?;
    println!("mul live s_g {}", md(&m_c, &m_g)?);

    // re-upload correct silu and mul
    let s_up = s_c.to_device(&wg)?;
    let m_up = (&s_up * &bg)?;
    println!("mul reupload silu {}", md(&m_c, &m_up)?);

    // mul using s_read uploaded
    let s_up2 = s_read.to_device(&wg)?;
    let m_up2 = (&s_up2 * &bg)?;
    println!("mul from gpu-read silu {}", md(&m_c, &m_up2)?);

    // Check if s_g still matches after mul
    println!("silu after mul still {}", md(&s_c, &s_g)?);

    // Clone s_g before mul
    let s_cl = s_g.copy()?; // or try_clone
    let m_cl = (&s_cl * &bg)?;
    println!("mul clone {}", md(&m_c, &m_cl)?);
    Ok(())
}
