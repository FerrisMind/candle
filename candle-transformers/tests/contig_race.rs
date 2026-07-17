use candle::{Device, Result, Tensor};

#[test]
fn contig_race() -> Result<()> {
    let wg = Device::new_wgpu(0)?;
    let cpu = Device::Cpu;
    // random data
    let a = Tensor::randn(0f32, 1.0, (1, 6, 768), &cpu)?;
    let b = Tensor::randn(0f32, 1.0, (1, 6, 768), &cpu)?;
    let ag = a.to_device(&wg)?;
    let bg = b.to_device(&wg)?;
    // CPU reference with silu*mul chain
    let s_c = (&a / (&a.neg()?.exp()? + 1.0)?)?;
    let m_c = (&s_c * &b)?;

    // GPU no intermediate sync
    let s_g = (&ag / (&ag.neg()?.exp()? + 1.0)?)?;
    let m_g = (&s_g * &bg)?;
    let md = {
        let vc = m_c.flatten_all()?.to_vec1::<f32>()?;
        let vg = m_g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        vc.iter().zip(vg.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max)
    };
    println!("no-sync chain md={md}");

    // GPU with sync after silu
    let s_g2 = (&ag / (&ag.neg()?.exp()? + 1.0)?)?;
    wg.synchronize()?;
    let m_g2 = (&s_g2 * &bg)?;
    let md2 = {
        let vc = m_c.flatten_all()?.to_vec1::<f32>()?;
        let vg = m_g2.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        vc.iter().zip(vg.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max)
    };
    println!("sync-after-silu md={md2}");

    // Just mul alone
    let m_g3 = (&ag * &bg)?;
    let m_c3 = (&a * &b)?;
    let md3 = {
        let vc = m_c3.flatten_all()?.to_vec1::<f32>()?;
        let vg = m_g3.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        vc.iter().zip(vg.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max)
    };
    println!("single mul md={md3}");

    // Long chain of contig adds
    let mut x_c = a.clone();
    let mut x_g = ag.clone();
    for _ in 0..20 {
        x_c = (&x_c + &b)?;
        x_g = (&x_g + &bg)?;
    }
    let md4 = {
        let vc = x_c.flatten_all()?.to_vec1::<f32>()?;
        let vg = x_g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        vc.iter().zip(vg.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max)
    };
    println!("20 adds md={md4}");
    Ok(())
}
