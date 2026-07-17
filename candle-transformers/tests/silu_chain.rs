use candle::{Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn silu_chain() -> Result<()> {
    let wg = Device::new_wgpu(0)?;
    let cpu = Device::Cpu;
    let a = Tensor::randn(0f32, 1.0, (1, 6, 768), &cpu)?;
    let ag = a.to_device(&wg)?;

    // full silu
    let s_c = (&a / (&a.neg()?.exp()? + 1.0)?)?;
    let s_g = (&ag / (&ag.neg()?.exp()? + 1.0)?)?;
    println!("silu full {}", md(&s_c, &s_g)?);

    // step with sync
    let n_g = ag.neg()?; wg.synchronize()?;
    println!("neg {}", md(&a.neg()?, &n_g)?);
    let e_g = n_g.exp()?; wg.synchronize()?;
    println!("exp {}", md(&a.neg()?.exp()?, &e_g)?);
    let d_g = (&e_g + 1.0)?; wg.synchronize()?;
    println!("add1 {}", md(&(a.neg()?.exp()? + 1.0)?, &d_g)?);
    let s2 = (&ag / &d_g)?; wg.synchronize()?;
    println!("div {}", md(&s_c, &s2)?);

    // step no sync
    let n2 = ag.neg()?;
    let e2 = n2.exp()?;
    let d2 = (&e2 + 1.0)?;
    let s3 = (&ag / &d2)?;
    println!("silu nosync steps {}", md(&s_c, &s3)?);

    // just neg.exp without sync
    let e3 = ag.neg()?.exp()?;
    println!("neg.exp nosync {}", md(&a.neg()?.exp()?, &e3)?);

    // exp alone
    let e4 = ag.exp()?;
    println!("exp alone {}", md(&a.exp()?, &e4)?);

    // neg alone  
    let n4 = ag.neg()?;
    println!("neg alone {}", md(&a.neg()?, &n4)?);

    // chain of unaries
    let u = ag.neg()?.exp()?.neg()?.exp()?;
    let uc = a.neg()?.exp()?.neg()?.exp()?;
    println!("unary chain {}", md(&uc, &u)?);
    Ok(())
}
