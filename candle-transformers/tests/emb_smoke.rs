use candle::{Device, DType, Result, Tensor};

#[test]
fn emb_smoke() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    // small vocab embedding via index_select
    let w = Tensor::arange(0f32, 32.0, &cpu)?.reshape((8, 4))?;
    let ids = Tensor::from_slice(&[1u32, 3, 5, 0], (1, 4), &cpu)?;
    let w_g = w.to_device(&wg)?;
    let ids_g = ids.to_device(&wg)?;
    let y_c = w.index_select(&ids.flatten_all()?, 0)?;
    let y_g = w_g.index_select(&ids_g.flatten_all()?, 0)?;
    wg.synchronize()?;
    println!("cpu {:?}", y_c.to_vec2::<f32>()?);
    println!("wg  {:?}", y_g.to_vec2::<f32>()?);
    // matmul
    let a = Tensor::randn(0f32, 1.0, (6, 16), &cpu)?;
    let b = Tensor::randn(0f32, 1.0, (16, 8), &cpu)?;
    let a_g = a.to_device(&wg)?; let b_g = b.to_device(&wg)?;
    let yc = a.matmul(&b)?;
    let yg = a_g.matmul(&b_g)?;
    wg.synchronize()?;
    let vc = yc.flatten_all()?.to_vec1::<f32>()?;
    let vg = yg.flatten_all()?.to_vec1::<f32>()?;
    let mut md=0f32;
    for (x,y) in vc.iter().zip(vg.iter()) { md=md.max((x-y).abs()); }
    println!("matmul maxdiff {md}");
    // rope_slow path
    let x = Tensor::randn(0f32, 1.0, (1, 4, 6, 8), &cpu)?;
    let cos = Tensor::randn(0f32, 1.0, (6, 4), &cpu)?;
    let sin = Tensor::randn(0f32, 1.0, (6, 4), &cpu)?;
    let xg = x.to_device(&wg)?; let cg = cos.to_device(&wg)?; let sg = sin.to_device(&wg)?;
    let rc = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
    let rg = candle_nn::rotary_emb::rope(&xg, &cg, &sg)?;
    wg.synchronize()?;
    let vc = rc.flatten_all()?.to_vec1::<f32>()?;
    let vg = rg.flatten_all()?.to_vec1::<f32>()?;
    md=0.0;
    for (x,y) in vc.iter().zip(vg.iter()) { md=md.max((x-y).abs()); }
    println!("rope maxdiff {md}");
    Ok(())
}
