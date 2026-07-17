use candle::{Device, Result, Tensor};

#[test]
fn neginf_where() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2,2), &cpu)?;
    let m = Tensor::from_slice(&[0u8, 1, 0, 1], (2,2), &cpu)?;
    let xg = x.to_device(&wg)?; let mg = m.to_device(&wg)?;
    let t = Tensor::new(f32::NEG_INFINITY, &cpu)?.broadcast_as((2,2))?;
    let tg = Tensor::new(f32::NEG_INFINITY, &wg)?.broadcast_as((2,2))?;
    println!("neginf cpu {:?}", t.to_vec2::<f32>()?);
    println!("neginf wg  {:?}", tg.to_vec2::<f32>()?);
    let yc = m.where_cond(&t, &x)?;
    let yg = mg.where_cond(&tg, &xg)?;
    wg.synchronize()?;
    println!("where cpu {:?}", yc.to_vec2::<f32>()?);
    println!("where wg  {:?}", yg.to_vec2::<f32>()?);
    Ok(())
}
