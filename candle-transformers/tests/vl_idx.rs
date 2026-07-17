use candle::{DType, Device, IndexOp, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn idx() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let t = Tensor::arange(0f32, (3*64*16*64) as f32, &cpu)?.reshape((3,64,16,64))?;
    let tv = t.to_device(&vk)?;
    for i in 0..3 {
        let c = t.i(i)?.squeeze(0)?.contiguous()?;
        let v = tv.i(i)?.squeeze(0)?.contiguous()?;
        vk.synchronize()?;
        println!("i({i}) {}", md(&v, &c)?);
    }
    // also after permute from (64,3,16,64)
    let h = Tensor::arange(0f32, (64*3*16*64) as f32, &cpu)?.reshape((64,3,16,64))?;
    let hv = h.to_device(&vk)?;
    let p = h.permute((1,0,2,3))?.contiguous()?;
    let pv = hv.permute((1,0,2,3))?.contiguous()?;
    vk.synchronize()?;
    println!("permute {}", md(&pv, &p)?);
    for i in 0..3 {
        let c = p.i(i)?.squeeze(0)?.contiguous()?;
        let v = pv.i(i)?.squeeze(0)?.contiguous()?;
        vk.synchronize()?;
        println!("after perm i({i}) {}", md(&v, &c)?);
    }
    Ok(())
}
