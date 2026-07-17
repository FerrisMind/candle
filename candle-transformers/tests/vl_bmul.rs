use candle::{DType, Device, Result, Tensor};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn bmul() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // same shapes as vision rope
    let a = Tensor::randn(0f32, 1.0, (64, 16, 64), &cpu)?;
    let cos = Tensor::randn(0f32, 1.0, (64, 64), &cpu)?;
    let cos_u = cos.unsqueeze(1)?; // (64,1,64)
    let y_c = a.broadcast_mul(&cos_u)?;
    let y_v = a.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    println!("randn bmul {}", md(&y_v, &y_c)?);

    // ones * cos
    let ones = Tensor::ones((64, 16, 64), DType::F32, &cpu)?;
    let y_c = ones.broadcast_mul(&cos_u)?;
    let y_v = ones.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    println!("ones*cos {}", md(&y_v, &y_c)?);

    // sequential: two muls with same cos
    let a2 = Tensor::randn(0f32, 1.0, (64, 16, 64), &cpu)?;
    let y1 = a.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    let y2 = a2.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    println!("two muls a {}", md(&y1, &a.broadcast_mul(&cos_u)?)?);
    println!("two muls a2 {}", md(&y2, &a2.broadcast_mul(&cos_u)?)?);

    // exact shapes with CONTIG binary path
    let a = Tensor::arange(0f32, (64*16*64) as f32, &cpu)?.reshape((64,16,64))?;
    let c = Tensor::arange(0f32, (64*64) as f32, &cpu)?.reshape((64,64))?;
    let cu = c.unsqueeze(1)?;
    let y_c = a.broadcast_mul(&cu)?;
    let y_v = a.to_device(&vk)?.broadcast_mul(&c.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    println!("arange bmul {}", md(&y_v, &y_c)?);
    // print a few values
    let yc = y_c.flatten_all()?.to_vec1::<f32>()?;
    let yv = y_v.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("cpu[0..5] {:?}", &yc[..5]);
    println!("vk [0..5] {:?}", &yv[..5]);
    // find first mismatch
    for i in 0..yc.len() {
        if (yc[i]-yv[i]).abs() > 1e-3 {
            println!("first mismatch i={i} cpu={} vk={}", yc[i], yv[i]);
            break;
        }
    }
    Ok(())
}
