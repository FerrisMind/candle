use candle::{Device, Result, Tensor, DType};
use candle_nn::Module;

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn vk_ops() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let x = Tensor::randn(0f32, 1.0, (1, 80, 3000), &cpu)?;
    let xv = x.to_device(&vk)?;
    let g_c = x.gelu()?;
    let g_v = xv.gelu()?;
    vk.synchronize()?;
    println!("gelu md {}", md(&g_c, &g_v)?);
    // small conv1d
    let w = Tensor::randn(0f32, 0.1, (16, 80, 3), &cpu)?;
    let wv = w.to_device(&vk)?;
    let c_c = x.conv1d(&w, 1, 1, 1, 1)?;
    let c_v = xv.conv1d(&wv, 1, 1, 1, 1)?;
    vk.synchronize()?;
    println!("conv1d md {}", md(&c_c, &c_v)?);
    let cg_c = c_c.gelu()?;
    let cg_v = c_v.gelu()?;
    println!("conv+gelu md {}", md(&cg_c, &cg_v)?);
    Ok(())
}
