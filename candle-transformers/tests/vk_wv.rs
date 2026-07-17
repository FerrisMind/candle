use candle::{Device, Result, Tensor, D};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn wv() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // Exact whisper shapes: heads=6, seq=1500, dim=64
    let q = Tensor::randn(0f32, 1.0, (1, 6, 1500, 64), &cpu)?;
    let k = Tensor::randn(0f32, 1.0, (1, 6, 64, 1500), &cpu)?; // pre-transposed
    let v = Tensor::randn(0f32, 1.0, (1, 6, 1500, 64), &cpu)?;
    let qv = q.to_device(&vk)?;
    let kv = k.to_device(&vk)?;
    let vv = v.to_device(&vk)?;

    let qk_c = q.matmul(&k)?;
    let qk_v = qv.matmul(&kv)?;
    vk.synchronize()?;
    println!("qk {}", md(&qk_c, &qk_v)?);

    let w_c = candle_nn::ops::softmax_last_dim(&qk_c)?;
    let w_v = candle_nn::ops::softmax_last_dim(&qk_v)?;
    vk.synchronize()?;
    println!("soft {}", md(&w_c, &w_v)?);

    // no sync path
    let qk_v2 = qv.matmul(&kv)?;
    let w_v2 = candle_nn::ops::softmax_last_dim(&qk_v2)?;
    let wv_v2 = w_v2.matmul(&vv)?;
    let wv_c = w_c.matmul(&v)?;
    println!("wv nosync {}", md(&wv_c, &wv_v2)?);

    // with sync
    let wv_v = w_v.matmul(&vv)?;
    vk.synchronize()?;
    println!("wv sync {}", md(&wv_c, &wv_v)?);

    // use CPU softmax uploaded
    let w_up = w_c.to_device(&vk)?;
    let wv_up = w_up.matmul(&vv)?;
    vk.synchronize()?;
    println!("wv cpu-soft upload {}", md(&wv_c, &wv_up)?);
    Ok(())
}
