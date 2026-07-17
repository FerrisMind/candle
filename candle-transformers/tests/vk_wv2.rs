use candle::{Device, Result, Tensor, D};

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn shapes() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // A: large_attn style
    let q = Tensor::randn(0f32, 0.1, (1, 6, 1500, 64), &cpu)?;
    let k = q.clone();
    let v = Tensor::randn(0f32, 0.1, (1, 6, 1500, 64), &cpu)?;
    let qv=q.to_device(&vk)?; let kv=k.to_device(&vk)?; let vv=v.to_device(&vk)?;
    let att_c = (q.matmul(&k.t()?)? / 8.0)?;
    let att_v = (qv.matmul(&kv.t()?)? / 8.0)?;
    let s_c = candle_nn::ops::softmax_last_dim(&att_c)?;
    let s_v = candle_nn::ops::softmax_last_dim(&att_v)?;
    let y_c = s_c.matmul(&v)?;
    let y_v = s_v.matmul(&vv)?;
    vk.synchronize()?;
    println!("style A (k.t) {}", md(&y_c, &y_v)?);

    // B: pretransposed k like whisper
    let k2 = Tensor::randn(0f32, 0.1, (1, 6, 64, 1500), &cpu)?;
    let k2v = k2.to_device(&vk)?;
    let att_c = (q.matmul(&k2)? / 8.0)?;
    let att_v = (qv.matmul(&k2v)? / 8.0)?;
    let s_c = candle_nn::ops::softmax_last_dim(&att_c)?;
    let s_v = candle_nn::ops::softmax_last_dim(&att_v)?;
    let y_c = s_c.matmul(&v)?;
    let y_v = s_v.matmul(&vv)?;
    vk.synchronize()?;
    println!("style B pre-T k {}", md(&y_c, &y_v)?);

    // C: large randn like vk_wv
    let q = Tensor::randn(0f32, 1.0, (1, 6, 1500, 64), &cpu)?;
    let k2 = Tensor::randn(0f32, 1.0, (1, 6, 64, 1500), &cpu)?;
    let v = Tensor::randn(0f32, 1.0, (1, 6, 1500, 64), &cpu)?;
    let qv=q.to_device(&vk)?; let k2v=k2.to_device(&vk)?; let vv=v.to_device(&vk)?;
    let att_c = q.matmul(&k2)?;
    let att_v = qv.matmul(&k2v)?;
    let s_c = candle_nn::ops::softmax_last_dim(&att_c)?;
    let s_v = candle_nn::ops::softmax_last_dim(&att_v)?;
    let y_c = s_c.matmul(&v)?;
    let y_v = s_v.matmul(&vv)?;
    vk.synchronize()?;
    println!("style C large vals {}", md(&y_c, &y_v)?);

    // D: only the matmul (1,6,1500,1500)@(1,6,1500,64) with random
    let w = Tensor::randn(0f32, 0.01, (1, 6, 1500, 1500), &cpu)?;
    // normalize rows roughly
    let w = candle_nn::ops::softmax_last_dim(&w)?;
    let wv = w.to_device(&vk)?;
    let y_c = w.matmul(&v)?;
    let y_v = wv.matmul(&vv)?;
    vk.synchronize()?;
    println!("style D random soft w@v {}", md(&y_c, &y_v)?);

    // E: identity-like w
    // just matmul random 1500x1500 @ 1500x64
    let a = Tensor::randn(0f32, 0.1, (1, 6, 1500, 1500), &cpu)?;
    let av = a.to_device(&vk)?;
    let y_c = a.matmul(&v)?;
    let y_v = av.matmul(&vv)?;
    vk.synchronize()?;
    println!("style E raw big@v {}", md(&y_c, &y_v)?);
    Ok(())
}
