use candle::{DType, Device, IndexOp, Result, Tensor};

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(candle::D::Minus1)?;
    let xs1 = xs.narrow(candle::D::Minus1, 0, last/2)?.contiguous()?;
    let xs2 = xs.narrow(candle::D::Minus1, last/2, last-last/2)?.contiguous()?;
    Tensor::cat(&[&xs2.neg()?, &xs1], candle::D::Minus1)
}

#[test]
fn attn_pieces() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // qkv reshape path
    let h = Tensor::randn(0f32, 0.1, (64, 3072), &cpu)?;
    let qkv_c = h.reshape((64, 3, 16, 64))?.permute((1,0,2,3))?.contiguous()?;
    let qkv_v = h.to_device(&vk)?.reshape((64, 3, 16, 64))?.permute((1,0,2,3))?.contiguous()?;
    vk.synchronize()?;
    println!("permute contig {}", maxdiff(&qkv_v, &qkv_c)?);
    let q_c = qkv_c.i(0)?.squeeze(0)?.contiguous()?;
    let q_v = qkv_v.i(0)?.squeeze(0)?.contiguous()?;
    println!("q slice {}", maxdiff(&q_v, &q_c)?);

    // vision rope style
    let cos = Tensor::randn(0f32, 0.1, (64, 64), &cpu)?;
    let sin = Tensor::randn(0f32, 0.1, (64, 64), &cpu)?;
    let q = Tensor::randn(0f32, 0.1, (64, 16, 64), &cpu)?;
    let cos_u = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin_u = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let qe_c = (q.broadcast_mul(&cos_u)? + rotate_half(&q)?.broadcast_mul(&sin_u)?)?;
    let qv = q.to_device(&vk)?;
    let cos_v = cos.to_device(&vk)?.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin_v = sin.to_device(&vk)?.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let qe_v = (qv.broadcast_mul(&cos_v)? + rotate_half(&qv)?.broadcast_mul(&sin_v)?)?;
    vk.synchronize()?;
    println!("vision rope {}", maxdiff(&qe_v, &qe_c)?);

    // linear 1024->3072
    let w = Tensor::randn(0f32, 0.05, (3072, 1024), &cpu)?;
    let b = Tensor::randn(0f32, 0.05, 3072, &cpu)?;
    let x = Tensor::randn(0f32, 0.1, (64, 1024), &cpu)?;
    let y_c = x.matmul(&w.t()?)?.broadcast_add(&b)?;
    let y_v = x.to_device(&vk)?.matmul(&w.to_device(&vk)?.t()?)?.broadcast_add(&b.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("linear qkv {}", maxdiff(&y_v, &y_c)?);

    // full mini attn like vision
    let seq=64usize; let nh=16; let hd=64;
    let q = Tensor::randn(0f32, 0.1, (seq, nh, hd), &cpu)?;
    let k = Tensor::randn(0f32, 0.1, (seq, nh, hd), &cpu)?;
    let v = Tensor::randn(0f32, 0.1, (seq, nh, hd), &cpu)?;
    let qc = q.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kc = k.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vc = v.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let att = (qc.matmul(&kc.t()?)? / (hd as f64).sqrt())?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let o_c = att.matmul(&vc)?.squeeze(0)?.transpose(0,1)?.reshape((seq, nh*hd))?;
    let qv=q.to_device(&vk)?; let kv=k.to_device(&vk)?; let vv=v.to_device(&vk)?;
    let qg = qv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kg = kv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vg = vv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let attg = (qg.matmul(&kg.t()?)? / (hd as f64).sqrt())?;
    let attg = candle_nn::ops::softmax_last_dim(&attg)?;
    let o_v = attg.matmul(&vg)?.squeeze(0)?.transpose(0,1)?.reshape((seq, nh*hd))?;
    vk.synchronize()?;
    println!("full mini attn {}", maxdiff(&o_v, &o_c)?);
    Ok(())
}
