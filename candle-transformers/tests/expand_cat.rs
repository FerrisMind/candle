use candle::{Device, Result, Tensor, D};
use candle_nn::ops::softmax;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max))
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

#[test]
fn expand_cat_mask() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    // expand like repeat_kv
    let x = Tensor::randn(0f32, 1.0, (1, 6, 6, 32), &cpu)?; // b,s,n_kv,hd
    let xg = x.to_device(&wg)?;
    let n_rep = 3usize;
    let expand = |t: &Tensor| -> Result<Tensor> {
        let (b,s,nkv,hd) = t.dims4()?;
        t.unsqueeze(3)?.expand((b,s,nkv,n_rep,hd))?.reshape((b,s,nkv*n_rep,hd))
    };
    let e_c = expand(&x)?;
    let e_g = expand(&xg)?;
    wg.synchronize()?;
    println!("expand {}", maxdiff(&e_c, &e_g)?);
    // cat
    let a = Tensor::randn(0f32, 1.0, (1, 3, 6, 32), &cpu)?;
    let b = Tensor::randn(0f32, 1.0, (1, 3, 6, 32), &cpu)?;
    let ag=a.to_device(&wg)?; let bg=b.to_device(&wg)?;
    let c_c = Tensor::cat(&[&a,&b], 1)?.contiguous()?;
    let c_g = Tensor::cat(&[&ag,&bg], 1)?.contiguous()?;
    wg.synchronize()?;
    println!("cat {}", maxdiff(&c_c, &c_g)?);
    // full att with mask like llama
    let q = Tensor::randn(0f32, 1.0, (1, 18, 6, 32), &cpu)?.transpose(1,2)?.contiguous()?;
    let k = q.clone();
    let v = Tensor::randn(0f32, 1.0, (1, 18, 6, 32), &cpu)?.transpose(1,2)?.contiguous()?;
    let qg=q.to_device(&wg)?; let kg=k.to_device(&wg)?; let vg=v.to_device(&wg)?;
    let att_c = (q.matmul(&k.t()?)? / (32f64).sqrt())?;
    let att_g = (qg.matmul(&kg.t()?)? / (32f64).sqrt())?;
    // lower triangular mask 0 allow, 1 mask
    let mut m = vec![0u8; 6*6];
    for i in 0..6 { for j in 0..6 { if j>i { m[i*6+j]=1; } } }
    let mask = Tensor::from_slice(&m, (1,1,6,6), &cpu)?.broadcast_as(att_c.shape())?;
    let mask_g = mask.to_device(&wg)?;
    let att_c = masked_fill(&att_c, &mask, f32::NEG_INFINITY)?;
    let att_g = masked_fill(&att_g, &mask_g, f32::NEG_INFINITY)?;
    let att_c = softmax(&att_c, D::Minus1)?;
    let att_g = softmax(&att_g, D::Minus1)?;
    let y_c = att_c.matmul(&v.contiguous()?)?;
    let y_g = att_g.matmul(&vg.contiguous()?)?;
    wg.synchronize()?;
    println!("full att block {}", maxdiff(&y_c, &y_g)?);
    Ok(())
}
