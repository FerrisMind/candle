use candle::{Device, DType, Result, Tensor, D};
use candle_nn::ops::softmax;

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[test]
fn attn_pieces() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let att_c = Tensor::randn(0f32, 1.0, (1, 2, 4, 4), &cpu)?;
    let att_g = att_c.to_device(&wg)?;
    let mask = Tensor::from_slice(
        &[0u8,1,1,1, 0,0,1,1, 0,0,0,1, 0,0,0,0],
        (1,1,4,4),
        &cpu,
    )?.broadcast_as((1,2,4,4))?;
    let mask_g = mask.to_device(&wg)?;
    let masked_c = masked_fill(&att_c, &mask, f32::NEG_INFINITY)?;
    let masked_g = masked_fill(&att_g, &mask_g, f32::NEG_INFINITY)?;
    wg.synchronize()?;
    let vc = masked_c.flatten_all()?.to_vec1::<f32>()?;
    let vg = masked_g.flatten_all()?.to_vec1::<f32>()?;
    println!("mask cpu first8 {:?}", &vc[..8]);
    println!("mask wg  first8 {:?}", &vg[..8]);
    let mut bad=0;
    for (a,b) in vc.iter().zip(vg.iter()) {
        if a.is_infinite() != b.is_infinite() || (!a.is_infinite() && (a-b).abs()>1e-4) { bad+=1; }
    }
    println!("mask mismatches {bad}");
    let sc = softmax(&masked_c, D::Minus1)?;
    let sg = softmax(&masked_g, D::Minus1)?;
    wg.synchronize()?;
    let vc = sc.flatten_all()?.to_vec1::<f32>()?;
    let vg = sg.flatten_all()?.to_vec1::<f32>()?;
    let mut md=0f32;
    for (a,b) in vc.iter().zip(vg.iter()) { md=md.max((a-b).abs()); }
    println!("softmax after mask maxdiff {md}");
    println!("soft cpu {:?}", &vc[..8]);
    println!("soft wg  {:?}", &vg[..8]);
    Ok(())
}
