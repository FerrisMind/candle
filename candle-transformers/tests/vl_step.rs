use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
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
fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q = q.contiguous()?; let k = k.contiguous()?;
    let cos = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    Ok(((q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?,
        (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?))
}

#[test]
fn step() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let cpu = Device::Cpu; let vk = Device::new_vulkan(0)?;
    let pe=3*2*16*16; let mut data=vec![0f32;64*pe]; let mut s=0xC0FFEEu64|1;
    for x in data.iter_mut(){ s=s.wrapping_mul(6364136223846793005).wrapping_add(1); *x=(f32::from_bits(((s>>41) as u32)|0x3f800000)-1.0)*2.0-1.0; }
    let xs=Tensor::from_vec(data,(64,pe),&cpu)?; let g=Tensor::from_vec(vec![1u32,8,8],(1,3),&cpu)?;
    let m=Qwen3VLModel::new(&cfg, unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?})?;
    let st=m.forward_vision_debug_stages(&xs,&g)?;
    let norm1=st.iter().find(|(n,_)|n=="b0_norm1").unwrap().1.clone();
    let cos=st.iter().find(|(n,_)|n=="rope_cos").unwrap().1.clone();
    let sin=st.iter().find(|(n,_)|n=="rope_sin").unwrap().1.clone();
    let vb_c=unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?};
    let vb_v=unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &vk)?};
    let qkv_c=linear(1024,3072, vb_c.pp("model.visual.blocks.0.attn.qkv"))?;
    let qkv_v=linear(1024,3072, vb_v.pp("model.visual.blocks.0.attn.qkv"))?;
    let proj_c=linear(1024,1024, vb_c.pp("model.visual.blocks.0.attn.proj"))?;
    let proj_v=linear(1024,1024, vb_v.pp("model.visual.blocks.0.attn.proj"))?;

    let hs_c=qkv_c.forward(&norm1)?; let hs_v=qkv_v.forward(&norm1.to_device(&vk)?)?; vk.synchronize()?;
    println!("1 qkv {}", md(&hs_v,&hs_c)?);
    let t_c=hs_c.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let t_v=hs_v.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?; vk.synchronize()?;
    println!("2 reshape {}", md(&t_v,&t_c)?);
    let q_c=t_c.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k_c=t_c.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let v_c=t_c.i(2)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let q_v=t_v.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k_v=t_v.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let v_v=t_v.i(2)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?; vk.synchronize()?;
    println!("3 q {}", md(&q_v,&q_c)?); println!("3 k {}", md(&k_v,&k_c)?); println!("3 v {}", md(&v_v,&v_c)?);
    let (qr_c,kr_c)=apply_rope(&q_c,&k_c,&cos,&sin)?;
    let (qr_v,kr_v)=apply_rope(&q_v,&k_v,&cos.to_device(&vk)?,&sin.to_device(&vk)?)?; vk.synchronize()?;
    println!("4 rope q {}", md(&qr_v,&qr_c)?);
    println!("4 rope k {}", md(&kr_v,&kr_c)?);
    // attn
    let qc=qr_c.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kc=kr_c.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vc=v_c.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let qg=qr_v.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kg=kr_v.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vg=v_v.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let ac=(qc.matmul(&kc.transpose(2,3)?.contiguous()?)?/8.0)?;
    let ag=(qg.matmul(&kg.transpose(2,3)?.contiguous()?)?/8.0)?; vk.synchronize()?;
    println!("5 scores {}", md(&ag,&ac)?);
    let ac=candle_nn::ops::softmax_last_dim(&ac)?;
    let ag=candle_nn::ops::softmax_last_dim(&ag)?; vk.synchronize()?;
    println!("6 soft {}", md(&ag,&ac)?);
    let oc=ac.matmul(&vc.contiguous()?)?;
    let og=ag.matmul(&vg.contiguous()?)?; vk.synchronize()?;
    println!("7 ctx {}", md(&og,&oc)?);
    let oc=oc.squeeze(0)?.transpose(0,1)?.contiguous()?.reshape((64,1024))?;
    let og=og.squeeze(0)?.transpose(0,1)?.contiguous()?.reshape((64,1024))?;
    println!("8 reshape {}", md(&og,&oc)?);
    let oc=proj_c.forward(&oc)?; let og=proj_v.forward(&og)?; vk.synchronize()?;
    println!("9 proj {}", md(&og,&oc)?);
    Ok(())
}

