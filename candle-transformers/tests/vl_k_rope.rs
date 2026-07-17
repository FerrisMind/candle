use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(candle::D::Minus1)?;
    let xs1 = xs.narrow(candle::D::Minus1, 0, last/2)?.contiguous()?;
    let xs2 = xs.narrow(candle::D::Minus1, last/2, last-last/2)?.contiguous()?;
    Tensor::cat(&[&xs2.neg()?, &xs1], candle::D::Minus1)
}
fn rope_one(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let cos = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    (x.broadcast_mul(&cos)? + rotate_half(&x)?.broadcast_mul(&sin)?)
}

#[test]
fn k_first() -> Result<()> {
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
    let hs_c=qkv_c.forward(&norm1)?; let hs_v=qkv_v.forward(&norm1.to_device(&vk)?)?;
    let t_c=hs_c.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let t_v=hs_v.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let q_c=t_c.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k_c=t_c.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let q_v=t_v.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k_v=t_v.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    // separate rope
    let qr_c=rope_one(&q_c,&cos,&sin)?;
    let kr_c=rope_one(&k_c,&cos,&sin)?;
    let qr_v=rope_one(&q_v,&cos.to_device(&vk)?,&sin.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("q rope alone {}", md(&qr_v,&qr_c)?);
    let kr_v=rope_one(&k_v,&cos.to_device(&vk)?,&sin.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("k rope alone after q {}", md(&kr_v,&kr_c)?);
    // k rope first on fresh
    let k_v2=t_v.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let kr_v2=rope_one(&k_v2,&cos.to_device(&vk)?,&sin.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("k rope first only {}", md(&kr_v2,&kr_c)?);
    // both without intermediate sync
    let cos_v=cos.to_device(&vk)?; let sin_v=sin.to_device(&vk)?;
    let qr=rope_one(&q_v,&cos_v,&sin_v)?;
    let kr=rope_one(&k_v,&cos_v,&sin_v)?;
    vk.synchronize()?;
    println!("q then k no mid-sync q={} k={}", md(&qr,&qr_c)?, md(&kr,&kr_c)?);
    Ok(())
}
