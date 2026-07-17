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

#[test]
fn k_rope_debug() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let pe = 3*2*16*16;
    let mut data = vec![0f32; 64*pe];
    let mut s = 0xC0FFEEu64|1;
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (f32::from_bits(((s>>41) as u32)|0x3f800000)-1.0)*2.0-1.0;
    }
    let xs = Tensor::from_vec(data, (64, pe), &cpu)?;
    let g = Tensor::from_vec(vec![1u32,8,8], (1,3), &cpu)?;
    let m = Qwen3VLModel::new(&cfg, unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?
    })?;
    let st = m.forward_vision_debug_stages(&xs, &g)?;
    let norm1 = st.iter().find(|(n,_)| n=="b0_norm1").unwrap().1.clone();
    let cos = st.iter().find(|(n,_)| n=="rope_cos").unwrap().1.clone();
    let sin = st.iter().find(|(n,_)| n=="rope_sin").unwrap().1.clone();

    let vb_c = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)? };
    let vb_v = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &vk)? };
    let qkv_c = linear(1024, 3072, vb_c.pp("model.visual.blocks.0.attn.qkv"))?;
    let qkv_v = linear(1024, 3072, vb_v.pp("model.visual.blocks.0.attn.qkv"))?;
    let hs_c = qkv_c.forward(&norm1)?;
    let hs_v = qkv_v.forward(&norm1.to_device(&vk)?)?;
    vk.synchronize()?;
    let t_c = hs_c.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let t_v = hs_v.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    vk.synchronize()?;
    // extract k only
    let k_c = t_c.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k_v = t_v.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    vk.synchronize()?;
    println!("k pre {:?}", k_c.dims());
    println!("k pre md {}", md(&k_v, &k_c)?);
    // force re-upload k from CPU to eliminate qkv path differences
    let k_up = k_c.to_device(&vk)?;
    vk.synchronize()?;
    println!("k reupload md {}", md(&k_up, &k_c)?);

    let cos_v = cos.to_device(&vk)?.contiguous()?;
    let sin_v = sin.to_device(&vk)?.contiguous()?;
    let cos_u = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin_u = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let cos_uv = cos_v.unsqueeze(candle::D::Minus2)?;
    let sin_uv = sin_v.unsqueeze(candle::D::Minus2)?;

    // step rope on reuploaded k
    let mul1_c = k_c.broadcast_mul(&cos_u)?;
    let mul1_v = k_up.broadcast_mul(&cos_uv)?;
    vk.synchronize()?;
    println!("k*cos {}", md(&mul1_v, &mul1_c)?);

    let rh_c = rotate_half(&k_c)?;
    let rh_v = rotate_half(&k_up)?;
    vk.synchronize()?;
    println!("rotate_half(k) {}", md(&rh_v, &rh_c)?);

    let mul2_c = rh_c.broadcast_mul(&sin_u)?;
    let mul2_v = rh_v.broadcast_mul(&sin_uv)?;
    vk.synchronize()?;
    println!("rh*sin {}", md(&mul2_v, &mul2_c)?);

    let out_c = (&mul1_c + &mul2_c)?;
    let out_v = (&mul1_v + &mul2_v)?;
    vk.synchronize()?;
    println!("k rope sum {}", md(&out_v, &out_c)?);

    // same for q reupload
    let q_c = t_c.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let q_up = q_c.to_device(&vk)?;
    let qout_c = (q_c.broadcast_mul(&cos_u)? + rotate_half(&q_c)?.broadcast_mul(&sin_u)?)?;
    let qout_v = (q_up.broadcast_mul(&cos_uv)? + rotate_half(&q_up)?.broadcast_mul(&sin_uv)?)?;
    vk.synchronize()?;
    println!("q rope reupload {}", md(&qout_v, &qout_c)?);

    // strides
    println!("k_c stride {:?} contiguous={}", k_c.stride(), k_c.is_contiguous());
    println!("k_up stride {:?} contiguous={}", k_up.stride(), k_up.is_contiguous());
    println!("q_c stride {:?} contiguous={}", q_c.stride(), q_c.is_contiguous());
    Ok(())
}
