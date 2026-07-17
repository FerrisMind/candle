use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

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
fn apply_rope(q: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let q = q.contiguous()?;
    let cos = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)
}

#[test]
fn rope_real() -> Result<()> {
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
    let m = Qwen3VLModel::new(&cfg, unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?})?;
    let st = m.forward_vision_debug_stages(&xs, &g)?;
    let norm1 = st.iter().find(|(n,_)| n=="b0_norm1").unwrap().1.clone();
    let cos = st.iter().find(|(n,_)| n=="rope_cos").unwrap().1.clone();
    let sin = st.iter().find(|(n,_)| n=="rope_sin").unwrap().1.clone();
    println!("cos/sin shapes {:?} {:?}", cos.dims(), sin.dims());

    let vb = unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?};
    let qkv = linear(1024, 3072, vb.pp("model.visual.blocks.0.attn.qkv"))?;
    let hs = qkv.forward(&norm1)?;
    let qkv_t = hs.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let q = qkv_t.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let k = qkv_t.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    println!("q shape {:?}", q.dims());

    let qe_c = apply_rope(&q, &cos, &sin)?;
    let qe_v = apply_rope(&q.to_device(&vk)?, &cos.to_device(&vk)?, &sin.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("apply rope real {}", maxdiff(&qe_v, &qe_c)?);

    // pieces
    let cos_u = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let cos_uv = cos.to_device(&vk)?.contiguous()?.unsqueeze(candle::D::Minus2)?;
    println!("broadcast mul cos {}", maxdiff(&q.to_device(&vk)?.broadcast_mul(&cos_uv)?, &q.broadcast_mul(&cos_u)?)?);
    println!("rotate_half {}", maxdiff(&rotate_half(&q.to_device(&vk)?)?, &rotate_half(&q)?)?);
    Ok(())
}
