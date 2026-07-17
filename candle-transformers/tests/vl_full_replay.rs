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
fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q = q.contiguous()?; let k = k.contiguous()?;
    let cos = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    Ok(((q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?,
        (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?))
}

fn attn_block(q: &Tensor, k: &Tensor, v: &Tensor, proj: &candle_nn::Linear) -> Result<Tensor> {
    let q2 = q.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let k2 = k.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let v2 = v.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let att = (q2.matmul(&k2.transpose(2,3)?.contiguous()?)? / 8.0)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let o = att.matmul(&v2.contiguous()?)?.squeeze(0)?.transpose(0,1)?.contiguous()?.reshape((64,1024))?;
    proj.forward(&o)
}

#[test]
fn full_replay() -> Result<()> {
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
    let attn_ref = st.iter().find(|(n,_)| n=="b0_attn").unwrap().1.clone();

    let load = |dev: &Device| -> Result<(candle_nn::Linear, candle_nn::Linear)> {
        let vb = unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, dev)?};
        Ok((
            linear(1024, 3072, vb.pp("model.visual.blocks.0.attn.qkv"))?,
            linear(1024, 1024, vb.pp("model.visual.blocks.0.attn.proj"))?,
        ))
    };
    let (qkv_c, proj_c) = load(&cpu)?;
    let (qkv_v, proj_v) = load(&vk)?;

    let run = |dev: &Device, norm: &Tensor, cos: &Tensor, sin: &Tensor, qkv: &candle_nn::Linear, proj: &candle_nn::Linear| -> Result<Tensor> {
        let hs = qkv.forward(norm)?;
        let t = hs.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
        let q = t.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
        let k = t.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
        let v = t.i(2)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
        let (q,k) = apply_rope(&q, &k, cos, sin)?;
        attn_block(&q, &k, &v, proj)
    };

    let out_c = run(&cpu, &norm1, &cos, &sin, &qkv_c, &proj_c)?;
    let out_v = run(&vk, &norm1.to_device(&vk)?, &cos.to_device(&vk)?, &sin.to_device(&vk)?, &qkv_v, &proj_v)?;
    vk.synchronize()?;
    println!("replay with rope {}", maxdiff(&out_v, &out_c)?);
    println!("replay vs cpu stage attn {}", maxdiff(&out_c, &attn_ref)?);
    println!("gpu replay vs cpu stage attn {}", maxdiff(&out_v, &attn_ref)?);
    Ok(())
}
