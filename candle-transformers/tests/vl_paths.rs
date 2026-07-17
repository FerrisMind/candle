use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

#[test]
fn paths() -> Result<()> {
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
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)? };
    let qkv = linear(1024, 3072, vb.pp("model.visual.blocks.0.attn.qkv"))?;
    let hs = qkv.forward(&norm1)?;
    let t = hs.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let k = t.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let kc = k.flatten_all()?.to_vec1::<f32>()?;
    let cc = cos.flatten_all()?.to_vec1::<f32>()?;

    // A: to_device
    let ya = k.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    let yav = ya.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("A to_device y0={} expect={}", yav[0], kc[0]*cc[0]);

    // B: from_vec on same device
    let kb = Tensor::from_vec(kc.clone(), (64,16,64), &vk)?;
    let cb = Tensor::from_vec(cc.clone(), (64,64), &vk)?;
    let yb = kb.broadcast_mul(&cb.unsqueeze(1)?)?;
    vk.synchronize()?;
    let ybv = yb.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("B from_vec y0={} expect={}", ybv[0], kc[0]*cc[0]);

    // C: to_device but force copy via flatten
    let kc2 = Tensor::from_vec(kc.clone(), (64,16,64), &cpu)?;
    let yc = kc2.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;
    let ycv = yc.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("C from_vec cpu then to_device y0={}", ycv[0]);
    Ok(())
}
