use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

#[test]
fn check() -> Result<()> {
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
    let q = t.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let kc = k.flatten_all()?.to_vec1::<f32>()?;
    let qc = q.flatten_all()?.to_vec1::<f32>()?;

    let k_vk = k.to_device(&vk)?;
    let cos_vk = cos.to_device(&vk)?;
    vk.synchronize()?;
    // re-read k before mul
    let kr = k_vk.to_vec1_check()?;
    println!("before mul k_vk[0]={}", kr[0]);

    let y = k_vk.broadcast_mul(&cos_vk.unsqueeze(1)?)?;
    vk.synchronize()?;
    let yv = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // re-read k after mul
    let kr2 = k_vk.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("after mul k_vk[0]={} (should still be k)", kr2[0]);
    println!("y[0]={} expect {}", yv[0], kc[0]);
    println!("equals q0? {}", (yv[0]-qc[0]).abs()<1e-6);
    // Try broadcast_mul with explicit contiguous clone of both
    let k2 = k.copy()?.to_device(&vk)?;
    let c2 = cos.copy()?.to_device(&vk)?;
    vk.synchronize()?;
    let y2 = k2.contiguous()?.broadcast_mul(&c2.contiguous()?.unsqueeze(1)?.contiguous()?)?;
    vk.synchronize()?;
    let y2v = y2.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("copy path y[0]={}", y2v[0]);
    Ok(())
}

trait ToVec {
    fn to_vec1_check(&self) -> Result<Vec<f32>>;
}
impl ToVec for Tensor {
    fn to_vec1_check(&self) -> Result<Vec<f32>> {
        self.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
    }
}
