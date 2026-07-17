use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn stages() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let vc = &cfg.vision_config;
    let pe = vc.in_chans * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    let n = 64usize;
    let mut data = vec![0f32; n*pe];
    let mut s=0xC0FFEEu64|1;
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (f32::from_bits((((s>>41) as u32)|0x3f800000)) - 1.0)*2.0 - 1.0;
    }
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let xs_c = Tensor::from_vec(data.clone(), (n, pe), &cpu)?;
    let xs_v = Tensor::from_vec(data, (n, pe), &vk)?;
    let g_c = Tensor::from_vec(vec![1u32,8,8], (1,3), &cpu)?;
    let g_v = g_c.to_device(&vk)?;
    let m_c = Qwen3VLModel::new(&cfg, unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?})?;
    let m_v = Qwen3VLModel::new(&cfg, unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &vk)?})?;
    let sc = m_c.forward_vision_debug_stages(&xs_c, &g_c)?;
    let sv = m_v.forward_vision_debug_stages(&xs_v, &g_v)?;
    vk.synchronize()?;
    for ((n1,t1),(n2,t2)) in sc.iter().zip(sv.iter()) {
        assert_eq!(n1,n2);
        println!("{n1:16} maxdiff={:.6e} shape={:?}", maxdiff(t2,t1)?, t1.dims());
    }
    Ok(())
}
