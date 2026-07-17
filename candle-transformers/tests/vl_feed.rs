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
fn feed_cpu_norm_to_gpu_attn() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let vc = &cfg.vision_config;
    let pe = vc.in_chans * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    let n = 64usize;
    let mut data = vec![0f32; n * pe];
    let mut s = 0xC0FFEEu64 | 1;
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (f32::from_bits(((s >> 41) as u32) | 0x3f80_0000) - 1.0) * 2.0 - 1.0;
    }
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let xs_c = Tensor::from_vec(data.clone(), (n, pe), &cpu)?;
    let g_c = Tensor::from_vec(vec![1u32, 8, 8], (1, 3), &cpu)?;
    let m_c = Qwen3VLModel::new(
        &cfg,
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?
        },
    )?;
    let stages_c = m_c.forward_vision_debug_stages(&xs_c, &g_c)?;
    let norm1_c = stages_c
        .iter()
        .find(|(n, _)| n == "b0_norm1")
        .unwrap()
        .1
        .clone();
    let attn_c = stages_c
        .iter()
        .find(|(n, _)| n == "b0_attn")
        .unwrap()
        .1
        .clone();
    let cos_c = stages_c
        .iter()
        .find(|(n, _)| n == "rope_cos")
        .unwrap()
        .1
        .clone();
    println!("have cpu norm1 {:?} attn {:?}", norm1_c.dims(), attn_c.dims());

    // Rebuild only vision path on GPU is hard without public attn.
    // Instead: run full GPU stages and also replace by comparing when we feed
    // GPU model but force inputs - use forward_vision_debug on GPU and compare
    // b0_attn when after_pos matches closely.
    let xs_v = Tensor::from_vec(data, (n, pe), &vk)?;
    let g_v = g_c.to_device(&vk)?;
    let m_v = Qwen3VLModel::new(
        &cfg,
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &vk)?
        },
    )?;
    let stages_v = m_v.forward_vision_debug_stages(&xs_v, &g_v)?;
    vk.synchronize()?;
    for name in ["b0_norm1", "b0_attn"] {
        let tc = stages_c.iter().find(|(n, _)| n == name).unwrap().1.clone();
        let tv = stages_v.iter().find(|(n, _)| n == name).unwrap().1.clone();
        println!("{name} {}", maxdiff(&tv, &tc)?);
    }
    // Upload CPU norm1 and compute a standalone linear+attn simulation is heavy.
    // Check absmax of CPU vs GPU attn outputs
    let ac = attn_c.flatten_all()?.to_vec1::<f32>()?;
    let av = attn_c.to_device(&Device::Cpu)?; // placeholder
    let av = stages_v
        .iter()
        .find(|(n, _)| n == "b0_attn")
        .unwrap()
        .1
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let amax_c = ac.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let amax_v = av.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("attn absmax cpu={amax_c} vk={amax_v}");
    println!("first5 cpu {:?} vk {:?}", &ac[..5], &av[..5]);
    let _ = cos_c;
    let _ = norm1_c;
    Ok(())
}
