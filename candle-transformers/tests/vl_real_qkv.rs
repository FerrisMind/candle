use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::Config;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn real_qkv() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // load first vision block qkv weight
    let load = |dev: &Device| -> Result<Linear> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, dev)? };
        linear(cfg.vision_config.hidden_size, cfg.vision_config.hidden_size * 3, vb.pp("model.visual.blocks.0.attn.qkv"))
    };
    let qkv_c = load(&cpu)?;
    let qkv_v = load(&vk)?;
    // use same after_pos-ish random input scaled
    let x = Tensor::randn(0f32, 1.0, (64, 1024), &cpu)?;
    let y_c = qkv_c.forward(&x)?;
    let y_v = qkv_v.forward(&x.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("real qkv linear {}", maxdiff(&y_v, &y_c)?);
    // reshape path
    let r_c = y_c.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let r_v = y_v.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    println!("real reshape {}", maxdiff(&r_v, &r_c)?);
    let qc = r_c.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let qv = r_v.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    println!("real q {}", maxdiff(&qv, &qc)?);
    Ok(())
}
