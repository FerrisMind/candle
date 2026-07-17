use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn weights() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let wpath = dir.join("model.safetensors");
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let vb_c = unsafe { VarBuilder::from_mmaped_safetensors(&[wpath.clone()], DType::F32, &cpu)? };
    let vb_v = unsafe { VarBuilder::from_mmaped_safetensors(&[wpath], DType::F32, &vk)? };
    for name in [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.attn.qkv.bias",
        "model.visual.blocks.0.attn.proj.weight",
        "model.visual.blocks.0.norm1.weight",
        "model.visual.blocks.0.norm1.bias",
        "model.visual.patch_embed.proj.weight",
    ] {
        // try get with common shapes - use contains via get_unchecked?
        // VarBuilder doesn't list - use known shapes
    }
    let w_c = vb_c.get((3072, 1024), "model.visual.blocks.0.attn.qkv.weight")?;
    let w_v = vb_v.get((3072, 1024), "model.visual.blocks.0.attn.qkv.weight")?;
    println!("qkv.w {}", maxdiff(&w_v, &w_c)?);
    let b_c = vb_c.get(3072, "model.visual.blocks.0.attn.qkv.bias")?;
    let b_v = vb_v.get(3072, "model.visual.blocks.0.attn.qkv.bias")?;
    println!("qkv.b {}", maxdiff(&b_v, &b_c)?);
    let n_c = vb_c.get(1024, "model.visual.blocks.0.norm1.weight")?;
    let n_v = vb_v.get(1024, "model.visual.blocks.0.norm1.weight")?;
    println!("norm1.w {}", maxdiff(&n_v, &n_c)?);
    // patch embed weight 5d
    let p_c = vb_c.get((1024, 3, 2, 16, 16), "model.visual.patch_embed.proj.weight")?;
    let p_v = vb_v.get((1024, 3, 2, 16, 16), "model.visual.patch_embed.proj.weight")?;
    println!("patch.w {}", maxdiff(&p_v, &p_c)?);
    Ok(())
}
