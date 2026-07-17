use candle::{Device, DType, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper;
use std::path::PathBuf;

fn md(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn find(root: &std::path::Path, name: &str, hint: &str) -> PathBuf {
    fn rec(p: &std::path::Path, name: &str, hint: &str, out: &mut Option<PathBuf>) {
        if out.is_some() { return; }
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() { rec(&path, name, hint, out); }
                else if path.file_name().and_then(|s| s.to_str()) == Some(name) {
                    if path.to_string_lossy().contains(hint) { *out = Some(path); }
                }
            }
        }
    }
    let mut o=None; rec(root, name, hint, &mut o); o.unwrap()
}

#[test]
fn w() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let weights = find(&hub, "model.safetensors", "whisper-tiny");
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let vb_c = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &cpu)? };
    let vb_v = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &vk)? };
    for name in [
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        "encoder.conv2.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.q_proj.bias",
        "encoder.layers.0.fc1.weight",
        "encoder.layer_norm.weight",
    ] {
        // try get with shape inference - use get_unchecked or similar
        // VarBuilder contains tensors - try common shapes
    }
    // Use contains
    let t_c = vb_c.get((384, 80, 3), "encoder.conv1.weight")?;
    let t_v = vb_v.get((384, 80, 3), "encoder.conv1.weight")?;
    println!("conv1.w {}", md(&t_c, &t_v)?);
    let t_c = vb_c.get(384, "encoder.conv1.bias")?;
    let t_v = vb_v.get(384, "encoder.conv1.bias")?;
    println!("conv1.b {}", md(&t_c, &t_v)?);
    let t_c = vb_c.get((384, 384), "encoder.layers.0.self_attn.q_proj.weight")?;
    let t_v = vb_v.get((384, 384), "encoder.layers.0.self_attn.q_proj.weight")?;
    println!("q_proj {}", md(&t_c, &t_v)?);
    Ok(())
}
