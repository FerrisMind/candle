use candle::{Device, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
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
fn whisper_stages() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let weights = find(&hub, "model.safetensors", "whisper-tiny");
    let config_p = find(&hub, "config.json", "whisper-tiny");
    let config: whisper::Config = serde_json::from_str(&std::fs::read_to_string(&config_p)?).unwrap();
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let cpu_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], whisper::DTYPE, &cpu)? };
    let vk_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], whisper::DTYPE, &vk)? };
    let mut cpu_m = whisper::model::Whisper::load(&cpu_vb, config.clone())?;
    let mut vk_m = whisper::model::Whisper::load(&vk_vb, config.clone())?;
    let mel_len = config.num_mel_bins * whisper::N_FRAMES;
    let mel: Vec<f32> = (0..mel_len).map(|i| {
        let mixed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(0x5151u64.wrapping_mul(1442695040888963407));
        (((mixed >> 17) % 257) as i64 - 128) as f32 / 19.0
    }).collect();
    let shape = (1, config.num_mel_bins, whisper::N_FRAMES);
    let mel_c = Tensor::from_vec(mel.clone(), shape, &cpu)?;
    let mel_v = Tensor::from_vec(mel, shape, &vk)?;
    // full encoder
    let e_c = cpu_m.encoder.forward(&mel_c, true)?;
    let e_v = vk_m.encoder.forward(&mel_v, true)?;
    vk.synchronize()?;
    println!("full encoder md {}", md(&e_c, &e_v)?);
    // dtype of model
    println!("dtype {:?}", whisper::DTYPE);
    Ok(())
}
