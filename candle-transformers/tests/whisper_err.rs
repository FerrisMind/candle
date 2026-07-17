use candle::{Device, DType, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper;

#[test]
fn whisper_error_stats() -> Result<()> {
    let config_path = std::env::var("USERPROFILE").unwrap() + "\\.cache\\huggingface\\hub";
    // use same download path as matrix via candle test helpers - reimplement minimal
    let device = Device::new_vulkan(0)?;
    let cpu = Device::Cpu;
    // Find weights
    let hub = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/huggingface/hub");
    let weights = walkdir_find(&hub, "model.safetensors", "whisper-tiny");
    let config_p = walkdir_find(&hub, "config.json", "whisper-tiny");
    println!("weights={weights:?} config={config_p:?}");
    let config: whisper::Config = serde_json::from_str(&std::fs::read_to_string(&config_p)?)
        .map_err(|e| candle::Error::msg(format!("{e}")))?;
    let cpu_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], whisper::DTYPE, &cpu)? };
    let dev_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], whisper::DTYPE, &device)? };
    let mut cpu_m = whisper::model::Whisper::load(&cpu_vb, config.clone())?;
    let mut dev_m = whisper::model::Whisper::load(&dev_vb, config.clone())?;
    let mel_len = config.num_mel_bins * whisper::N_FRAMES;
    let mel: Vec<f32> = (0..mel_len).map(|i| {
        let mixed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(0x5151u64.wrapping_mul(1442695040888963407));
        let bucket = ((mixed >> 17) % 257) as i64 - 128;
        bucket as f32 / 19.0
    }).collect();
    let shape = (1, config.num_mel_bins, whisper::N_FRAMES);
    let mel_c = Tensor::from_vec(mel.clone(), shape, &cpu)?;
    let mel_d = Tensor::from_vec(mel, shape, &device)?;
    let ec = cpu_m.encoder.forward(&mel_c, true)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let ed = dev_m.encoder.forward(&mel_d, true)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    device.synchronize()?;
    let mut max_abs=0f32; let mut max_rel=0f32; let mut sum_sq=0f32; let mut sum_e=0f32;
    for (a,e) in ed.iter().zip(ec.iter()) {
        let d=(a-e).abs();
        max_abs=max_abs.max(d);
        let den=a.abs().max(e.abs()).max(1e-8);
        max_rel=max_rel.max(d/den);
        sum_sq += d*d;
        sum_e += e*e;
    }
    println!("n={} max_abs={} max_rel={} rmse={} rmse_rel={}", ed.len(), max_abs, max_rel, (sum_sq/ed.len() as f32).sqrt(), (sum_sq/sum_e.max(1e-12)).sqrt());
    println!("first5 cpu {:?} vulkan {:?}", &ec[..5], &ed[..5]);
    Ok(())
}

fn walkdir_find(root: &std::path::Path, name: &str, hint: &str) -> std::path::PathBuf {
    fn rec(p: &std::path::Path, name: &str, hint: &str, out: &mut Option<std::path::PathBuf>) {
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
    let mut o=None; rec(root, name, hint, &mut o); o.expect("not found")
}
