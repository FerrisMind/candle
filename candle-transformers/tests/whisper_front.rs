use candle::{Device, Result, Tensor, DType};
use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder};
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
fn enc_front() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let weights = find(&hub, "model.safetensors", "whisper-tiny");
    let config_p = find(&hub, "config.json", "whisper-tiny");
    let config: whisper::Config = serde_json::from_str(&std::fs::read_to_string(&config_p)?).unwrap();
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let vb_c = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &cpu)? };
    let vb_v = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &vk)? };
    // rebuild conv path like encoder
    use candle_nn::{conv1d, Conv1dConfig};
    let n_state = config.d_model;
    let cfg1 = Conv1dConfig { padding: 1, stride: 1, groups: 1, dilation: 1, cudnn_fwd_algo: None };
    let cfg2 = Conv1dConfig { padding: 1, stride: 2, groups: 1, dilation: 1, cudnn_fwd_algo: None };
    let conv1_c = conv1d(config.num_mel_bins, n_state, 3, cfg1, vb_c.pp("encoder.conv1"))?;
    let conv1_v = conv1d(config.num_mel_bins, n_state, 3, cfg1, vb_v.pp("encoder.conv1"))?;
    let conv2_c = conv1d(n_state, n_state, 3, cfg2, vb_c.pp("encoder.conv2"))?;
    let conv2_v = conv1d(n_state, n_state, 3, cfg2, vb_v.pp("encoder.conv2"))?;
    let mel_len = config.num_mel_bins * whisper::N_FRAMES;
    let mel: Vec<f32> = (0..mel_len).map(|i| {
        let mixed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(0x5151u64.wrapping_mul(1442695040888963407));
        (((mixed >> 17) % 257) as i64 - 128) as f32 / 19.0
    }).collect();
    let shape = (1, config.num_mel_bins, whisper::N_FRAMES);
    let mel_c = Tensor::from_vec(mel.clone(), shape, &cpu)?;
    let mel_v = Tensor::from_vec(mel, shape, &vk)?;
    let x1_c = conv1_c.forward(&mel_c)?.gelu()?;
    let x1_v = conv1_v.forward(&mel_v)?.gelu()?;
    vk.synchronize()?;
    println!("conv1+gelu {}", md(&x1_c, &x1_v)?);
    let x2_c = conv2_c.forward(&x1_c)?.gelu()?;
    let x2_v = conv2_v.forward(&x1_v)?.gelu()?;
    vk.synchronize()?;
    println!("conv2+gelu {}", md(&x2_c, &x2_v)?);
    let x_c = x2_c.transpose(1,2)?;
    let x_v = x2_v.transpose(1,2)?;
    println!("transpose {}", md(&x_c, &x_v)?);
    // positional
    let pos_c = vb_c.get((config.max_source_positions, n_state), "encoder.positional_embedding")?;
    let pos_v = vb_v.get((config.max_source_positions, n_state), "encoder.positional_embedding")?;
    let seq = x_c.dims()[1];
    let pos_c = pos_c.narrow(0,0,seq)?;
    let pos_v = pos_v.narrow(0,0,seq)?;
    println!("pos {}", md(&pos_c, &pos_v)?);
    let x_c = x_c.broadcast_add(&pos_c)?;
    let x_v = x_v.broadcast_add(&pos_v)?;
    vk.synchronize()?;
    println!("+pos {}", md(&x_c, &x_v)?);
    // one block via re-running full encoder is hard; use layer_norm on long seq
    let ln = layer_norm(n_state, 1e-5, vb_c.pp("encoder.layer_norm"))?;
    let lnv = layer_norm(n_state, 1e-5, vb_v.pp("encoder.layer_norm"))?;
    let y_c = ln.forward(&x_c)?;
    let y_v = lnv.forward(&x_v)?;
    vk.synchronize()?;
    println!("ln_post-like on +pos {}", md(&y_c, &y_v)?);
    Ok(())
}
