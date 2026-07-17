use candle::{Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::{llama2_c, llama2_c_weights};
use std::fs::File;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(candle::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_dtype(candle::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max))
}

fn find(p: &std::path::Path, name: &str) -> Option<PathBuf> {
    if p.is_file() && p.file_name()?.to_str()? == name { return Some(p.to_path_buf()); }
    if p.is_dir() {
        for e in std::fs::read_dir(p).ok()? {
            if let Ok(e) = e { if let Some(f) = find(&e.path(), name) { return Some(f); } }
        }
    }
    None
}

#[test]
fn llama_parts() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let load_vb = |device: &Device| -> Result<(candle_nn::VarBuilder<'static>, llama2_c::Config)> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        // leak for 'static - tests only
        let vb = unsafe { std::mem::transmute::<_, candle_nn::VarBuilder<'static>>(vb) };
        Ok((vb, config))
    };
    let (vb_c, cfg) = load_vb(&cpu)?;
    let (vb_g, _) = load_vb(&wg)?;
    let wte_c = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_c.pp("model.embed_tokens"))?;
    let wte_g = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_g.pp("model.embed_tokens"))?;
    let ids = [1u32, 13, 42, 7, 19, 5];
    let ids_c = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_g = Tensor::from_slice(&ids, (1, ids.len()), &wg)?;
    let x_c = wte_c.forward(&ids_c)?;
    let x_g = wte_g.forward(&ids_g)?;
    wg.synchronize()?;
    println!("embed maxdiff {}", maxdiff(&x_c, &x_g)?);
    // linear projection
    let lin_c = candle_nn::linear(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.q_proj"))?;
    let lin_g = candle_nn::linear(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.q_proj"))?;
    // check weight match
    let q_c = lin_c.forward(&x_c)?;
    let q_g = lin_g.forward(&x_g)?;
    wg.synchronize()?;
    println!("q_proj maxdiff {}", maxdiff(&q_c, &q_g)?);
    // rms
    let rms_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp("model.layers.0.input_layernorm"))?;
    let rms_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp("model.layers.0.input_layernorm"))?;
    let n_c = rms_c.forward(&x_c)?;
    let n_g = rms_g.forward(&x_g)?;
    wg.synchronize()?;
    println!("rms maxdiff {}", maxdiff(&n_c, &n_g)?);
    // full model again after pieces
    Ok(())
}
