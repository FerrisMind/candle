use candle::{Device, Result, Tensor, DType};
use candle_nn::Module;
use candle_transformers::models::{llama2_c, llama2_c_weights};
use std::fs::File;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
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
fn llama_manual() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let load = |device: &Device| -> Result<(candle_nn::VarBuilder, llama2_c::Config, llama2_c::Cache)> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        let cache = llama2_c::Cache::new(false, &config, vb.pp("rot"))?;
        Ok((vb, config, cache))
    };
    let (vb_c, cfg, cache_c) = load(&cpu)?;
    let (vb_g, _, cache_g) = load(&wg)?;
    let emb_c = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_c.pp("model.embed_tokens"))?;
    let emb_g = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_g.pp("model.embed_tokens"))?;
    let ids = [1u32, 13, 42, 7, 19, 5];
    let x_c = emb_c.forward(&Tensor::from_slice(&ids, (1, 6), &cpu)?)?;
    let x_g = emb_g.forward(&Tensor::from_slice(&ids, (1, 6), &wg)?)?;
    wg.synchronize()?;
    println!("emb {}", maxdiff(&x_c, &x_g)?);

    let w_c = vb_c.pp("model.layers.0.self_attn.q_proj").get((cfg.dim, cfg.dim), "weight")?;
    let w_g = vb_g.pp("model.layers.0.self_attn.q_proj").get((cfg.dim, cfg.dim), "weight")?;
    println!("W {}", maxdiff(&w_c, &w_g)?);
    let x2_c = x_c.reshape((6, cfg.dim))?;
    let x2_g = x_g.reshape((6, cfg.dim))?;
    let q_c = x2_c.matmul(&w_c.t()?)?.reshape((1, 6, cfg.dim))?;
    let q_g = x2_g.matmul(&w_g.t()?)?.reshape((1, 6, cfg.dim))?;
    wg.synchronize()?;
    println!("q matmul {}", maxdiff(&q_c, &q_g)?);

    let rms_w_c = vb_c.pp("model.layers.0.input_layernorm").get(cfg.dim, "weight")?;
    let rms_w_g = vb_g.pp("model.layers.0.input_layernorm").get(cfg.dim, "weight")?;
    let n_c = candle_nn::ops::rms_norm(&x_c, &rms_w_c, cfg.norm_eps as f32)?;
    let n_g = candle_nn::ops::rms_norm(&x_g, &rms_w_g, cfg.norm_eps as f32)?;
    wg.synchronize()?;
    println!("rms {}", maxdiff(&n_c, &n_g)?);

    let head_dim = cfg.dim / cfg.n_heads;
    let q4_c = n_c.reshape((1, 6, cfg.n_heads, head_dim))?.transpose(1,2)?.contiguous()?;
    let q4_g = n_g.reshape((1, 6, cfg.n_heads, head_dim))?.transpose(1,2)?.contiguous()?;
    let cos_c = cache_c.cos.narrow(0, 0, 6)?.squeeze(2)?;
    let sin_c = cache_c.sin.narrow(0, 0, 6)?.squeeze(2)?;
    let cos_g = cache_g.cos.narrow(0, 0, 6)?.squeeze(2)?;
    let sin_g = cache_g.sin.narrow(0, 0, 6)?.squeeze(2)?;
    println!("cos {}", maxdiff(&cos_c, &cos_g.to_device(&cpu)?)?);
    println!("sin {}", maxdiff(&sin_c, &sin_g.to_device(&cpu)?)?);
    let r_c = candle_nn::rotary_emb::rope_slow(&q4_c, &cos_c, &sin_c)?;
    let r_g = candle_nn::rotary_emb::rope_slow(&q4_g, &cos_g, &sin_g)?;
    wg.synchronize()?;
    println!("rope_slow {}", maxdiff(&r_c, &r_g)?);

    // attention scores q @ k^T
    let k4_c = r_c.clone();
    let k4_g = r_g.clone();
    let att_c = (r_c.matmul(&k4_c.t()?)? / (head_dim as f64).sqrt())?;
    let att_g = (r_g.matmul(&k4_g.t()?)? / (head_dim as f64).sqrt())?;
    wg.synchronize()?;
    println!("att scores {}", maxdiff(&att_c, &att_g)?);
    Ok(())
}
