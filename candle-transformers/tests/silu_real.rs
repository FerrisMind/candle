use candle::{Device, Result, Tensor, DType};
use candle_nn::Module;
use candle_transformers::models::{llama2_c, llama2_c_weights};
use std::fs::File;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn find(p: &std::path::Path, name: &str) -> Option<PathBuf> {
    if p.is_file() && p.file_name()?.to_str()? == name { return Some(p.to_path_buf()); }
    if p.is_dir() { for e in std::fs::read_dir(p).ok()? { if let Ok(e)=e { if let Some(f)=find(&e.path(), name){return Some(f);} } } }
    None
}
fn silu(xs: &Tensor) -> Result<Tensor> { xs / (xs.neg()?.exp()? + 1.0)? }

#[test]
fn silu_real() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let mut file = File::open(&model_path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    // Build a realistic gate by running through weights from a known residual
    let mut file = File::open(&model_path)?;
    let _ = llama2_c::Config::from_reader(&mut file)?;
    let weights_c = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &cpu)?;
    let mut file = File::open(&model_path)?;
    let _ = llama2_c::Config::from_reader(&mut file)?;
    let weights_g = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &wg)?;
    let vb_c = weights_c.var_builder(&config, &cpu)?;
    let vb_g = weights_g.var_builder(&config, &wg)?;
    // use post-attn from embedding+identity residual scaled
    let emb_c = candle_nn::embedding(config.vocab_size, config.dim, vb_c.pp("model.embed_tokens"))?;
    let ids = [1u32,13,42,7,19,5];
    let x_c = emb_c.forward(&Tensor::from_slice(&ids,(1,6),&cpu)?)?;
    // crude: just use embedding as residual-like
    let rms2_c = candle_nn::rms_norm(config.dim, config.norm_eps, vb_c.pp("model.layers.0.post_attention_layernorm"))?;
    let rms2_g = candle_nn::rms_norm(config.dim, config.norm_eps, vb_g.pp("model.layers.0.post_attention_layernorm"))?;
    let n_c = rms2_c.forward(&x_c)?;
    let n_g = n_c.to_device(&wg)?;
    let g_c = candle_nn::linear_no_bias(config.dim, config.hidden_dim, vb_c.pp("model.layers.0.mlp.gate_proj"))?.forward(&n_c)?;
    let g_g = candle_nn::linear_no_bias(config.dim, config.hidden_dim, vb_g.pp("model.layers.0.mlp.gate_proj"))?.forward(&n_g)?;
    wg.synchronize()?;
    println!("gate md {}", maxdiff(&g_c, &g_g)?);
    let va = g_c.flatten_all()?.to_vec1::<f32>()?;
    let absmax = va.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let amin = va.iter().cloned().fold(f32::INFINITY, f32::min);
    let amax = va.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("gate range [{amin}, {amax}] absmax={absmax}");

    let neg_c = g_c.neg()?; let neg_g = g_g.neg()?;
    println!("neg md {}", maxdiff(&neg_c, &neg_g)?);
    let exp_c = neg_c.exp()?; let exp_g = neg_g.exp()?;
    println!("exp md {}", maxdiff(&exp_c, &exp_g)?);
    let ec = exp_c.flatten_all()?.to_vec1::<f32>()?;
    let eg = exp_g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut worst = 0f32; let mut wi = 0;
    for (i,(a,b)) in ec.iter().zip(eg.iter()).enumerate() {
        let d=(a-b).abs(); if d>worst { worst=d; wi=i; }
    }
    println!("exp worst i={wi} cpu={} gpu={} gate_cpu={}", ec[wi], eg[wi], va[wi]);

    let s_c = silu(&g_c)?; let s_g = silu(&g_g)?;
    println!("silu md {}", maxdiff(&s_c, &s_g)?);
    let sc = s_c.flatten_all()?.to_vec1::<f32>()?;
    let sg = s_g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    worst=0.0; wi=0;
    for (i,(a,b)) in sc.iter().zip(sg.iter()).enumerate() {
        let d=(a-b).abs(); if d>worst { worst=d; wi=i; }
    }
    println!("silu worst i={wi} cpu={} gpu={} gate={}", sc[wi], sg[wi], va[wi]);

    // Test exp on large positive (neg of large negative gate)
    // If gate is -10, neg is +10, exp(10) is large
    let big_pos: Vec<f32> = (-20..40).map(|i| i as f32 * 0.5).collect();
    let t_c = Tensor::from_vec(big_pos.clone(), big_pos.len(), &cpu)?;
    let t_g = t_c.to_device(&wg)?;
    let e_c = t_c.exp()?; let e_g = t_g.exp()?;
    println!("exp sweep md {}", maxdiff(&e_c, &e_g)?);
    let ec = e_c.flatten_all()?.to_vec1::<f32>()?;
    let eg = e_g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    for i in 0..big_pos.len() {
        let d = (ec[i]-eg[i]).abs();
        if d > 1.0 || ec[i].is_nan() || eg[i].is_nan() || ec[i].is_infinite() != eg[i].is_infinite() {
            println!("  x={} cpu={} gpu={} d={}", big_pos[i], ec[i], eg[i], d);
        }
    }
    Ok(())
}
