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
fn upload_silu() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let mut file = File::open(&model_path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &cpu)?;
    let vb = weights.var_builder(&config, &cpu)?;
    let emb = candle_nn::embedding(config.vocab_size, config.dim, vb.pp("model.embed_tokens"))?;
    let ids = [1u32,13,42,7,19,5];
    let x = emb.forward(&Tensor::from_slice(&ids,(1,6),&cpu)?)?;
    let rms = candle_nn::rms_norm(config.dim, config.norm_eps, vb.pp("model.layers.0.post_attention_layernorm"))?;
    let n = rms.forward(&x)?;
    let g_c = candle_nn::linear_no_bias(config.dim, config.hidden_dim, vb.pp("model.layers.0.mlp.gate_proj"))?.forward(&n)?;
    let u_c = candle_nn::linear_no_bias(config.dim, config.hidden_dim, vb.pp("model.layers.0.mlp.up_proj"))?.forward(&n)?;
    let s_c = silu(&g_c)?;
    let mul_c = (&s_c * &u_c)?;

    // Upload and ops
    let g_g = g_c.to_device(&wg)?;
    wg.synchronize()?;
    println!("upload gate md {}", maxdiff(&g_c, &g_g)?);
    let s_g = silu(&g_g)?;
    wg.synchronize()?;
    println!("silu upload md {}", maxdiff(&s_c, &s_g)?);
    // step by step
    let neg = g_g.neg()?;
    wg.synchronize()?;
    println!("neg md {}", maxdiff(&g_c.neg()?, &neg)?);
    let exp = neg.exp()?;
    wg.synchronize()?;
    println!("exp md {}", maxdiff(&g_c.neg()?.exp()?, &exp)?);
    let den = (exp + 1.0)?;
    wg.synchronize()?;
    println!("den md {}", maxdiff(&(g_c.neg()?.exp()? + 1.0)?, &den)?);
    let s2 = (&g_g / &den)?;
    wg.synchronize()?;
    println!("div md {}", maxdiff(&s_c, &s2)?);

    let u_g = u_c.to_device(&wg)?;
    let mul_g = (&s_g * &u_g)?;
    wg.synchronize()?;
    println!("mul md {}", maxdiff(&mul_c, &mul_g)?);

    // Without intermediate sync
    let g2 = g_c.to_device(&wg)?;
    let u2 = u_c.to_device(&wg)?;
    let s2 = silu(&g2)?;
    let m2 = (&s2 * &u2)?;
    println!("no-sync-mul md {}", maxdiff(&mul_c, &m2)?);

    // Binary mul only after ensuring s2 is materialised by reading
    let _ = s2.flatten_all()?.to_vec1::<f32>()?; // force sync read
    let m3 = (&s2 * &u2)?;
    println!("after-read-mul md {}", maxdiff(&mul_c, &m3)?);
    Ok(())
}
