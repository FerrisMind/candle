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
fn report(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let md = maxdiff(a,b)?;
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    println!("{label:16} md={md:.4e} cpu0={} gpu0={}", va[0], vb[0]);
    Ok(())
}
fn find(p: &std::path::Path, name: &str) -> Option<PathBuf> {
    if p.is_file() && p.file_name()?.to_str()? == name { return Some(p.to_path_buf()); }
    if p.is_dir() { for e in std::fs::read_dir(p).ok()? { if let Ok(e)=e { if let Some(f)=find(&e.path(), name){return Some(f);} } } }
    None
}
fn silu(xs: &Tensor) -> Result<Tensor> { xs / (xs.neg()?.exp()? + 1.0)? }

#[test]
fn mlp() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    // Use random residual-like input matching post-attn scale
    let x_c = Tensor::randn(0f32, 0.1, (1, 6, 288), &cpu)?;
    let x_g = x_c.to_device(&wg)?;
    let mut file = File::open(&model_path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights_c = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &cpu)?;
    // reopen
    let mut file = File::open(&model_path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights_g = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &wg)?;
    let vb_c = weights_c.var_builder(&config, &cpu)?;
    let vb_g = weights_g.var_builder(&config, &wg)?;
    let dim = config.dim;
    let hd = config.hidden_dim;
    let eps = config.norm_eps;
    let rms_c = candle_nn::rms_norm(dim, eps, vb_c.pp("model.layers.0.post_attention_layernorm"))?;
    let rms_g = candle_nn::rms_norm(dim, eps, vb_g.pp("model.layers.0.post_attention_layernorm"))?;
    let n_c = rms_c.forward(&x_c)?;
    let n_g = rms_g.forward(&x_g)?;
    report("rms2", &n_c, &n_g)?;
    let g_c = candle_nn::linear_no_bias(dim, hd, vb_c.pp("model.layers.0.mlp.gate_proj"))?.forward(&n_c)?;
    let g_g = candle_nn::linear_no_bias(dim, hd, vb_g.pp("model.layers.0.mlp.gate_proj"))?.forward(&n_g)?;
    report("gate", &g_c, &g_g)?;
    let u_c = candle_nn::linear_no_bias(dim, hd, vb_c.pp("model.layers.0.mlp.up_proj"))?.forward(&n_c)?;
    let u_g = candle_nn::linear_no_bias(dim, hd, vb_g.pp("model.layers.0.mlp.up_proj"))?.forward(&n_g)?;
    report("up", &u_c, &u_g)?;
    // silu pieces
    let neg_c = g_c.neg()?;
    let neg_g = g_g.neg()?;
    report("neg", &neg_c, &neg_g)?;
    let exp_c = neg_c.exp()?;
    let exp_g = neg_g.exp()?;
    report("exp", &exp_c, &exp_g)?;
    let den_c = (exp_c + 1.0)?;
    let den_g = (exp_g + 1.0)?;
    report("den", &den_c, &den_g)?;
    let s_c = (&g_c / &den_c)?;
    let s_g = (&g_g / &den_g)?;
    report("silu", &s_c, &s_g)?;
    let m_c = (&s_c * &u_c)?;
    let m_g = (&s_g * &u_g)?;
    report("silu*up", &m_c, &m_g)?;
    let d_c = candle_nn::linear_no_bias(hd, dim, vb_c.pp("model.layers.0.mlp.down_proj"))?.forward(&m_c)?;
    let d_g = candle_nn::linear_no_bias(hd, dim, vb_g.pp("model.layers.0.mlp.down_proj"))?.forward(&m_g)?;
    report("down", &d_c, &d_g)?;
    // also test silu of large values
    let big = Tensor::from_vec(vec![20.0f32, -20.0, 0.0, 5.0], 4, &cpu)?;
    let bigg = big.to_device(&wg)?;
    report("silu_big", &silu(&big)?, &silu(&bigg)?)?;
    // exp of large negative
    let t = Tensor::from_vec(vec![-20.0f32, -50.0, 0.0, 10.0], 4, &cpu)?;
    let tg = t.to_device(&wg)?;
    report("exp_vals", &t.exp()?, &tg.exp()?)?;
    Ok(())
}
