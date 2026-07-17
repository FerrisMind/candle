use candle::{Device, Result, Tensor, D, DType};
use candle_nn::{ops, Module};
use candle_transformers::models::{llama2_c, llama2_c_weights};
use std::fs::File;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn report(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    println!("{label:24} md={:.4e}", maxdiff(a,b)?); Ok(())
}
fn find(p: &std::path::Path, name: &str) -> Option<PathBuf> {
    if p.is_file() && p.file_name()?.to_str()? == name { return Some(p.to_path_buf()); }
    if p.is_dir() { for e in std::fs::read_dir(p).ok()? { if let Ok(e)=e { if let Some(f)=find(&e.path(), name){return Some(f);} } } }
    None
}
fn silu(xs: &Tensor) -> Result<Tensor> { xs / (xs.neg()?.exp()? + 1.0)? }
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

#[test]
fn after_attn() -> Result<()> {
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
    let head_dim = cfg.dim / cfg.n_heads;
    let seq_len = 6usize;
    let ids = [1u32,13,42,7,19,5];
    let emb_c = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_c.pp("model.embed_tokens"))?;
    let emb_g = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_g.pp("model.embed_tokens"))?;
    let x0_c = emb_c.forward(&Tensor::from_slice(&ids,(1,seq_len),&cpu)?)?;
    let x0_g = emb_g.forward(&Tensor::from_slice(&ids,(1,seq_len),&wg)?)?;
    // build post-attn residual same as before (abbreviated using real model path via one block ops)
    // Use CPU post-attn then upload to GPU fresh for MLP isolation
    let rms1_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp("model.layers.0.input_layernorm"))?;
    let n_c = rms1_c.forward(&x0_c)?;
    let q_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.q_proj"))?.forward(&n_c)?;
    let k_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.k_proj"))?.forward(&n_c)?;
    let v_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.v_proj"))?.forward(&n_c)?;
    let rope_in = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Result<Tensor> {
        let x = x.transpose(1,2)?.contiguous()?;
        candle_nn::rotary_emb::rope_slow(&x, cos, sin)?.transpose(1,2)?.contiguous()
    };
    let cos_c = cache_c.cos.narrow(0,0,seq_len)?.squeeze(2)?;
    let sin_c = cache_c.sin.narrow(0,0,seq_len)?.squeeze(2)?;
    let q_c = rope_in(&q_c.reshape((1,seq_len,cfg.n_heads,head_dim))?, &cos_c, &sin_c)?;
    let k_c = rope_in(&k_c.reshape((1,seq_len,cfg.n_heads,head_dim))?, &cos_c, &sin_c)?;
    let v_c = v_c.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let q_c = q_c.transpose(1,2)?.contiguous()?;
    let k_c = k_c.transpose(1,2)?.contiguous()?;
    let v_c = v_c.transpose(1,2)?.contiguous()?;
    let att_c = (q_c.matmul(&k_c.t()?)? / (head_dim as f64).sqrt())?;
    let mut m = vec![0u8; seq_len*seq_len];
    for i in 0..seq_len { for j in 0..seq_len { if j>i { m[i*seq_len+j]=1; } } }
    let mask_c = Tensor::from_slice(&m, (1,1,seq_len,seq_len), &cpu)?.broadcast_as(att_c.shape())?;
    let att_c = masked_fill(&att_c, &mask_c, f32::NEG_INFINITY)?;
    let att_c = ops::softmax(&att_c, D::Minus1)?;
    let y_c = att_c.matmul(&v_c.contiguous()?)?.transpose(1,2)?.reshape(&[1,seq_len,cfg.dim])?;
    let o_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.o_proj"))?.forward(&y_c)?;
    let post_c = (&o_c + &x0_c)?;
    // fresh GPU copy of post-attn
    let post_g = post_c.to_device(&wg)?;
    wg.synchronize()?;
    report("post fresh", &post_c, &post_g)?;
    // MLP on fresh
    let rms2_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp("model.layers.0.post_attention_layernorm"))?;
    let rms2_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp("model.layers.0.post_attention_layernorm"))?;
    let n2_c = rms2_c.forward(&post_c)?;
    let n2_g = rms2_g.forward(&post_g)?;
    report("rms2", &n2_c, &n2_g)?;
    let g_c = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_c.pp("model.layers.0.mlp.gate_proj"))?.forward(&n2_c)?;
    let g_g = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp("model.layers.0.mlp.gate_proj"))?.forward(&n2_g)?;
    report("gate", &g_c, &g_g)?;
    let u_c = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_c.pp("model.layers.0.mlp.up_proj"))?.forward(&n2_c)?;
    let u_g = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp("model.layers.0.mlp.up_proj"))?.forward(&n2_g)?;
    report("up", &u_c, &u_g)?;
    let m_c = (silu(&g_c)? * u_c.clone())?;
    let m_g = (silu(&g_g)? * u_g)?;
    report("mul", &m_c, &m_g)?;
    let d_c = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.dim, vb_c.pp("model.layers.0.mlp.down_proj"))?.forward(&m_c)?;
    let d_g = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.dim, vb_g.pp("model.layers.0.mlp.down_proj"))?.forward(&m_g)?;
    report("down", &d_c, &d_g)?;
    let out_c = (&d_c + &post_c)?;
    let out_g = (&d_g + &post_g)?;
    report("block0", &out_c, &out_g)?;

    // Now: continuous GPU path without intermediate CPU (same as t6full L0)
    let rms1_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp("model.layers.0.input_layernorm"))?;
    let n_g = rms1_g.forward(&x0_g)?;
    let q_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.q_proj"))?.forward(&n_g)?;
    let k_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.k_proj"))?.forward(&n_g)?;
    let v_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.v_proj"))?.forward(&n_g)?;
    let cos_g = cache_g.cos.narrow(0,0,seq_len)?.squeeze(2)?;
    let sin_g = cache_g.sin.narrow(0,0,seq_len)?.squeeze(2)?;
    let q_g = rope_in(&q_g.reshape((1,seq_len,cfg.n_heads,head_dim))?, &cos_g, &sin_g)?;
    let k_g = rope_in(&k_g.reshape((1,seq_len,cfg.n_heads,head_dim))?, &cos_g, &sin_g)?;
    let v_g = v_g.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let q_g = q_g.transpose(1,2)?.contiguous()?;
    let k_g = k_g.transpose(1,2)?.contiguous()?;
    let v_g = v_g.transpose(1,2)?.contiguous()?;
    let att_g = (q_g.matmul(&k_g.t()?)? / (head_dim as f64).sqrt())?;
    let mask_g = mask_c.to_device(&wg)?;
    let att_g = masked_fill(&att_g, &mask_g, f32::NEG_INFINITY)?;
    let att_g = ops::softmax(&att_g, D::Minus1)?;
    let y_g = att_g.matmul(&v_g.contiguous()?)?.transpose(1,2)?.reshape(&[1,seq_len,cfg.dim])?;
    let o_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.o_proj"))?.forward(&y_g)?;
    let post_g2 = (&o_g + &x0_g)?;
    report("post cont", &post_c, &post_g2)?;
    let n2_g2 = rms2_g.forward(&post_g2)?;
    report("rms2 cont", &n2_c, &n2_g2)?;
    let g_g2 = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp("model.layers.0.mlp.gate_proj"))?.forward(&n2_g2)?;
    report("gate cont", &g_c, &g_g2)?;
    let u_g2 = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp("model.layers.0.mlp.up_proj"))?.forward(&n2_g2)?;
    report("up cont", &u_c, &u_g2)?;
    let m_g2 = (silu(&g_g2)? * u_g2)?;
    report("mul cont", &m_c, &m_g2)?;
    let d_g2 = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.dim, vb_g.pp("model.layers.0.mlp.down_proj"))?.forward(&m_g2)?;
    report("down cont", &d_c, &d_g2)?;
    Ok(())
}

