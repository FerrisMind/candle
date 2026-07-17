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
    let md = maxdiff(a,b)?;
    println!("{label:24} maxdiff={md:.6e}");
    Ok(())
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
fn t6() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let load_vb = |device: &Device| -> Result<(candle_nn::VarBuilder, llama2_c::Config, llama2_c::Cache)> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        let cache = llama2_c::Cache::new(false, &config, vb.pp("rot"))?;
        Ok((vb, config, cache))
    };
    let (vb_c, cfg, cache_c) = load_vb(&cpu)?;
    let (vb_g, _, cache_g) = load_vb(&wg)?;
    let head_dim = cfg.dim / cfg.n_heads;
    let seq_len = 6usize;
    let ids = [1u32,13,42,7,19,5];
    let ids_c = Tensor::from_slice(&ids, (1, seq_len), &cpu)?;
    let ids_g = Tensor::from_slice(&ids, (1, seq_len), &wg)?;
    let emb_c = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_c.pp("model.embed_tokens"))?;
    let emb_g = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_g.pp("model.embed_tokens"))?;
    let x_c = emb_c.forward(&ids_c)?;
    let x_g = emb_g.forward(&ids_g)?;
    let rms1_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp("model.layers.0.input_layernorm"))?;
    let rms1_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp("model.layers.0.input_layernorm"))?;
    let n_c = rms1_c.forward(&x_c)?;
    let n_g = rms1_g.forward(&x_g)?;
    report("rms1", &n_c, &n_g)?;
    let q_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.q_proj"))?.forward(&n_c)?;
    let q_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.q_proj"))?.forward(&n_g)?;
    let k_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.k_proj"))?.forward(&n_c)?;
    let k_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.k_proj"))?.forward(&n_g)?;
    let v_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp("model.layers.0.self_attn.v_proj"))?.forward(&n_c)?;
    let v_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp("model.layers.0.self_attn.v_proj"))?.forward(&n_g)?;
    report("q", &q_c, &q_g)?;
    let rope_in = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Result<Tensor> {
        let x = x.transpose(1,2)?.contiguous()?;
        let rope = candle_nn::rotary_emb::rope_slow(&x, cos, sin)?;
        rope.transpose(1,2)?.contiguous()
    };
    let cos_c = cache_c.cos.narrow(0,0,seq_len)?.squeeze(2)?;
    let sin_c = cache_c.sin.narrow(0,0,seq_len)?.squeeze(2)?;
    let cos_g = cache_g.cos.narrow(0,0,seq_len)?.squeeze(2)?;
    let sin_g = cache_g.sin.narrow(0,0,seq_len)?.squeeze(2)?;
    let q4_c = q_c.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let q4_g = q_g.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let k4_c = k_c.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let k4_g = k_g.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let v4_c = v_c.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let v4_g = v_g.reshape((1,seq_len,cfg.n_heads,head_dim))?;
    let q_c = rope_in(&q4_c, &cos_c, &sin_c)?;
    let q_g = rope_in(&q4_g, &cos_g, &sin_g)?;
    let k_c = rope_in(&k4_c, &cos_c, &sin_c)?;
    let k_g = rope_in(&k4_g, &cos_g, &sin_g)?;
    report("q_rope", &q_c, &q_g)?;
    let q_c = q_c.transpose(1,2)?.contiguous()?;
    let q_g = q_g.transpose(1,2)?.contiguous()?;
    let k_c = k_c.transpose(1,2)?.contiguous()?;
    let k_g = k_g.transpose(1,2)?.contiguous()?;
    let v_c = v4_c.transpose(1,2)?.contiguous()?;
    let v_g = v4_g.transpose(1,2)?.contiguous()?;
    let att_c = (q_c.matmul(&k_c.t()?)? / (head_dim as f64).sqrt())?;
    let att_g = (q_g.matmul(&k_g.t()?)? / (head_dim as f64).sqrt())?;
    report("att", &att_c, &att_g)?;
    // causal mask like llama Cache::mask
    let mut m = vec![0u8; seq_len*seq_len];
    for i in 0..seq_len { for j in 0..seq_len { if j>i { m[i*seq_len+j]=1; } } }
    let mask_c = Tensor::from_slice(&m, (1,1,seq_len,seq_len), &cpu)?.broadcast_as(att_c.shape())?;
    let mask_g = mask_c.to_device(&wg)?;
    let att_c = masked_fill(&att_c, &mask_c, f32::NEG_INFINITY)?;
    let att_g = masked_fill(&att_g, &mask_g, f32::NEG_INFINITY)?;
    report("masked", &att_c, &att_g)?;
    let att_c = ops::softmax(&att_c, D::Minus1)?;
    let att_g = ops::softmax(&att_g, D::Minus1)?;
    report("softmax", &att_c, &att_g)?;
    let y_c = att_c.matmul(&v_c.contiguous()?)?;
    let y_g = att_g.matmul(&v_g.contiguous()?)?;
    report("attn_out", &y_c, &y_g)?;
    Ok(())
}
