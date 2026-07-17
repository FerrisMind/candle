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
    println!("{label:24} maxdiff={:.6e}", maxdiff(a,b)?);
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
fn t6full() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let load_vb = |device: &Device| -> Result<(candle_nn::VarBuilder, llama2_c::Config, llama2_c::Cache)> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
        Ok((vb, config, cache))
    };
    let (vb_c, cfg, mut cache_c) = load_vb(&cpu)?;
    let (vb_g, _, mut cache_g) = load_vb(&wg)?;
    let head_dim = cfg.dim / cfg.n_heads;
    let seq_len = 6usize;
    let ids = [1u32,13,42,7,19,5];
    let ids_c = Tensor::from_slice(&ids, (1, seq_len), &cpu)?;
    let ids_g = Tensor::from_slice(&ids, (1, seq_len), &wg)?;
    let emb_c = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_c.pp("model.embed_tokens"))?;
    let emb_g = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb_g.pp("model.embed_tokens"))?;
    let mut x_c = emb_c.forward(&ids_c)?;
    let mut x_g = emb_g.forward(&ids_g)?;
    report("embed", &x_c, &x_g)?;

    for layer in 0..cfg.n_layers {
        let residual_c = x_c.clone();
        let residual_g = x_g.clone();
        let rms1_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp(format!("model.layers.{layer}.input_layernorm")))?;
        let rms1_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp(format!("model.layers.{layer}.input_layernorm")))?;
        let n_c = rms1_c.forward(&x_c)?;
        let n_g = rms1_g.forward(&x_g)?;
        let q_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp(format!("model.layers.{layer}.self_attn.q_proj")))?.forward(&n_c)?;
        let q_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp(format!("model.layers.{layer}.self_attn.q_proj")))?.forward(&n_g)?;
        let k_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp(format!("model.layers.{layer}.self_attn.k_proj")))?.forward(&n_c)?;
        let k_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp(format!("model.layers.{layer}.self_attn.k_proj")))?.forward(&n_g)?;
        let v_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp(format!("model.layers.{layer}.self_attn.v_proj")))?.forward(&n_c)?;
        let v_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp(format!("model.layers.{layer}.self_attn.v_proj")))?.forward(&n_g)?;
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
        let mut v4_c = v_c.reshape((1,seq_len,cfg.n_heads,head_dim))?;
        let mut v4_g = v_g.reshape((1,seq_len,cfg.n_heads,head_dim))?;
        let mut k_c = rope_in(&k4_c, &cos_c, &sin_c)?;
        let mut k_g = rope_in(&k4_g, &cos_g, &sin_g)?;
        let q_c = rope_in(&q4_c, &cos_c, &sin_c)?;
        let q_g = rope_in(&q4_g, &cos_g, &sin_g)?;
        // kv cache
        if let Some((ck, cv)) = &cache_c.kvs[layer] {
            k_c = Tensor::cat(&[ck, &k_c], 1)?.contiguous()?;
            v4_c = Tensor::cat(&[cv, &v4_c], 1)?.contiguous()?;
        }
        cache_c.kvs[layer] = Some((k_c.clone(), v4_c.clone()));
        if let Some((ck, cv)) = &cache_g.kvs[layer] {
            k_g = Tensor::cat(&[ck, &k_g], 1)?.contiguous()?;
            v4_g = Tensor::cat(&[cv, &v4_g], 1)?.contiguous()?;
        }
        cache_g.kvs[layer] = Some((k_g.clone(), v4_g.clone()));
        let q_c = q_c.transpose(1,2)?.contiguous()?;
        let q_g = q_g.transpose(1,2)?.contiguous()?;
        let k_c = k_c.transpose(1,2)?.contiguous()?;
        let k_g = k_g.transpose(1,2)?.contiguous()?;
        let v_c = v4_c.transpose(1,2)?.contiguous()?;
        let v_g = v4_g.transpose(1,2)?.contiguous()?;
        let att_c = (q_c.matmul(&k_c.t()?)? / (head_dim as f64).sqrt())?;
        let att_g = (q_g.matmul(&k_g.t()?)? / (head_dim as f64).sqrt())?;
        let mut m = vec![0u8; seq_len*seq_len];
        for i in 0..seq_len { for j in 0..seq_len { if j>i { m[i*seq_len+j]=1; } } }
        let mask_c = Tensor::from_slice(&m, (1,1,seq_len,seq_len), &cpu)?.broadcast_as(att_c.shape())?;
        let mask_g = mask_c.to_device(&wg)?;
        let att_c = masked_fill(&att_c, &mask_c, f32::NEG_INFINITY)?;
        let att_g = masked_fill(&att_g, &mask_g, f32::NEG_INFINITY)?;
        let att_c = ops::softmax(&att_c, D::Minus1)?;
        let att_g = ops::softmax(&att_g, D::Minus1)?;
        let y_c = att_c.matmul(&v_c.contiguous()?)?;
        let y_g = att_g.matmul(&v_g.contiguous()?)?;
        let y_c = y_c.transpose(1,2)?.reshape(&[1, seq_len, cfg.dim])?;
        let y_g = y_g.transpose(1,2)?.reshape(&[1, seq_len, cfg.dim])?;
        let o_c = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_c.pp(format!("model.layers.{layer}.self_attn.o_proj")))?.forward(&y_c)?;
        let o_g = candle_nn::linear_no_bias(cfg.dim, cfg.dim, vb_g.pp(format!("model.layers.{layer}.self_attn.o_proj")))?.forward(&y_g)?;
        x_c = (&o_c + &residual_c)?;
        x_g = (&o_g + &residual_g)?;
        report(&format!("L{layer} post-attn"), &x_c, &x_g)?;
        let residual_c = x_c.clone();
        let residual_g = x_g.clone();
        let rms2_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp(format!("model.layers.{layer}.post_attention_layernorm")))?;
        let rms2_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp(format!("model.layers.{layer}.post_attention_layernorm")))?;
        let n2_c = rms2_c.forward(&x_c)?;
        let n2_g = rms2_g.forward(&x_g)?;
        let g_c = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_c.pp(format!("model.layers.{layer}.mlp.gate_proj")))?.forward(&n2_c)?;
        let g_g = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp(format!("model.layers.{layer}.mlp.gate_proj")))?.forward(&n2_g)?;
        let u_c = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_c.pp(format!("model.layers.{layer}.mlp.up_proj")))?.forward(&n2_c)?;
        let u_g = candle_nn::linear_no_bias(cfg.dim, cfg.hidden_dim, vb_g.pp(format!("model.layers.{layer}.mlp.up_proj")))?.forward(&n2_g)?;
        let m_c = (silu(&g_c)? * u_c)?;
        let m_g = (silu(&g_g)? * u_g)?;
        let d_c = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.dim, vb_c.pp(format!("model.layers.{layer}.mlp.down_proj")))?.forward(&m_c)?;
        let d_g = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.dim, vb_g.pp(format!("model.layers.{layer}.mlp.down_proj")))?.forward(&m_g)?;
        x_c = (&d_c + &residual_c)?;
        x_g = (&d_g + &residual_g)?;
        report(&format!("L{layer} out"), &x_c, &x_g)?;
        wg.synchronize()?;
    }
    let ln_c = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_c.pp("model.norm"))?;
    let ln_g = candle_nn::rms_norm(cfg.dim, cfg.norm_eps, vb_g.pp("model.norm"))?;
    let x_c = ln_c.forward(&x_c)?;
    let x_g = ln_g.forward(&x_g)?;
    let logits_c = candle_nn::linear_no_bias(cfg.dim, cfg.vocab_size, vb_c.pp("lm_head"))?.forward(&x_c)?;
    let logits_g = candle_nn::linear_no_bias(cfg.dim, cfg.vocab_size, vb_g.pp("lm_head"))?.forward(&x_g)?;
    report("logits", &logits_c, &logits_g)?;
    Ok(())
}
