use candle::{Device, Result, Tensor};
use candle_nn::Module;
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

fn load(path: &std::path::Path, device: &Device) -> Result<(llama2_c::Llama, llama2_c::Cache)> {
    let mut file = File::open(path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
    let vb = weights.var_builder(&config, device)?;
    let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
    let model = llama2_c::Llama::load(vb, config)?;
    Ok((model, cache))
}

#[test]
fn llama_block0() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let (m_c, mut cache_c) = load(&model_path, &cpu)?;
    let (m_g, mut cache_g) = load(&model_path, &wg)?;
    let ids = [1u32, 13, 42, 7, 19, 5];
    let ids_c = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_g = Tensor::from_slice(&ids, (1, ids.len()), &wg)?;
    // access via forward full is only public API - use intermediate by cloning path through one layer
    // instead compare after embedding using weight dump
    // Use full forward with n_layers temporarily? can't.
    // Compare cache cos/sin
    // Actually run full and also check if first token logits vs mean
    let y_c = m_c.forward(&ids_c, 0, &mut cache_c)?;
    let y_g = m_g.forward(&ids_g, 0, &mut cache_g)?;
    wg.synchronize()?;
    println!("full maxdiff {}", maxdiff(&y_c, &y_g)?);
    // disable kv cache path: Cache::new(false,...)
    let mut file = File::open(&model_path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &wg)?;
    let vb = weights.var_builder(&config, &wg)?;
    let mut cache_nc = llama2_c::Cache::new(false, &config, vb.pp("rot"))?;
    let m_nc = llama2_c::Llama::load(vb, config)?;
    let y_nc = m_nc.forward(&ids_g, 0, &mut cache_nc)?;
    wg.synchronize()?;
    println!("no-cache vs cpu maxdiff {}", maxdiff(&y_c, &y_nc)?);
    Ok(())
}
