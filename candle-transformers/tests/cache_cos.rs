use candle::{Device, Result, Tensor, DType};
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
fn cache_cos() -> Result<()> {
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let mk = |device: &Device| -> Result<llama2_c::Cache> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        println!("device {:?} dim {} heads {} layers {} seq {}", device, config.dim, config.n_heads, config.n_layers, config.seq_len);
        llama2_c::Cache::new(false, &config, vb.pp("rot"))
    };
    let c_c = mk(&cpu)?;
    let c_g = mk(&wg)?;
    println!("cos {}", maxdiff(&c_c.cos, &c_g.cos.to_device(&cpu)?)?);
    println!("sin {}", maxdiff(&c_c.sin, &c_g.sin.to_device(&cpu)?)?);
    println!("cos shape {:?}", c_c.cos.dims());
    // also check freq tensors if present
    Ok(())
}
