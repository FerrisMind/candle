use candle::{Device, Result, Tensor};
use candle_transformers::models::{llama2_c, llama2_c_weights};
use std::fs::File;
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(candle::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_dtype(candle::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max))
}

fn load(path: &std::path::Path, device: &Device) -> Result<(llama2_c::Llama, llama2_c::Cache, llama2_c::Config)> {
    let mut file = File::open(path)?;
    let config = llama2_c::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
    let vb = weights.var_builder(&config, device)?;
    let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
    let model = llama2_c::Llama::load(vb, config.clone())?;
    Ok((model, cache, config))
}

#[test]
fn llama_stage() -> Result<()> {
    let path = PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/huggingface/hub");
    // find stories15M.bin
    fn find(p: &std::path::Path, name: &str) -> Option<PathBuf> {
        if p.is_file() && p.file_name()?.to_str()? == name { return Some(p.to_path_buf()); }
        if p.is_dir() {
            for e in std::fs::read_dir(p).ok()? {
                if let Ok(e) = e {
                    if let Some(f) = find(&e.path(), name) { return Some(f); }
                }
            }
        }
        None
    }
    let model_path = find(&path, "stories15M.bin").expect("stories15M.bin");
    println!("model {model_path:?}");
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let (m_c, mut cache_c, _) = load(&model_path, &cpu)?;
    let (m_g, mut cache_g, _) = load(&model_path, &wg)?;
    let ids = [1u32, 13, 42, 7, 19, 5];
    let ids_c = Tensor::from_slice(&ids, (1, ids.len()), &cpu)?;
    let ids_g = Tensor::from_slice(&ids, (1, ids.len()), &wg)?;
    let y_c = m_c.forward(&ids_c, 0, &mut cache_c)?;
    let y_g = m_g.forward(&ids_g, 0, &mut cache_g)?;
    wg.synchronize()?;
    let md = maxdiff(&y_c, &y_g)?;
    println!("full forward maxdiff {md}");
    let vc = y_c.flatten_all()?.to_vec1::<f32>()?;
    let vg = y_g.flatten_all()?.to_vec1::<f32>()?;
    println!("cpu first5 {:?}", &vc[..5]);
    println!("wg  first5 {:?}", &vg[..5]);
    // check if any non-finite
    println!("wg finite {}", vg.iter().all(|x| x.is_finite()));
    println!("wg absmax {}", vg.iter().map(|x| x.abs()).fold(0.0f32, f32::max));
    Ok(())
}
