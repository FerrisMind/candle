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

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[test]
fn silu_and_layers() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let x = Tensor::randn(0f32, 1.0, (1, 6, 288), &cpu)?;
    let xg = x.to_device(&wg)?;
    let s_c = silu(&x)?;
    let s_g = silu(&xg)?;
    wg.synchronize()?;
    println!("silu {}", maxdiff(&s_c, &s_g)?);

    // full model layer-by-layer via forward is opaque; compare after 1 token seq_len=1 (no causal mask complexity)
    let hub = PathBuf::from(std::env::var("USERPROFILE").unwrap()).join(".cache/huggingface/hub");
    let model_path = find(&hub, "stories15M.bin").unwrap();
    let load = |device: &Device| -> Result<(llama2_c::Llama, llama2_c::Cache)> {
        let mut file = File::open(&model_path)?;
        let config = llama2_c::Config::from_reader(&mut file)?;
        let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, device)?;
        let vb = weights.var_builder(&config, device)?;
        let cache = llama2_c::Cache::new(false, &config, vb.pp("rot"))?;
        Ok((llama2_c::Llama::load(vb, config)?, cache))
    };
    let (m_c, mut c_c) = load(&cpu)?;
    let (m_g, mut c_g) = load(&wg)?;
    let ids = [1u32];
    let y_c = m_c.forward(&Tensor::from_slice(&ids, (1,1), &cpu)?, 0, &mut c_c)?;
    let y_g = m_g.forward(&Tensor::from_slice(&ids, (1,1), &wg)?, 0, &mut c_g)?;
    wg.synchronize()?;
    println!("seq1 maxdiff {}", maxdiff(&y_c, &y_g)?);
    let vc = y_c.flatten_all()?.to_vec1::<f32>()?;
    let vg = y_g.flatten_all()?.to_vec1::<f32>()?;
    println!("seq1 cpu {:?}", &vc[..5]);
    println!("seq1 wg  {:?}", &vg[..5]);
    Ok(())
}
