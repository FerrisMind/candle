//! Stage-by-stage Qwen3-VL vision tower on Vulkan.
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(va
        .iter()
        .zip(vb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max))
}

fn dir() -> PathBuf {
    PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking")
}

fn patches(device: &Device, n: usize, pe: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n * pe);
    let mut state = 0xC0FFEEu64 | 1;
    for _ in 0..(n * pe) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 41) as u32) | 0x3f80_0000;
        data.push((f32::from_bits(bits) - 1.0) * 2.0 - 1.0);
    }
    Tensor::from_vec(data, (n, pe), device)
}

#[test]
fn vl_stage_vulkan() -> Result<()> {
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let dir = dir();
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let vc = &cfg.vision_config;
    let pe = vc.in_chans * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    let (gt, gh, gw) = (1usize, 8, 8);
    let n = gt * gh * gw;
    let xs_c = patches(&cpu, n, pe)?;
    let xs_v = xs_c.to_device(&vk)?;
    let grid_c = Tensor::from_vec(vec![gt as u32, gh as u32, gw as u32], (1, 3), &cpu)?;
    let grid_v = grid_c.to_device(&vk)?;

    let m_c = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?
        };
        Qwen3VLModel::new(&cfg, vb)?
    };
    let m_v = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &vk)?
        };
        Qwen3VLModel::new(&cfg, vb)?
    };

    // Full vision
    let (e_c, d_c) = m_c.forward_vision_only(&xs_c, &grid_c)?;
    let (e_v, d_v) = m_v.forward_vision_only(&xs_v, &grid_v)?;
    vk.synchronize()?;
    println!("full emb maxdiff {}", maxdiff(&e_v, &e_c)?);
    for i in 0..d_c.len() {
        println!("deep[{i}] {}", maxdiff(&d_v[i], &d_c[i])?);
    }

    // Isolated ops used in vision
    // conv2d sanity
    let w = Tensor::randn(0f32, 0.1, (8, 3, 16, 16), &cpu)?;
    let x = Tensor::randn(0f32, 0.1, (4, 3, 32, 32), &cpu)?;
    let y_c = x.conv2d(&w, 0, 16, 1, 1)?;
    let y_v = x.to_device(&vk)?.conv2d(&w.to_device(&vk)?, 0, 16, 1, 1)?;
    vk.synchronize()?;
    println!("conv2d stride16 {}", maxdiff(&y_v, &y_c)?);

    // layer_norm
    let ln_x = Tensor::randn(0f32, 1.0, (64, 1024), &cpu)?;
    let wln = Tensor::randn(0f32, 0.1, 1024, &cpu)?;
    let bln = Tensor::randn(0f32, 0.1, 1024, &cpu)?;
    let n_c = candle_nn::ops::layer_norm(&ln_x, &wln, &bln, 1e-6)?;
    let n_v = candle_nn::ops::layer_norm(
        &ln_x.to_device(&vk)?,
        &wln.to_device(&vk)?,
        &bln.to_device(&vk)?,
        1e-6,
    )?;
    vk.synchronize()?;
    println!("layernorm {}", maxdiff(&n_v, &n_c)?);

    // matmul (16,16,64)x(16,64,64) heads style - seq=64 head=64
    let q = Tensor::randn(0f32, 0.1, (1, 16, 64, 64), &cpu)?;
    let k = Tensor::randn(0f32, 0.1, (1, 16, 64, 64), &cpu)?;
    let v = Tensor::randn(0f32, 0.1, (1, 16, 64, 64), &cpu)?;
    let att_c = (q.matmul(&k.t()?)? / 8.0)?;
    let att_v = (q.to_device(&vk)?.matmul(&k.to_device(&vk)?.t()?)? / 8.0)?;
    vk.synchronize()?;
    println!("attn scores {}", maxdiff(&att_v, &att_c)?);
    let s_c = candle_nn::ops::softmax_last_dim(&att_c)?;
    let s_v = candle_nn::ops::softmax_last_dim(&att_v)?;
    println!("softmax {}", maxdiff(&s_v, &s_c)?);
    let o_c = s_c.matmul(&v)?;
    let o_v = s_v.matmul(&v.to_device(&vk)?)?;
    vk.synchronize()?;
    println!("attn out {}", maxdiff(&o_v, &o_c)?);

    // index_select style pos
    let emb = Tensor::randn(0f32, 0.1, (2304, 1024), &cpu)?;
    let idx = Tensor::from_vec((0i64..64).collect::<Vec<_>>(), (64,), &cpu)?;
    let p_c = emb.index_select(&idx, 0)?;
    let p_v = emb
        .to_device(&vk)?
        .index_select(&idx.to_device(&vk)?, 0)?;
    vk.synchronize()?;
    println!("index_select {}", maxdiff(&p_v, &p_c)?);

    // gelu
    let g = Tensor::randn(0f32, 1.0, (64, 1024), &cpu)?;
    let g_c = g.gelu()?;
    let g_v = g.to_device(&vk)?.gelu()?;
    println!("gelu {}", maxdiff(&g_v, &g_c)?);

    Ok(())
}
