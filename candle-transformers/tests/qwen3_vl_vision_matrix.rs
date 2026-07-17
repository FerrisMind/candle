//! Qwen3-VL-2B vision tower parity (backend check; model code = upstream candle).
//!
//! Local: C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking
//! or CANDLE_QWEN3_VL_DIR.

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLVisionModel};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn vl_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CANDLE_QWEN3_VL_DIR") {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    if root.join("model.safetensors").is_file() {
        return root;
    }
    root
}

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if va.len() != vb.len() {
        candle::bail!("size mismatch {} vs {}", va.len(), vb.len());
    }
    Ok(va
        .iter()
        .zip(vb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max))
}

fn make_patches(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    in_chans: usize,
    t_patch: usize,
    patch: usize,
    device: &Device,
    seed: u64,
) -> Result<(Tensor, Tensor)> {
    let n = grid_t * grid_h * grid_w;
    let patch_elems = in_chans * t_patch * patch * patch;
    let mut data = Vec::with_capacity(n * patch_elems);
    let mut state = seed | 1;
    for _ in 0..(n * patch_elems) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 41) as u32) | 0x3f80_0000;
        let u = f32::from_bits(bits) - 1.0;
        data.push(u * 2.0 - 1.0);
    }
    let xs = Tensor::from_vec(data, (n, patch_elems), device)?;
    let grid = Tensor::from_vec(
        vec![grid_t as u32, grid_h as u32, grid_w as u32],
        (1, 3),
        device,
    )?;
    Ok((xs, grid))
}

fn load_cfg(dir: &Path) -> Result<Config> {
    let s = std::fs::read_to_string(dir.join("config.json"))?;
    serde_json::from_str(&s).map_err(|e| candle::Error::msg(format!("VL config: {e}")))
}

fn load_vision(dir: &Path, device: &Device, dtype: DType) -> Result<Qwen3VLVisionModel> {
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        candle::bail!("missing {weights:?}");
    }
    let cfg = load_cfg(dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
    Qwen3VLVisionModel::new(&cfg.vision_config, vb.pp("model").pp("visual"))
}

fn run_vision_parity(device: Device, label: &str) -> Result<()> {
    let dir = vl_dir();
    println!("=== Qwen3-VL vision parity on {label} dir={dir:?} ===");
    let cfg = load_cfg(&dir)?;
    let vc = &cfg.vision_config;
    let (gt, gh, gw) = (1usize, 8, 8);
    let (xs_cpu, grid_cpu) = make_patches(
        gt,
        gh,
        gw,
        vc.in_chans,
        vc.temporal_patch_size,
        vc.patch_size,
        &Device::Cpu,
        0xC0FFEE,
    )?;
    let xs_dev = xs_cpu.to_device(&device)?;
    let grid_dev = grid_cpu.to_device(&device)?;

    let cpu_dtype = DType::F32;
    let gpu_dtype = if device.is_wgpu() {
        DType::F16
    } else {
        DType::F32
    };
    let tol = if device.is_wgpu() { 1.0 } else { 5e-2 };

    let (cpu_emb, cpu_deep) = {
        let t0 = Instant::now();
        let model = load_vision(&dir, &Device::Cpu, cpu_dtype)?;
        println!("{label}: CPU vision load {:.2?}", t0.elapsed());
        let t1 = Instant::now();
        let out = model.forward(&xs_cpu, &grid_cpu)?;
        println!(
            "{label}: CPU vision {:.2?} emb={:?} deep={}",
            t1.elapsed(),
            out.0.dims(),
            out.1.len()
        );
        out
    };

    let (dev_emb, dev_deep) = {
        let t2 = Instant::now();
        let model = load_vision(&dir, &device, gpu_dtype)?;
        println!("{label}: GPU vision load {:.2?} dtype={gpu_dtype:?}", t2.elapsed());
        let t3 = Instant::now();
        let xs = xs_dev.to_dtype(gpu_dtype)?;
        let out = model.forward(&xs, &grid_dev)?;
        device.synchronize()?;
        println!(
            "{label}: GPU vision {:.2?} emb={:?} deep={}",
            t3.elapsed(),
            out.0.dims(),
            out.1.len()
        );
        out
    };

    let md = maxdiff(&dev_emb, &cpu_emb)?;
    println!("{label}: vision emb maxdiff={md:.6e} (tol={tol})");
    if cpu_deep.len() != dev_deep.len() {
        candle::bail!("deepstack len {} vs {}", cpu_deep.len(), dev_deep.len());
    }
    for (i, (c, d)) in cpu_deep.iter().zip(dev_deep.iter()).enumerate() {
        let mdi = maxdiff(d, c)?;
        println!("{label}: deepstack[{i}] maxdiff={mdi:.6e}");
        if mdi > tol {
            candle::bail!("{label}: deepstack[{i}] maxdiff {mdi} exceeds {tol}");
        }
    }
    if md > tol {
        candle::bail!("{label}: vision emb maxdiff {md} exceeds {tol}");
    }
    println!("{label}: VISION PARITY PASS");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual VL vision matrix"]
fn qwen3_vl_vision_cuda() -> Result<()> {
    run_vision_parity(Device::new_cuda(0)?, "cuda")
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "manual VL vision matrix"]
fn qwen3_vl_vision_vulkan() -> Result<()> {
    run_vision_parity(Device::new_vulkan(0)?, "vulkan")
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "manual VL vision matrix"]
fn qwen3_vl_vision_wgpu() -> Result<()> {
    run_vision_parity(Device::new_wgpu(0)?, "wgpu")
}
