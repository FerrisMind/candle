use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(va.iter().zip(vb.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max))
}
fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(candle::D::Minus1)?;
    let xs1 = xs.narrow(candle::D::Minus1, 0, last/2)?.contiguous()?;
    let xs2 = xs.narrow(candle::D::Minus1, last/2, last-last/2)?.contiguous()?;
    Tensor::cat(&[&xs2.neg()?, &xs1], candle::D::Minus1)
}
fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q = q.contiguous()?; let k = k.contiguous()?;
    let cos = cos.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let sin = sin.contiguous()?.unsqueeze(candle::D::Minus2)?;
    let qe = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
    let ke = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;
    Ok((qe, ke))
}

#[test]
fn replay() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    // CPU stages for inputs
    let pe = 3*2*16*16;
    let mut data = vec![0f32; 64*pe];
    let mut s = 0xC0FFEEu64|1;
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (f32::from_bits(((s>>41) as u32)|0x3f800000)-1.0)*2.0-1.0;
    }
    let xs = Tensor::from_vec(data, (64, pe), &cpu)?;
    let g = Tensor::from_vec(vec![1u32,8,8], (1,3), &cpu)?;
    let m = Qwen3VLModel::new(&cfg, unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?})?;
    let st = m.forward_vision_debug_stages(&xs, &g)?;
    let norm1 = st.iter().find(|(n,_)| n=="b0_norm1").unwrap().1.clone();
    let cos = st.iter().find(|(n,_)| n=="rope_cos").unwrap().1.clone();
    // sin from cos path - recompute on both devices from same cos source is wrong
    // Get sin by re-running rot on GPU - instead load model and get stages for sin
    // Recompute emb cos/sin from rot_pos path on both via stages - we only exported cos
    // Export sin too by comparing attention replay with CPU sin derived from stages

    // Load linears on both devices
    let load_lin = |dev: &Device, path: &str, i: usize, o: usize| -> Result<Linear> {
        let vb = unsafe{VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, dev)?};
        linear(i, o, vb.pp(path))
    };
    let qkv_c = load_lin(&cpu, "model.visual.blocks.0.attn.qkv", 1024, 3072)?;
    let qkv_v = load_lin(&vk, "model.visual.blocks.0.attn.qkv", 1024, 3072)?;
    let proj_c = load_lin(&cpu, "model.visual.blocks.0.attn.proj", 1024, 1024)?;
    let proj_v = load_lin(&vk, "model.visual.blocks.0.attn.proj", 1024, 1024)?;

    // Need sin - recompute from model on cpu via rot_pos by re-exporting
    // Hack: get sin from GPU stages by adding rope_sin to debug - for now recompute
    // from cos? no. Add rope_sin export quickly.

    let hs_c = qkv_c.forward(&norm1)?;
    let hs_v = qkv_v.forward(&norm1.to_device(&vk)?)?;
    println!("qkv {}", maxdiff(&hs_v, &hs_c)?);

    let qkv_c = hs_c.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let qkv_v = hs_v.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let mut qc = qkv_c.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let mut kc = qkv_c.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let mut vc = qkv_c.i(2)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let mut qv = qkv_v.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let mut kv = qkv_v.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let mut vv = qkv_v.i(2)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    println!("q pre-rope {}", maxdiff(&qv, &qc)?);

    // Get sin: reconstruct from emb
    // Use CPU stages rope - recompute sin/cos from same rot_pos_emb logic by reading cos from stage
    // We'll get sin by re-running on CPU only and uploading
    // Actually export: read forward_debug - add rope_sin in a quick path
    // For this test: compute emb on CPU from model by calling stages - we need sin.
    // Use cos from stage and recompute sin on both from identical emb path via second stage field
    // Simpler: skip rope and compare attn without rope
    let qc2 = qc.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kc2 = kc.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vc2 = vc.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let att_c = (qc2.matmul(&kc2.transpose(2,3)?.contiguous()?)? / 8.0)?;
    let att_c = candle_nn::ops::softmax_last_dim(&att_c)?;
    let o_c = att_c.matmul(&vc2.contiguous()?)?.squeeze(0)?.transpose(0,1)?.contiguous()?.reshape((64, 1024))?;

    let qv2 = qv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let kv2 = kv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let vv2 = vv.transpose(0,1)?.contiguous()?.unsqueeze(0)?;
    let att_v = (qv2.matmul(&kv2.transpose(2,3)?.contiguous()?)? / 8.0)?;
    let att_v = candle_nn::ops::softmax_last_dim(&att_v)?;
    let o_v = att_v.matmul(&vv2.contiguous()?)?.squeeze(0)?.transpose(0,1)?.contiguous()?.reshape((64, 1024))?;
    vk.synchronize()?;
    println!("attn no rope {}", maxdiff(&o_v, &o_c)?);

    let out_c = proj_c.forward(&o_c)?;
    let out_v = proj_v.forward(&o_v)?;
    println!("proj no rope {}", maxdiff(&out_v, &out_c)?);
    Ok(())
}
