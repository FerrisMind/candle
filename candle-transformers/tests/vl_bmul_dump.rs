use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::path::PathBuf;

#[test]
fn dump() -> Result<()> {
    let dir = PathBuf::from(r"C:\Users\PC\Documents\models\unsloth\Qwen3-VL-2B-Thinking");
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| candle::Error::msg(e.to_string()))?;
    let cpu = Device::Cpu;
    let vk = Device::new_vulkan(0)?;
    let pe = 3*2*16*16;
    let mut data = vec![0f32; 64*pe];
    let mut s = 0xC0FFEEu64|1;
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (f32::from_bits(((s>>41) as u32)|0x3f800000)-1.0)*2.0-1.0;
    }
    let xs = Tensor::from_vec(data, (64, pe), &cpu)?;
    let g = Tensor::from_vec(vec![1u32,8,8], (1,3), &cpu)?;
    let m = Qwen3VLModel::new(&cfg, unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)?
    })?;
    let st = m.forward_vision_debug_stages(&xs, &g)?;
    let norm1 = st.iter().find(|(n,_)| n=="b0_norm1").unwrap().1.clone();
    let cos = st.iter().find(|(n,_)| n=="rope_cos").unwrap().1.clone();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)? };
    let qkv = linear(1024, 3072, vb.pp("model.visual.blocks.0.attn.qkv"))?;
    let hs = qkv.forward(&norm1)?;
    let t = hs.reshape((64,3,16,64))?.permute((1,0,2,3))?.contiguous()?;
    let k = t.i(1)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;
    let q = t.i(0)?.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;

    let cos_u = cos.unsqueeze(1)?; // 64,1,64
    let yk_c = k.broadcast_mul(&cos_u)?;
    let yq_c = q.broadcast_mul(&cos_u)?;
    let yk_v = k.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    let yq_v = q.to_device(&vk)?.broadcast_mul(&cos.to_device(&vk)?.unsqueeze(1)?)?;
    vk.synchronize()?;

    let kc = k.flatten_all()?.to_vec1::<f32>()?;
    let qc = q.flatten_all()?.to_vec1::<f32>()?;
    let cc = cos.flatten_all()?.to_vec1::<f32>()?;
    let ykc = yk_c.flatten_all()?.to_vec1::<f32>()?;
    let ykv = yk_v.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let yqc = yq_c.flatten_all()?.to_vec1::<f32>()?;
    let yqv = yq_v.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    println!("k[0..4]={:?}", &kc[..4]);
    println!("q[0..4]={:?}", &qc[..4]);
    println!("cos[0..4]={:?}", &cc[..4]);
    // layout (64,16,64) index i,h,d -> i*1024 + h*64 + d
    // for i=0,h=0,d=0: k*cos = k[0]*cos[0]
    println!("expected k0*cos0={}", kc[0]*cc[0]);
    println!("cpu yk[0]={} vk yk[0]={}", ykc[0], ykv[0]);
    println!("cpu yq[0]={} vk yq[0]={}", yqc[0], yqv[0]);

    // find first big k mismatch
    let mut nbig = 0;
    for i in 0..ykc.len() {
        let d = (ykc[i]-ykv[i]).abs();
        if d > 0.1 {
            if nbig < 5 {
                let seq = i / 1024;
                let rem = i % 1024;
                let h = rem / 64;
                let ddim = rem % 64;
                // cos index: cos is (64,64) -> seq * 64 + ddim
                let cidx = seq * 64 + ddim;
                println!("mis i={i} seq={seq} h={h} d={ddim} k={} cos={} exp={} cpu={} vk={} ",
                    kc[i], cc[cidx], kc[i]*cc[cidx], ykc[i], ykv[i]);
            }
            nbig += 1;
        }
    }
    println!("num big mismatches k*cos: {nbig} / {}", ykc.len());

    // check if vk used wrong cos index: maybe treated cos as (64,16,64) contig?
    nbig = 0;
    for i in 0..5 {
        // if cos broadcast wrong: using cos[seq, h, d] with stride as if (64,16,64) from (64,64) flat
        println!("sample i={i}");
    }
    Ok(())
}
